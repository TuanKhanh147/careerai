import os, json, re, io, csv, datetime as dt
from urllib.parse import quote
from typing import Optional
import bcrypt
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, ForeignKey, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SECRET = os.getenv("SECRET_KEY", "dev-secret")
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db").replace("postgres://", "postgresql://")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
Session_ = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    pw = Column(String)
    role = Column(String, default="student")  # student | employer | admin
    active = Column(Boolean, default=True)
    skills = Column(Text, default="")
    goal = Column(String, default="")
    bio = Column(Text, default="")
    cv_name = Column(String, default="")
    cv_data = Column(LargeBinary)
    cv_text = Column(Text, default="")

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    employer_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    company = Column(String)
    kind = Column(String, default="Thực tập")
    skills = Column(Text, default="")
    description = Column(Text, default="")
    url = Column(String, default="")
    source = Column(String, default="")

class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    student_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="Đã nộp")
    note = Column(Text, default="")
    created = Column(String, default="")

Base.metadata.create_all(engine)

def db():
    s = Session_()
    try: yield s
    finally: s.close()

def hash_pw(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
def check_pw(p, h): return bcrypt.checkpw(p.encode(), h.encode())

def import_remote(s, n=40):
    """Nhập tin tuyển dụng thật từ API công khai Remotive."""
    import urllib.request, html
    try:
        req = urllib.request.Request(f"https://remotive.com/api/remote-jobs?category=software-dev&limit={n}", headers={"User-Agent": "CareerAI"})
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())["jobs"]
    except Exception as e:
        print("[IMPORT ERROR]", e); return 0
    have = {u for (u,) in s.query(Job.url)}
    kinds = {"full_time": "Toàn thời gian", "part_time": "Bán thời gian", "contract": "Hợp đồng", "internship": "Thực tập"}
    c = 0
    for x in data:
        if x["url"] in have: continue
        d = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", x.get("description", "")))).strip()[:700]
        s.add(Job(title=x["title"], company=x["company_name"], kind=kinds.get(x.get("job_type"), "Khác"),
                  skills=", ".join(t.lower() for t in x.get("tags", [])[:8]), description=d, url=x["url"], source="Remotive"))
        c += 1
    s.commit(); return c

def seed():
    s = Session_()
    if not s.query(User).filter_by(role="admin").first():
        s.add(User(email=os.getenv("ADMIN_EMAIL", "admin@example.com"), name="Admin",
                   pw=hash_pw(os.getenv("ADMIN_PASSWORD", "Admin@123")), role="admin"))
    if not s.query(Job).first() and not import_remote(s):
        for t, c, k, sk, d in [
            ("Thực tập sinh Backend Python", "TechVN", "Thực tập", "python, fastapi, sql, git", "Phát triển API cho hệ thống nội bộ."),
            ("Data Analyst Intern", "DataCo", "Thực tập", "sql, excel, power bi, python", "Phân tích dữ liệu bán hàng, dựng dashboard."),
            ("Frontend Developer", "WebLab", "Toàn thời gian", "javascript, react, css, html", "Xây dựng giao diện web."),
            ("AI/ML Engineer Junior", "AIFirst", "Toàn thời gian", "python, machine learning, pytorch, nlp", "Huấn luyện và triển khai mô hình NLP."),
            ("Digital Marketing Intern", "BrandX", "Thực tập", "seo, content, facebook ads, canva", "Chạy chiến dịch và viết nội dung."),
        ]:
            s.add(Job(title=t, company=c, kind=k, skills=sk, description=d))
    s.commit(); s.close()
seed()

app = FastAPI(title="CareerAI")
oauth = OAuth2PasswordBearer(tokenUrl="/api/login")

def make_token(u): return jwt.encode({"sub": str(u.id), "exp": dt.datetime.utcnow() + dt.timedelta(days=2)}, SECRET, "HS256")

def current(token: str = Depends(oauth), s: Session = Depends(db)) -> User:
    try:
        p = jwt.decode(token, SECRET, ["HS256"]); uid = int(p["sub"])
        if "pr" in p: raise ValueError
    except (JWTError, ValueError): raise HTTPException(401, "Phiên đăng nhập không hợp lệ")
    u = s.get(User, uid)
    if not u or not u.active: raise HTTPException(401, "Tài khoản không khả dụng")
    return u

def need(*roles):
    def dep(u: User = Depends(current)):
        if u.role not in roles: raise HTTPException(403, "Không đủ quyền")
        return u
    return dep

def pub(u): return {"id": u.id, "email": u.email, "name": u.name, "role": u.role, "active": u.active,
                    "skills": u.skills, "goal": u.goal, "bio": u.bio, "cv": u.cv_name or ""}

# ---------- AI (Claude) ----------
def ask_ai(system: str, prompt: str, as_json=False):
    gk, ak = os.getenv("GEMINI_API_KEY"), os.getenv("ANTHROPIC_API_KEY")
    prov = os.getenv("AI_PROVIDER") or ("gemini" if gk else "claude")
    if prov == "gemini":
        if not gk: raise HTTPException(503, "Chưa cấu hình GEMINI_API_KEY trên máy chủ")
        import urllib.request, urllib.error
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        body = json.dumps({"system_instruction": {"parts": [{"text": system}]},
                           "contents": [{"role": "user", "parts": [{"text": prompt}]}]}).encode()
        req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                     data=body, headers={"Content-Type": "application/json", "x-goog-api-key": gk})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code == 429: raise HTTPException(429, "Đã hết hạn mức miễn phí của AI, hãy thử lại sau ít phút")
            raise HTTPException(502, f"Gemini lỗi {e.code}: {e.read().decode()[:200]}")
        except Exception as e:
            raise HTTPException(502, f"Không gọi được Gemini: {e}")
        try: text = r["candidates"][0]["content"]["parts"][0]["text"]
        except Exception: raise HTTPException(502, "Gemini không trả về nội dung")
    else:
        if not ak: raise HTTPException(503, "Chưa cấu hình ANTHROPIC_API_KEY trên máy chủ")
        import anthropic
        r = anthropic.Anthropic(api_key=ak).messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"), max_tokens=1500, system=system,
            messages=[{"role": "user", "content": prompt}])
        text = r.content[0].text
    if as_json:
        m = re.search(r"\{.*\}", text, re.S)
        try: return json.loads(m.group(0))
        except Exception: return {"raw": text}
    return text

def profile_text(u): return f"Mục tiêu: {u.goal}\nKỹ năng: {u.skills}\nGiới thiệu: {u.bio}\nCV: {(u.cv_text or '')[:3000]}"

# ---------- Auth ----------
class Reg(BaseModel):
    email: EmailStr; name: str; password: str; role: str = "student"

@app.post("/api/register")
def register(d: Reg, s: Session = Depends(db)):
    if d.role not in ("student", "employer"): raise HTTPException(400, "Vai trò không hợp lệ")
    if len(d.password) < 6: raise HTTPException(400, "Mật khẩu tối thiểu 6 ký tự")
    if s.query(User).filter_by(email=d.email.lower()).first(): raise HTTPException(400, "Email đã tồn tại")
    u = User(email=d.email.lower(), name=d.name, pw=hash_pw(d.password), role=d.role)
    s.add(u); s.commit()
    return {"token": make_token(u), "user": pub(u)}

@app.post("/api/login")
def login(f: OAuth2PasswordRequestForm = Depends(), s: Session = Depends(db)):
    u = s.query(User).filter_by(email=f.username.lower()).first()
    if not u or not check_pw(f.password, u.pw): raise HTTPException(400, "Sai email hoặc mật khẩu")
    if not u.active: raise HTTPException(403, "Tài khoản đã bị khóa")
    return {"access_token": make_token(u), "token_type": "bearer", "user": pub(u)}

@app.get("/api/me")
def me(u: User = Depends(current)): return pub(u)

class Prof(BaseModel):
    name: str; skills: str = ""; goal: str = ""; bio: str = ""

@app.put("/api/me")
def update_me(d: Prof, u: User = Depends(current), s: Session = Depends(db)):
    u.name, u.skills, u.goal, u.bio = d.name, d.skills, d.goal, d.bio
    s.commit(); return pub(u)

# ---------- Jobs ----------
class JobIn(BaseModel):
    title: str; company: str; kind: str = "Thực tập"; skills: str = ""; description: str = ""

def jdict(j): return {"id": j.id, "title": j.title, "company": j.company, "kind": j.kind,
                      "skills": j.skills, "description": j.description, "employer_id": j.employer_id, "url": j.url, "source": j.source}

@app.get("/api/jobs")
def jobs(s: Session = Depends(db), u: User = Depends(current)):
    return [jdict(j) for j in s.query(Job).order_by(Job.id.desc())]

@app.post("/api/jobs")
def add_job(d: JobIn, u: User = Depends(need("employer", "admin")), s: Session = Depends(db)):
    j = Job(**d.model_dump(), employer_id=u.id); s.add(j); s.commit(); return jdict(j)

@app.delete("/api/jobs/{jid}")
def del_job(jid: int, u: User = Depends(need("employer", "admin")), s: Session = Depends(db)):
    j = s.get(Job, jid)
    if not j: raise HTTPException(404, "Không tìm thấy")
    if u.role != "admin" and j.employer_id != u.id: raise HTTPException(403, "Không đủ quyền")
    s.delete(j); s.commit(); return {"ok": True}

# ---------- ML matching (TF-IDF + cosine + skill overlap) ----------
def toks(t): return {x.strip().lower() for x in re.split(r"[,;\n]", t or "") if x.strip()}

def match(user_text, user_skills, jobs_):
    if not jobs_: return []
    docs = [f"{j.title} {j.skills} {j.description}" for j in jobs_]
    v = TfidfVectorizer().fit(docs + [user_text])
    sim = cosine_similarity(v.transform([user_text]), v.transform(docs))[0]
    out = []
    for j, sc in zip(jobs_, sim):
        req = toks(j.skills); have = user_skills & req
        overlap = len(have) / len(req) if req else 0
        score = round(100 * (0.4 * sc + 0.6 * overlap))
        out.append({**jdict(j), "score": score, "matched": sorted(have), "missing": sorted(req - user_skills)})
    return sorted(out, key=lambda x: -x["score"])

@app.get("/api/recommend")
def recommend(u: User = Depends(need("student")), s: Session = Depends(db)):
    return match(profile_text(u), toks(u.skills), s.query(Job).all())[:10]

@app.get("/api/candidates/{jid}")
def candidates(jid: int, u: User = Depends(need("employer", "admin")), s: Session = Depends(db)):
    j = s.get(Job, jid)
    if not j: raise HTTPException(404, "Không tìm thấy")
    req = toks(j.skills); res = []
    studs = s.query(User).filter_by(role="student", active=True).all()
    if studs:
        v = TfidfVectorizer().fit([f"{j.title} {j.skills} {j.description}"] + [profile_text(x) for x in studs])
        sims = cosine_similarity(v.transform([f"{j.title} {j.skills} {j.description}"]),
                                 v.transform([profile_text(x) for x in studs]))[0]
        for x, sc in zip(studs, sims):
            ov = len(toks(x.skills) & req) / len(req) if req else 0
            res.append({"name": x.name, "email": x.email, "skills": x.skills, "score": round(100 * (0.4 * sc + 0.6 * ov))})
    return sorted(res, key=lambda r: -r["score"])[:10]

# ---------- AI features ----------
class Txt(BaseModel):
    text: str = ""

SYS = "Bạn là cố vấn nghề nghiệp cho sinh viên Việt Nam. Trả lời bằng tiếng Việt, ngắn gọn, cụ thể, thực tế."

@app.post("/api/ai/cv-review")
def cv_review(d: Txt, u: User = Depends(need("student"))):
    return {"result": ask_ai(SYS, f"{profile_text(u)}\n\nNội dung CV/portfolio:\n{d.text}\n\nHãy chấm CV (thang 10), nêu điểm mạnh, điểm yếu và viết lại 3 gạch đầu dòng thành tích theo công thức hành động + kết quả đo lường.")}

@app.post("/api/ai/skill-gap")
def skill_gap(d: Txt, u: User = Depends(need("student"))):
    goal = d.text or u.goal
    return {"result": ask_ai(SYS, f"Hồ sơ:\n{profile_text(u)}\n\nMục tiêu nghề: {goal}\nPhân tích khoảng cách kỹ năng và lập lộ trình học 8 tuần (tuần, nội dung, tài nguyên miễn phí).")}

@app.post("/api/ai/trends")
def trends(u: User = Depends(current), s: Session = Depends(db)):
    from collections import Counter
    c = Counter(k for j in s.query(Job).all() for k in toks(j.skills))
    top = c.most_common(10)
    note = ask_ai(SYS, f"Kỹ năng được tuyển dụng nhiều nhất trên nền tảng: {top}. Nhận xét xu hướng thị trường IT/kinh doanh tại Việt Nam và đưa ra 3 lời khuyên.")
    return {"top_skills": top, "result": note}

class Chat(BaseModel):
    role_name: str; history: list = []; answer: str = ""

@app.post("/api/ai/interview")
def interview(d: Chat, u: User = Depends(need("student"))):
    sysm = (f"Bạn là nhà tuyển dụng đang phỏng vấn ứng viên cho vị trí {d.role_name}. Hỏi MỘT câu mỗi lượt. "
            "Sau khi ứng viên trả lời, nhận xét ngắn (điểm tốt, cần cải thiện, cách trả lời mẫu theo STAR) rồi hỏi câu tiếp theo. Tiếng Việt.")
    convo = "\n".join(f"{m['who']}: {m['text']}" for m in d.history)
    p = convo + (f"\nỨng viên: {d.answer}" if d.answer else "") or "Hãy bắt đầu bằng lời chào và câu hỏi đầu tiên."
    return {"result": ask_ai(sysm, p)}

# ---------- Admin ----------
@app.get("/api/admin/users")
def users(u: User = Depends(need("admin")), s: Session = Depends(db)):
    return [pub(x) for x in s.query(User).order_by(User.id)]

@app.post("/api/admin/users/{uid}/toggle")
def toggle(uid: int, u: User = Depends(need("admin")), s: Session = Depends(db)):
    x = s.get(User, uid)
    if not x or x.id == u.id: raise HTTPException(400, "Không thể thực hiện")
    x.active = not x.active; s.commit(); return pub(x)

@app.delete("/api/admin/users/{uid}")
def del_user(uid: int, u: User = Depends(need("admin")), s: Session = Depends(db)):
    x = s.get(User, uid)
    if not x or x.id == u.id: raise HTTPException(400, "Không thể thực hiện")
    s.delete(x); s.commit(); return {"ok": True}

@app.get("/api/admin/stats")
def stats(u: User = Depends(need("admin")), s: Session = Depends(db)):
    return {"students": s.query(User).filter_by(role="student").count(),
            "employers": s.query(User).filter_by(role="employer").count(), "jobs": s.query(Job).count()}

# ---------- CV ----------
def extract_text(name, data):
    n = name.lower()
    try:
        if n.endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        if n.endswith(".docx"):
            import docx
            return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
        return data.decode("utf-8", "ignore")
    except Exception:
        return ""

def cv_resp(x):
    if not x or not x.cv_data: raise HTTPException(404, "Chưa có CV")
    return Response(x.cv_data, media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename*=UTF-8\'\'" + quote(x.cv_name)})

@app.post("/api/me/cv")
async def up_cv(file: UploadFile = File(...), u: User = Depends(need("student")), s: Session = Depends(db)):
    data = await file.read()
    if len(data) > 5_000_000: raise HTTPException(400, "File tối đa 5MB")
    if not (file.filename or "").lower().endswith((".pdf", ".docx", ".txt")): raise HTTPException(400, "Chỉ nhận PDF, DOCX, TXT")
    u.cv_name, u.cv_data, u.cv_text = file.filename, data, extract_text(file.filename, data)[:20000]
    s.commit(); return pub(u)

@app.get("/api/me/cv")
def my_cv(u: User = Depends(current)): return cv_resp(u)

# ---------- Ứng tuyển & theo dõi ----------
STATUS = ["Đã nộp", "Đang xem xét", "Phỏng vấn", "Trúng tuyển", "Từ chối"]

@app.post("/api/apply/{jid}")
def apply(jid: int, d: Txt, u: User = Depends(need("student")), s: Session = Depends(db)):
    if not s.get(Job, jid): raise HTTPException(404, "Không tìm thấy tin")
    if not u.cv_data: raise HTTPException(400, "Hãy tải CV lên trong tab Hồ sơ trước khi ứng tuyển")
    if s.query(Application).filter_by(job_id=jid, student_id=u.id).first(): raise HTTPException(400, "Bạn đã ứng tuyển vị trí này")
    s.add(Application(job_id=jid, student_id=u.id, note=d.text, created=dt.datetime.utcnow().strftime("%d/%m/%Y")))
    s.commit(); return {"ok": True}

@app.get("/api/applications")
def my_apps(u: User = Depends(need("student")), s: Session = Depends(db)):
    out = []
    for a in s.query(Application).filter_by(student_id=u.id).order_by(Application.id.desc()):
        j = s.get(Job, a.job_id)
        if j: out.append({"id": a.id, "status": a.status, "created": a.created, "job": jdict(j)})
    return out

def own_job(jid, u, s):
    j = s.get(Job, jid)
    if not j or (u.role != "admin" and j.employer_id != u.id): raise HTTPException(403, "Không đủ quyền")
    return j

@app.get("/api/jobs/{jid}/applicants")
def applicants(jid: int, u: User = Depends(need("employer", "admin")), s: Session = Depends(db)):
    own_job(jid, u, s); out = []
    for a in s.query(Application).filter_by(job_id=jid):
        x = s.get(User, a.student_id)
        out.append({"id": a.id, "status": a.status, "created": a.created, "note": a.note, "name": x.name, "email": x.email, "skills": x.skills})
    return out

class St(BaseModel):
    status: str

@app.put("/api/applications/{aid}/status")
def set_status(aid: int, d: St, u: User = Depends(need("employer", "admin")), s: Session = Depends(db)):
    a = s.get(Application, aid)
    if not a: raise HTTPException(404, "Không tìm thấy")
    own_job(a.job_id, u, s)
    if d.status not in STATUS: raise HTTPException(400, "Trạng thái không hợp lệ")
    a.status = d.status; s.commit(); return {"ok": True}

@app.get("/api/applications/{aid}/cv")
def app_cv(aid: int, u: User = Depends(need("employer", "admin")), s: Session = Depends(db)):
    a = s.get(Application, aid)
    if not a: raise HTTPException(404, "Không tìm thấy")
    own_job(a.job_id, u, s); return cv_resp(s.get(User, a.student_id))

# ---------- Mật khẩu ----------
class Chg(BaseModel):
    old: str; new: str

@app.post("/api/change-password")
def chg(d: Chg, u: User = Depends(current), s: Session = Depends(db)):
    if not check_pw(d.old, u.pw): raise HTTPException(400, "Mật khẩu hiện tại không đúng")
    if len(d.new) < 6: raise HTTPException(400, "Mật khẩu mới tối thiểu 6 ký tự")
    u.pw = hash_pw(d.new); s.commit(); return {"ok": True}

def send_mail(to, subject, body):
    host = os.getenv("SMTP_HOST")
    if not host:
        print("[MAIL - chưa cấu hình SMTP]", to, body); return
    import smtplib
    from email.message import EmailMessage
    m = EmailMessage(); m["To"] = to; m["Subject"] = subject
    m["From"] = os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "")); m.set_content(body)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=15) as c:
        c.starttls(); c.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")); c.send_message(m)

class Em(BaseModel):
    email: EmailStr

@app.post("/api/forgot")
def forgot(d: Em, s: Session = Depends(db)):
    u = s.query(User).filter_by(email=d.email.lower()).first()
    if u:
        tok = jwt.encode({"sub": str(u.id), "pr": u.pw[-12:], "exp": dt.datetime.utcnow() + dt.timedelta(minutes=30)}, SECRET, "HS256")
        link = os.getenv("BASE_URL", "").rstrip("/") + "/#reset=" + tok
        try: send_mail(u.email, "Đặt lại mật khẩu CareerAI", f"Nhấn vào liên kết (hiệu lực 30 phút):\n{link}")
        except Exception as e: print("[MAIL ERROR]", e)
    return {"ok": True}

class Rs(BaseModel):
    token: str; password: str

@app.post("/api/reset")
def reset(d: Rs, s: Session = Depends(db)):
    try: p = jwt.decode(d.token, SECRET, ["HS256"])
    except JWTError: raise HTTPException(400, "Liên kết không hợp lệ hoặc đã hết hạn")
    u = s.get(User, int(p["sub"]))
    if not u or u.pw[-12:] != p.get("pr"): raise HTTPException(400, "Liên kết đã được sử dụng")
    if len(d.password) < 6: raise HTTPException(400, "Mật khẩu tối thiểu 6 ký tự")
    u.pw = hash_pw(d.password); s.commit(); return {"ok": True}

# ---------- Nhập dữ liệu việc làm (admin) ----------
@app.post("/api/admin/import-remote")
def imp_remote(u: User = Depends(need("admin")), s: Session = Depends(db)):
    n = import_remote(s)
    if n == 0: raise HTTPException(502, "Không nhập được (không có tin mới hoặc không kết nối được Remotive)")
    return {"added": n}

@app.post("/api/admin/import-csv")
async def imp_csv(file: UploadFile = File(...), u: User = Depends(need("admin")), s: Session = Depends(db)):
    n = 0
    for r in csv.DictReader(io.StringIO((await file.read()).decode("utf-8-sig"))):
        if r.get("title"):
            s.add(Job(title=r["title"], company=r.get("company", ""), kind=r.get("kind") or "Thực tập", skills=(r.get("skills") or "").lower(),
                      description=r.get("description", ""), url=r.get("url", ""), source="CSV", employer_id=u.id)); n += 1
    s.commit(); return {"added": n}

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
def index(): return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
