# CareerAI – Hệ thống AI kết nối sinh viên với thực tập & việc làm

**Công nghệ:** FastAPI · SQLAlchemy (SQLite/PostgreSQL) · JWT + bcrypt · scikit-learn (TF-IDF + cosine + skill overlap) · Claude API · SPA thuần HTML/JS.

**Phân quyền:** `student` (gợi ý việc, CV AI, kỹ năng, phỏng vấn, xu hướng) · `employer` (đăng tin, xem ứng viên phù hợp) · `admin` (quản lý user, khóa/xóa, thống kê).

## Chạy thử trên máy (chỉ để kiểm tra)
```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```
Tài khoản admin mặc định: `ADMIN_EMAIL` / `ADMIN_PASSWORD` (mặc định admin@example.com / Admin@123 – **phải đổi khi deploy**).

## Deploy online (Render – miễn phí, có URL công khai)
1. Đẩy thư mục này lên GitHub.
2. Render.com → New → **Blueprint** → chọn repo (đọc `render.yaml`).
3. Điền `ANTHROPIC_API_KEY`, `ADMIN_PASSWORD`. Để dữ liệu không mất khi restart, tạo Render PostgreSQL (free) và dán Internal URL vào `DATABASE_URL`.
4. Deploy xong sẽ có URL dạng `https://careerai.onrender.com`.
(Railway / Fly.io cũng chạy được bằng `Dockerfile`.)

## Kiến trúc matching
`điểm = 40% độ tương đồng TF-IDF (hồ sơ ↔ mô tả job) + 60% tỉ lệ kỹ năng trùng`; trả về kỹ năng đã có/còn thiếu cho từng job.

## Tính năng bổ sung (v2)
- **CV:** sinh viên tải PDF/DOCX/TXT (≤5MB, lưu trong DB), nội dung CV được dùng cho matching.
- **Ứng tuyển & theo dõi:** nút *Ứng tuyển*; sinh viên xem trạng thái (Đã nộp → Đang xem xét → Phỏng vấn → Trúng tuyển/Từ chối); doanh nghiệp xem đơn, tải CV, đổi trạng thái.
- **Mật khẩu:** đổi mật khẩu trong tab *Tài khoản*; *Quên mật khẩu* gửi link 30 phút, dùng một lần qua SMTP (`SMTP_*`, `BASE_URL`). Nếu chưa cấu hình SMTP, link chỉ in ra log máy chủ.
- **Dữ liệu việc làm thật:** lần chạy đầu tự nhập tin từ API công khai Remotive (việc làm IT remote quốc tế); admin bấm *Nhập tin từ Remotive* để làm mới, hoặc nhập CSV (mẫu: `data/jobs_template.csv`) cho tin trong nước từ doanh nghiệp đối tác.
- **Lưu ý nâng cấp:** thêm cột mới, nên xóa `app.db` cũ (hoặc dùng DB mới) trước khi chạy lại.

## AI miễn phí với Gemini
Đặt `GEMINI_API_KEY` (lấy tại aistudio.google.com/apikey). Nếu có key này, hệ thống dùng Gemini; muốn dùng Claude thì đặt `AI_PROVIDER=claude` và `ANTHROPIC_API_KEY`. Đổi model bằng `GEMINI_MODEL`.
