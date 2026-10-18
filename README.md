---
title: Bất động sản thông minh
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Bất động sản thông minh

Bảng điều khiển phân tích dữ liệu bất động sản với chức năng tra cứu dữ liệu, phân tích thị trường và dự đoán giá bằng mô hình Random Forest đã huấn luyện.

## Chạy trên Hugging Face Spaces miễn phí

Space cần ở chế độ công khai để dùng phần cứng miễn phí. Khi công khai, các tệp được tải lên Space có thể được mọi người xem và tải xuống. Cấu hình Docker sử dụng cổng 7860 theo yêu cầu của Spaces.

Sau khi tạo Space mới với SDK **Docker**, đẩy các tệp cần thiết lên Space:

```powershell
git clone https://huggingface.co/spaces/TEN_TAI_KHOAN/TEN_SPACE
Copy-Item README.md,Dockerfile,requirements.txt TEN_SPACE\
Copy-Item web_app,data TEN_SPACE\ -Recurse
Set-Location TEN_SPACE
git add README.md Dockerfile requirements.txt web_app data
git commit -m "Deploy Vietnamese real estate dashboard"
git push
```

Thay `TEN_TAI_KHOAN` và `TEN_SPACE` bằng tên tài khoản và Space của bạn. Hugging Face sẽ tự build Docker image và cung cấp URL dạng `https://TEN_TAI_KHOAN-TEN_SPACE.hf.space`.

## Chạy sau khi clone từ GitHub

Cần cài Python 3.12. Mở PowerShell hoặc Terminal tại thư mục muốn lưu dự án, sau đó chạy:

```powershell
git clone https://github.com/khanhlinh107/real-estate-big-data.git
cd real-estate-big-data
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\web_app\app.py
```

Mở http://127.0.0.1:5000. Để lần sau khởi động nhanh hơn trên Windows, nhấp đúp `run_dashboard.bat`; tập lệnh sẽ tự tạo môi trường và cài thư viện nếu chưa có.

Trên macOS/Linux, dùng các lệnh sau thay cho bước tạo môi trường và cài thư viện:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python web_app/app.py
```

Sau đó mở http://127.0.0.1:5000.

## Chạy bằng Docker

```powershell
docker build -t real-estate-dashboard:local .
docker run --rm -p 5000:7860 real-estate-dashboard:local
```

Sau đó mở http://127.0.0.1:5000.

## Chức năng

- Tổng quan số liệu và biểu đồ thị trường.
- Tra cứu có gợi ý theo từ khóa một phần, hỗ trợ tiếng Việt không dấu; sắp xếp, phân trang và xuất dữ liệu CSV.
- So sánh hai khu vực dựa trên từng tin đăng trong bộ dữ liệu đã làm sạch.
- Phân tích giá theo khu vực, loại bất động sản, số phòng và diện tích.
- Dự đoán giá qua mô hình Random Forest và API `/api/predict`.

Mô hình dự đoán được lưu tại `data/output/random_forest_model.joblib`; dữ liệu đã xử lý được lưu tại `data/processed/real_estate_clean.csv`.

## Dữ liệu và quyền riêng tư

Repository công khai có chứa dữ liệu bất động sản, dữ liệu đã xử lý và mô hình dự đoán. Người clone dự án sẽ tải được các tệp đó cùng mã nguồn. Không đưa mật khẩu, access token, tệp `.env` hoặc dữ liệu cá nhân vào repository.

## Thành phần được tải lên Space

Chỉ tải lên các thành phần cần để chạy website: README, Dockerfile, requirements.txt, `web_app/` và `data/`. Không tải notebook, dữ liệu thô hoặc tệp môi trường cá nhân lên Space.
