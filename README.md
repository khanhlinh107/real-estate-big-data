---
title: Trí tuệ Bất động sản
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Trí tuệ Bất động sản

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

## Chạy cục bộ

```powershell
.\.venv\Scripts\Activate.ps1
python .\web_app\app.py
```

Mở http://127.0.0.1:5000.

## Chạy bằng Docker

```powershell
docker build -t real-estate-dashboard:local .
docker run --rm -p 5000:7860 real-estate-dashboard:local
```

Sau đó mở http://127.0.0.1:5000.

## Chức năng

- Tổng quan số liệu và biểu đồ thị trường.
- Tra cứu, sắp xếp, phân trang và xuất dữ liệu CSV.
- Phân tích giá theo khu vực, loại bất động sản, số phòng và diện tích.
- Dự đoán giá qua mô hình Random Forest và API `/api/predict`.
- Trang Power BI hỗ trợ URL nhúng qua biến môi trường `POWERBI_EMBED_URL`.

Mô hình dự đoán được lưu tại `data/output/random_forest_model.joblib`; dữ liệu đã xử lý được lưu tại `data/processed/real_estate_clean.csv`.

## Thành phần được tải lên Space

Chỉ tải lên các thành phần cần để chạy website: README, Dockerfile, requirements.txt, `web_app/` và `data/`. Không tải notebook, Power BI PBIX, dữ liệu thô hoặc tệp môi trường cá nhân lên Space.
