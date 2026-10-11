FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=web_app/app.py \
    PORT=7860

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY web_app/app.py ./web_app/app.py
COPY web_app/templates ./web_app/templates
COPY web_app/static ./web_app/static
COPY data/processed/real_estate_clean.csv ./data/processed/real_estate_clean.csv
COPY data/output/price_by_area.csv ./data/output/price_by_area.csv
COPY data/output/price_by_district.csv ./data/output/price_by_district.csv
COPY data/output/price_by_rooms.csv ./data/output/price_by_rooms.csv
COPY data/output/price_by_type.csv ./data/output/price_by_type.csv
COPY data/output/random_forest_model.joblib ./data/output/random_forest_model.joblib
COPY data/output/random_forest_metrics.joblib ./data/output/random_forest_metrics.joblib

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-7860} web_app.app:app"]
