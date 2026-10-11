import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "real_estate_clean.csv"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "random_forest_model.joblib"
MODEL_METRICS_PATH = OUTPUT_DIR / "random_forest_metrics.joblib"
POWERBI_URL = os.getenv("POWERBI_EMBED_URL", "")
PREDICTION_FEATURES = ["area", "rooms", "district", "type"]
PROPERTY_TYPE_LABELS = {
    "Studio": "Căn hộ studio",
    "1-room apartment": "Căn hộ 1 phòng",
    "2-room apartment": "Căn hộ 2 phòng",
    "Two-room apartment": "Căn hộ 2 phòng",
    "3-room apartment": "Căn hộ 3 phòng",
    "4-room apartment": "Căn hộ 4 phòng",
    "5 or more room apartment": "Căn hộ từ 5 phòng",
}
DATA_VALUE_LABELS = {
    "Original condition": "Nguyên trạng",
    "Partial reconstruction": "Đã cải tạo một phần",
    "Complete reconstruction": "Đã cải tạo toàn bộ",
    "New building": "Công trình mới",
    "Under construction": "Đang xây dựng",
    "Development project": "Dự án đang phát triển",
    "Unknown": "Chưa rõ",
    "none": "Không có",
    "Brick": "Gạch",
    "Panel": "Bê tông tấm",
    "Other": "Khác",
    "Cube": "Khối hộp",
    "Mixed": "Vật liệu hỗn hợp",
    "Panel, Brick": "Bê tông tấm, gạch",
    "Skeletal": "Kết cấu khung",
    "Brick, Reinforced concrete": "Gạch, bê tông cốt thép",
    "Stone": "Đá",
    "Mounted": "Lắp ghép",
    "Wooden": "Gỗ",
    "Southwest": "Tây Nam",
    "South": "Nam",
    "West": "Tây",
    "East": "Đông",
    "Southeast": "Đông Nam",
    "Northeast": "Đông Bắc",
    "Northwest": "Tây Bắc",
    "North": "Bắc",
}
BOOLEAN_DATA_FIELDS = {"provision", "lift", "balkonies", "loggia", "cellar"}

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.template_filter("property_type_vi")
def property_type_vi(value):
    return PROPERTY_TYPE_LABELS.get(value, value)


@app.template_filter("data_value_vi")
def data_value_vi(value, field=None):
    if field in BOOLEAN_DATA_FIELDS and str(value) in {"0", "1"}:
        return "Có" if str(value) == "1" else "Không"
    return PROPERTY_TYPE_LABELS.get(value, DATA_VALUE_LABELS.get(value, value))


def load_dataset():
    return pd.read_csv(PROCESSED_PATH, encoding="utf-8")


def build_filter(df):
    query = request.args.get("q", "").strip()
    if query:
        searchable = df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).fillna(False)
        df = df[searchable.any(axis=1)]
    return df


def build_prediction_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", ["area", "rooms"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["district", "type"]),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_depth=16,
        min_samples_leaf=2,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_model_and_save(force_retrain=False):
    if MODEL_PATH.exists() and MODEL_METRICS_PATH.exists() and not force_retrain:
        return joblib.load(MODEL_PATH), joblib.load(MODEL_METRICS_PATH)

    df = load_dataset()
    model_df = df[["price", "area", "rooms", "district", "type"]].dropna().copy()
    if model_df.empty:
        raise ValueError("No usable records found for model training")

    X = model_df[PREDICTION_FEATURES]
    y = model_df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_prediction_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions, squared=False)),
        "R2": float(r2_score(y_test, predictions)),
    }

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(metrics, MODEL_METRICS_PATH)
    return pipeline, metrics


def load_prediction_model():
    if not MODEL_PATH.exists() or not MODEL_METRICS_PATH.exists():
        return train_model_and_save()[0]
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return train_model_and_save(force_retrain=True)[0]


def predict_price(payload):
    area = float(payload.get("area"))
    rooms = float(payload.get("rooms"))
    district = str(payload.get("district", "")).strip()
    property_type = str(payload.get("type", "")).strip()

    if not district or not property_type:
        raise ValueError("District and property type are required")
    if area <= 0 or rooms <= 0:
        raise ValueError("Area and rooms must be positive values")

    model = load_prediction_model()
    record = pd.DataFrame(
        [{"area": area, "rooms": rooms, "district": district, "type": property_type}],
        columns=PREDICTION_FEATURES,
    )
    prediction = float(model.predict(record)[0])
    return max(prediction, 0.0)


@lru_cache(maxsize=1)
def get_dataset_snapshot():
    df = load_dataset()
    avg_price = float(df["price"].mean())
    median_price = float(df["price"].median())
    total_listings = int(len(df))
    avg_area = float(df["area"].mean())
    return {
        "total_listings": total_listings,
        "avg_price": avg_price,
        "median_price": median_price,
        "avg_area": avg_area,
        "price_by_area": pd.read_csv(OUTPUT_DIR / "price_by_area.csv", encoding="utf-8").to_dict(orient="records"),
        "price_by_type": pd.read_csv(OUTPUT_DIR / "price_by_type.csv", encoding="utf-8").to_dict(orient="records"),
        "price_by_district": pd.read_csv(OUTPUT_DIR / "price_by_district.csv", encoding="utf-8").to_dict(orient="records"),
        "price_by_rooms": pd.read_csv(OUTPUT_DIR / "price_by_rooms.csv", encoding="utf-8").to_dict(orient="records"),
        "districts": sorted(df["district"].dropna().unique().tolist()),
        "property_types": sorted(df["type"].dropna().unique().tolist()),
    }


@app.route("/")
def index():
    summary = get_dataset_snapshot()
    return render_template(
        "overview.html",
        summary=summary,
        title="Tổng quan",
        currency_symbol="€",
    )


@app.route("/overview")
def overview():
    return index()


@app.route("/data-explorer")
def data_explorer():
    df = build_filter(load_dataset())
    sort_col = request.args.get("sort", "price")
    sort_dir = request.args.get("dir", "desc")
    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=sort_dir == "asc", na_position="last")
    page = max(1, int(request.args.get("page", 1)))
    per_page = 12
    total_rows = len(df)
    pages = max(1, (total_rows + per_page - 1) // per_page)
    page = min(page, pages)
    offset = (page - 1) * per_page
    page_df = df.iloc[offset:offset + per_page]
    return render_template(
        "data_explorer.html",
        page_df=page_df,
        sort_col=sort_col,
        sort_dir=sort_dir,
        page=page,
        pages=pages,
        total_rows=total_rows,
        query=request.args.get("q", ""),
        columns=[
            column
            for column in ["name_nsi", "price", "area", "rooms", "district", "type", "condition", "environment"]
            if column in df.columns
        ],
        column_labels={
            "name_nsi": "Tên địa điểm",
            "price": "Giá (€)",
            "index": "Chỉ số",
            "quality_of_living": "Chất lượng sống",
            "area": "Diện tích (m²)",
            "energy_costs": "Chi phí năng lượng",
            "provision": "Tiện nghi",
            "certificate": "Chứng nhận",
            "construction_type": "Loại công trình",
            "orientation": "Hướng",
            "year_built": "Năm xây dựng",
            "last_reconstruction": "Năm cải tạo gần nhất",
            "total_floors": "Tổng số tầng",
            "floor": "Tầng",
            "lift": "Thang máy",
            "balkonies": "Ban công",
            "loggia": "Lô gia",
            "cellar": "Tầng hầm",
            "district": "Khu vực",
            "type": "Loại bất động sản",
            "condition": "Tình trạng",
            "environment": "Môi trường",
            "safety": "An ninh",
            "transport": "Giao thông",
            "services": "Tiện ích",
            "relax": "Không gian thư giãn",
            "rooms": "Số phòng",
        },
        title="Khám phá dữ liệu",
        summary=get_dataset_snapshot(),
    )


@app.route("/market-analytics")
def market_analytics():
    summary = get_dataset_snapshot()
    return render_template("market_analytics.html", summary=summary, title="Phân tích thị trường")


@app.route("/price-prediction", methods=["GET", "POST"])
def price_prediction():
    summary = get_dataset_snapshot()
    district_choices = summary["districts"]
    type_choices = summary["property_types"]
    model_ready = MODEL_PATH.exists()
    model_metrics = {}
    if MODEL_METRICS_PATH.exists():
        model_metrics = joblib.load(MODEL_METRICS_PATH)
    return render_template(
        "price_prediction.html",
        summary=summary,
        district_choices=district_choices,
        type_choices=type_choices,
        model_metrics=model_metrics,
        model_ready=model_ready,
        powerbi_url=POWERBI_URL,
        title="Dự đoán giá",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()
    try:
        value = predict_price(payload)
        return jsonify({
            "status": "ok",
            "prediction": round(value, 2),
            "currency": "EUR",
            "formatted": f"€{value:,.0f}",
        })
    except Exception as exc:  # pragma: no cover - defensive validation
        message = str(exc)
        translations = {
            "District and property type are required": "Vui lòng chọn khu vực và loại bất động sản.",
            "Area and rooms must be positive values": "Diện tích và số phòng phải lớn hơn 0.",
        }
        return jsonify({"status": "error", "message": translations.get(message, "Thông tin nhập chưa hợp lệ.")}), 400


@app.route("/pipeline")
def pipeline():
    pipeline_steps = [
        {
            "title": "Tệp dữ liệu bất động sản gốc",
            "status": "Verified",
            "detail": "Tệp dữ liệu nguồn data/raw/Real Estate Dataset.csv có 14.969 bản ghi.",
        },
        {
            "title": "Làm sạch dữ liệu và chuẩn bị đặc trưng",
            "status": "Verified",
            "detail": "Dữ liệu đã xử lý loại bỏ giá trị không hợp lệ và chuẩn bị các trường đầu vào cho mô hình dự đoán.",
        },
        {
            "title": "Phân tích bằng PySpark",
            "status": "Verified",
            "detail": "Notebook khởi tạo SparkSession để tạo thống kê thị trường và kết quả so sánh mô hình.",
        },
        {
            "title": "Mô hình Random Forest đã huấn luyện",
            "status": "Active",
            "detail": "Mô hình scikit-learn được lưu tại data/output/random_forest_model.joblib và dùng để dự đoán trực tiếp.",
        },
        {
            "title": "Chỉ số đánh giá",
            "status": "Verified",
            "detail": "Các chỉ số MAE, RMSE và R² được lưu tại data/output/random_forest_metrics.joblib và hiển thị trên giao diện.",
        },
        {
            "title": "API và bảng điều khiển",
            "status": "Active",
            "detail": "Ứng dụng Flask đọc dữ liệu và kết quả phân tích trong dự án, đồng thời cung cấp dự đoán trực tiếp qua /api/predict.",
        },
    ]
    return render_template("pipeline.html", pipeline_steps=pipeline_steps, summary=get_dataset_snapshot(), title="Quy trình dữ liệu")


@app.route("/powerbi")
def powerbi():
    return render_template(
        "powerbi.html",
        summary=get_dataset_snapshot(),
        pbix_exists=(ROOT / "dashboard" / "Real_Estate_Dashboard.pbix").exists(),
        embed_url=POWERBI_URL,
        title="Power BI",
    )


@app.route("/about")
def about():
    summary = get_dataset_snapshot()
    return render_template("about.html", summary=summary, title="Giới thiệu")


@app.route("/api/export")
def export_csv():
    df = build_filter(load_dataset())
    sort_col = request.args.get("sort", "price")
    sort_dir = request.args.get("dir", "desc")
    if sort_col in df.columns:
        df = df.sort_values(by=sort_col, ascending=sort_dir == "asc", na_position="last")
    csv_bytes = df.to_csv(index=False, encoding="utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="real_estate_export.csv"'},
    )


@app.route("/api/overview")
def overview_api():
    summary = get_dataset_snapshot()
    return {
        "total_listings": summary["total_listings"],
        "avg_price": round(summary["avg_price"], 2),
        "median_price": round(summary["median_price"], 2),
        "avg_area": round(summary["avg_area"], 2),
        "districts": summary["price_by_district"],
        "types": summary["price_by_type"],
    }


@app.context_processor
def inject_common():
    return {"project_name": "Trí tuệ Bất động sản", "app_year": 2026}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
