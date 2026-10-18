import re
import unicodedata
from functools import lru_cache
from math import inf
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
PREDICTION_FEATURES = ["area", "rooms", "district", "type"]
SEARCHABLE_COLUMNS = [
    "name_nsi",
    "price",
    "area",
    "rooms",
    "district",
    "type",
    "condition",
    "environment",
]
NUMERIC_SEARCHABLE_COLUMNS = {"price", "area", "rooms"}
SEARCH_COLUMN_LABELS = {
    "name_nsi": "Tên địa điểm",
    "price": "Giá (€)",
    "area": "Diện tích (m²)",
    "rooms": "Số phòng",
    "district": "Khu vực",
    "type": "Loại bất động sản",
    "condition": "Tình trạng",
    "environment": "Môi trường",
}
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


def normalize_search_text(value):
    decomposed = unicodedata.normalize("NFKD", str(value).casefold().replace("đ", "d"))
    without_diacritics = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_diacritics.split())


def _search_display_value(value, field):
    if pd.isna(value):
        return ""
    return str(data_value_vi(value, field))


def _numeric_query_variants(query):
    normalized_query = normalize_search_text(query)
    variants = {normalized_query}

    if re.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+", normalized_query):
        variants.add(re.sub(r"\s", "", normalized_query))
    elif "," in normalized_query and "." in normalized_query:
        if normalized_query.rfind(",") > normalized_query.rfind("."):
            variants.add(
                re.sub(r"\.", "", normalized_query).replace(",", ".")
            )
        else:
            variants.add(normalized_query.replace(",", ""))
    elif "," in normalized_query:
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", normalized_query):
            variants.add(normalized_query.replace(",", ""))
        else:
            variants.add(normalized_query.replace(",", "."))
    elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", normalized_query):
        variants.add(normalized_query.replace(".", ""))

    return variants


def _matches_search_value(value, query, field):
    normalized_value = normalize_search_text(_search_display_value(value, field))
    normalized_query = normalize_search_text(query)
    if normalized_query in normalized_value:
        return True

    if field in NUMERIC_SEARCHABLE_COLUMNS and re.fullmatch(r"[+-]?[\d\s.,]+", query):
        return any(
            variant in normalized_value
            for variant in _numeric_query_variants(query)
        )
    return False


def build_filter(df, query="", field=None):
    normalized_query = normalize_search_text(query)
    if not normalized_query:
        return df

    selected_column = (
        field if field in SEARCHABLE_COLUMNS and field in df.columns else None
    )
    columns = [selected_column] if selected_column else [
        column for column in SEARCHABLE_COLUMNS if column in df.columns
    ]
    matching_rows = pd.Series(False, index=df.index)
    for column in columns:
        matching_rows |= df[column].map(
            lambda value: _matches_search_value(value, query, column)
        )
    df = df[matching_rows]
    return df


def _numeric_values(series, positive_only=False):
    values = pd.to_numeric(series, errors="coerce").replace([inf, -inf], float("nan")).dropna()
    if positive_only:
        values = values[values > 0]
    return values


def _comparison_metric(values_a, values_b, operation, unit):
    value_a = operation(values_a) if not values_a.empty else None
    value_b = operation(values_b) if not values_b.empty else None
    delta = value_b - value_a if value_a is not None and value_b is not None else None
    percentage = (
        delta / value_a * 100
        if delta is not None and value_a != 0
        else None
    )
    return {
        "a": float(value_a) if value_a is not None else None,
        "b": float(value_b) if value_b is not None else None,
        "delta": float(delta) if delta is not None else None,
        "percentage": float(percentage) if percentage is not None else None,
        "unit": unit,
    }


def build_area_comparison(df_a, df_b):
    prices_a = _numeric_values(df_a["price"], positive_only=True)
    prices_b = _numeric_values(df_b["price"], positive_only=True)
    areas_a = _numeric_values(df_a["area"], positive_only=True)
    areas_b = _numeric_values(df_b["area"], positive_only=True)

    paired_a = (
        df_a[["price", "area"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([inf, -inf], float("nan"))
        .dropna()
    )
    paired_b = (
        df_b[["price", "area"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([inf, -inf], float("nan"))
        .dropna()
    )
    valid_a = paired_a.loc[(paired_a["price"] > 0) & (paired_a["area"] > 0)]
    valid_b = paired_b.loc[(paired_b["price"] > 0) & (paired_b["area"] > 0)]
    ratios_a = valid_a["price"] / valid_a["area"]
    ratios_b = valid_b["price"] / valid_b["area"]

    metrics = {
        "listing_count": _comparison_metric(
            pd.Series([len(df_a)]), pd.Series([len(df_b)]), lambda values: values.iloc[0], "tin"
        ),
        "average_price": _comparison_metric(prices_a, prices_b, lambda values: values.mean(), "€"),
        "median_price": _comparison_metric(prices_a, prices_b, lambda values: values.median(), "€"),
        "average_area": _comparison_metric(areas_a, areas_b, lambda values: values.mean(), "m²"),
        "average_price_per_area": _comparison_metric(
            ratios_a, ratios_b, lambda values: values.mean(), "€/m²"
        ),
    }

    def category_counts(column):
        if column not in df_a or column not in df_b:
            return []
        counts_a = df_a[column].dropna().astype(str).value_counts()
        counts_b = df_b[column].dropna().astype(str).value_counts()
        categories = sorted(set(counts_a.index) | set(counts_b.index), key=normalize_search_text)
        return [
            {
                "label": category,
                "a": int(counts_a.get(category, 0)),
                "b": int(counts_b.get(category, 0)),
            }
            for category in categories
        ]

    type_counts = category_counts("type")
    room_counts = category_counts("rooms")
    distribution = {"labels": [], "a": [], "b": []}
    combined_prices = pd.concat([prices_a, prices_b], ignore_index=True)
    if not combined_prices.empty:
        minimum = float(combined_prices.min())
        maximum = float(combined_prices.max())
        if minimum == maximum:
            minimum = max(0.0, minimum - 0.5)
            maximum += 0.5
        bin_width = (maximum - minimum) / 10
        edges = [minimum + bin_width * index for index in range(11)]
        distribution["labels"] = [
            f"€{edges[index]:,.0f}–€{edges[index + 1]:,.0f}" for index in range(10)
        ]
        for prices, output in ((prices_a, "a"), (prices_b, "b")):
            buckets = [0] * 10
            for price in prices:
                bucket = min(int((float(price) - minimum) / bin_width), 9)
                buckets[bucket] += 1
            distribution[output] = buckets

    return {
        "metrics": metrics,
        "types": type_counts,
        "rooms": room_counts,
        "price_distribution": distribution,
        "has_price_distribution": not combined_prices.empty,
    }


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
    scatter_data = df[["area", "price"]].apply(pd.to_numeric, errors="coerce").dropna()
    scatter_data = scatter_data[(scatter_data["area"] > 0) & (scatter_data["price"] > 0)]
    if len(scatter_data) > 500:
        scatter_data = scatter_data.sample(n=500, random_state=42)
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
        "price_area_points": scatter_data.to_dict(orient="records"),
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
    query = request.args.get("q", "").strip()
    search_field = request.args.get("field", "").strip()
    df = build_filter(load_dataset(), query=query, field=search_field)
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
        query=query,
        search_field=search_field,
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


@app.route("/api/search")
def search_suggestions():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"suggestions": []})

    df = load_dataset()
    suggestions = []
    for column in SEARCHABLE_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column].dropna().map(
            lambda value: _search_display_value(value, column)
        ).value_counts()
        for value, count in values.items():
            if _matches_search_value(value, query, column):
                suggestions.append({
                    "value": value,
                    "field": column,
                    "field_label": SEARCH_COLUMN_LABELS[column],
                    "count": int(count),
                })

    suggestions.sort(
        key=lambda item: (
            SEARCHABLE_COLUMNS.index(item["field"]),
            normalize_search_text(item["value"]),
        )
    )
    return jsonify({"suggestions": suggestions[:10]})


@app.route("/area-comparison")
def area_comparison():
    districts = get_dataset_snapshot()["districts"]
    district_a = request.args.get("area_a", "").strip()
    district_b = request.args.get("area_b", "").strip()
    comparison = None
    error = None

    if district_a or district_b:
        if district_a not in districts or district_b not in districts:
            error = "Vui lòng chọn hai khu vực có trong dữ liệu."
        elif district_a == district_b:
            error = "Vui lòng chọn hai khu vực khác nhau để so sánh."
        else:
            dataset = load_dataset()
            comparison = build_area_comparison(
                dataset[dataset["district"] == district_a],
                dataset[dataset["district"] == district_b],
            )

    return render_template(
        "area_comparison.html",
        districts=districts,
        district_a=district_a,
        district_b=district_b,
        comparison=comparison,
        error=error,
        summary=get_dataset_snapshot(),
        title="So sánh thị trường bất động sản",
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


@app.route("/about")
def about():
    summary = get_dataset_snapshot()
    return render_template("about.html", summary=summary, title="Giới thiệu")


@app.route("/api/export")
def export_csv():
    df = build_filter(
        load_dataset(),
        query=request.args.get("q", "").strip(),
        field=request.args.get("field", "").strip(),
    )
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
    return {"project_name": "Bất động sản thông minh", "app_year": 2026}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
