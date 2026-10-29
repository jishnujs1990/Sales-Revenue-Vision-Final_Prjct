"""RevenueVision: beginner-friendly Streamlit sales and forecasting dashboard.
Run with: streamlit run streamlit_app.py
"""
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title="RevenueVision", page_icon="📈", layout="wide")
st.title("📈 RevenueVision | Sales Analytics & Forecasting")
st.caption("Upload your sales CSV to explore revenue, compare models, and forecast future months.")

@st.cache_data
def clean_data(csv_bytes):
    """Read and clean the same columns used in the simplified notebook."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    if not {"sales", "order_date"}.issubset(df.columns):
        raise ValueError("CSV must contain 'sales' and 'order_date' columns.")
    df = df.drop_duplicates().copy()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "profit" in df.columns:
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df = df.dropna(subset=["sales", "order_date"])
    df = df[df["sales"] >= 0].copy()
    if df.empty:
        raise ValueError("No valid rows remain after cleaning.")
    return df

@st.cache_data(show_spinner="Training the forecasting models...")
def analyze(csv_bytes, horizon):
    df = clean_data(csv_bytes)
    monthly = df.set_index("order_date")["sales"].resample("MS").sum()
    # Three previous months are used to predict each month's revenue.
    data = pd.DataFrame({"revenue": monthly})
    for lag in [1, 2, 3]:
        data[f"lag_{lag}"] = data["revenue"].shift(lag)
    data["month_number"] = data.index.month
    model_data = data.dropna()
    if len(model_data) < 12:
        raise ValueError("At least 15 consecutive calendar months are needed for model comparison.")
    features = ["lag_1", "lag_2", "lag_3", "month_number"]
    test_size = max(3, int(len(model_data) * 0.2))
    train, test = model_data.iloc[:-test_size], model_data.iloc[-test_size:]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                 objective="reg:squarederror", random_state=42),
    }
    results, predictions = [], {}
    for name, model in models.items():
        model.fit(train[features], train["revenue"])
        predicted = model.predict(test[features])
        predictions[name] = predicted
        results.append({"Model": name,
                        "MAE": mean_absolute_error(test["revenue"], predicted),
                        "RMSE": np.sqrt(mean_squared_error(test["revenue"], predicted))})
    comparison = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    best_name = comparison.iloc[0]["Model"]
    test_predictions = pd.DataFrame({"month": test.index,
                                     "actual": test["revenue"].to_numpy(),
                                     "predicted": predictions[best_name]})
    # Refit the chosen model on all known monthly values before future forecasting.
    best_model = models[best_name]
    best_model.fit(model_data[features], model_data["revenue"])
    history = list(monthly.to_numpy(dtype=float))
    future_months = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1),
                                  periods=horizon, freq="MS")
    future_values = []
    for month in future_months:
        row = pd.DataFrame([{"lag_1": history[-1], "lag_2": history[-2],
                             "lag_3": history[-3], "month_number": month.month}])
        value = max(0.0, float(best_model.predict(row[features])[0]))
        future_values.append(value)
        history.append(value)
    forecast = pd.DataFrame({"month": future_months, "forecast_revenue": future_values})
    return df, monthly, comparison, best_name, test_predictions, forecast

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Sales CSV file", type=["csv"])
    horizon = st.slider("Months to forecast", min_value=1, max_value=12, value=6)
    st.info("Uses existing order_date values. Dates are not generated or replaced.")

if uploaded is None:
    st.info("Upload your Global Superstore estimated-dates CSV using the sidebar to start.")
    st.stop()

try:
    df, monthly, comparison, best_name, test_predictions, forecast = analyze(uploaded.getvalue(), horizon)
except Exception as exc:
    st.error(f"Unable to analyze this file: {exc}")
    st.stop()

st.warning("If your order dates are estimated, monthly trends and forecasts are illustrative—not verified calendar-month predictions.")

# Dashboard tabs separate the main story into four clear parts.
overview, analysis, models_tab, future_tab = st.tabs([
    "📊 Sales Overview", "🧭 Sales Analysis", "🤖 Model Performance", "🔮 Revenue Forecast"
])

with overview:
    st.subheader("Business KPIs")
    total_sales = df["sales"].sum()
    cols = st.columns(4)
    cols[0].metric("Total Revenue", f"{total_sales:,.0f}")
    cols[1].metric("Average Sales / Row", f"{df['sales'].mean():,.0f}")
    cols[2].metric("Sales Rows", f"{len(df):,}")
    if "profit" in df.columns:
        cols[3].metric("Total Profit", f"{df['profit'].sum():,.0f}")
    else:
        cols[3].metric("Months in Dataset", len(monthly))
    st.plotly_chart(px.line(monthly.rename("Revenue").reset_index(), x="order_date", y="Revenue",
                            markers=True, title="Monthly Revenue Trend"), use_container_width=True)

with analysis:
    left, right = st.columns(2)
    with left:
        if "category" in df.columns:
            cat = df.groupby("category", as_index=False)["sales"].sum().sort_values("sales")
            st.plotly_chart(px.bar(cat, x="sales", y="category", orientation="h",
                                   title="Revenue by Category", labels={"sales": "Revenue"}),
                            use_container_width=True)
        else:
            st.info("Category chart unavailable: no category column.")
    with right:
        if "region" in df.columns:
            region = df.groupby("region", as_index=False)["sales"].sum()
            st.plotly_chart(px.pie(region, names="region", values="sales", hole=0.5,
                                   title="Revenue Share by Region"), use_container_width=True)
        else:
            st.info("Region chart unavailable: no region column.")
    st.subheader("Quick Business Insights")
    st.write(f"• Highest revenue month: **{monthly.idxmax():%B %Y}** ({monthly.max():,.0f}).")
    if "category" in df.columns:
        st.write(f"• Top category: **{df.groupby('category')['sales'].sum().idxmax()}**.")
    if "region" in df.columns:
        st.write(f"• Top region: **{df.groupby('region')['sales'].sum().idxmax()}**.")
    if "profit" in df.columns and total_sales:
        st.write(f"• Overall profit margin: **{df['profit'].sum() / total_sales * 100:.1f}%**.")

with models_tab:
    st.subheader("Compare 3 Forecasting Models")
    st.dataframe(comparison.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}"}),
                 use_container_width=True, hide_index=True)
    st.success(f"Lowest test MAE: {best_name}")
    st.caption("MAE and RMSE measure prediction error: lower is better. Model selection uses the test period, so this is an exploratory comparison, not an unbiased final evaluation.")
    st.plotly_chart(px.bar(comparison, x="Model", y="MAE", title="Model Error Comparison (MAE)"),
                    use_container_width=True)
    test_fig = go.Figure()
    test_fig.add_trace(go.Scatter(x=test_predictions["month"], y=test_predictions["actual"],
                                  name="Actual", mode="lines+markers"))
    test_fig.add_trace(go.Scatter(x=test_predictions["month"], y=test_predictions["predicted"],
                                  name="Predicted", mode="lines+markers"))
    test_fig.update_layout(title="Historical Test Months: Actual vs Predicted", yaxis_title="Revenue")
    st.plotly_chart(test_fig, use_container_width=True)

with future_tab:
    st.subheader(f"Final Visual: Historical Revenue + Next {horizon} Months")
    final_fig = go.Figure()
    final_fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, name="Historical revenue",
                                   mode="lines", line=dict(width=3)))
    # Connect the forecast to the last historical month for a continuous visual.
    final_fig.add_trace(go.Scatter(x=[monthly.index[-1], *forecast["month"].tolist()],
                                   y=[monthly.iloc[-1], *forecast["forecast_revenue"].tolist()],
                                   name="Future forecast", mode="lines+markers",
                                   line=dict(width=3, dash="dash")))
    final_fig.update_layout(title="RevenueVision | Revenue Forecast (Illustrative)",
                            xaxis_title="Month", yaxis_title="Revenue", hovermode="x unified")
    st.plotly_chart(final_fig, use_container_width=True)
    a, b = st.columns(2)
    a.metric("Forecast Total", f"{forecast['forecast_revenue'].sum():,.0f}")
    b.metric("Average Forecast / Month", f"{forecast['forecast_revenue'].mean():,.0f}")
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download Future Forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                       file_name="RevenueVision_Future_Forecast.csv", mime="text/csv")
    st.download_button("⬇ Download Model Comparison CSV", comparison.to_csv(index=False).encode("utf-8"),
                       file_name="RevenueVision_Model_Comparison.csv", mime="text/csv")
    st.download_button("⬇ Download Cleaned Sales CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="RevenueVision_Cleaned_Data.csv", mime="text/csv")

st.caption("Revenue = sum of sales values. Currency is not assumed; charts use the same units as your CSV.")
