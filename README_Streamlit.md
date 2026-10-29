# RevenueVision Streamlit Dashboard

A simple interactive dashboard based on `RevenueVision_Simplified_Project.ipynb`.

## Run on your computer

1. Put `streamlit_app.py` and `requirements_streamlit.txt` in the same folder.
2. Open a terminal in that folder.
3. Install packages:

   ```bash
   pip install -r requirements_streamlit.txt
   ```

4. Start the dashboard:

   ```bash
   streamlit run streamlit_app.py
   ```

5. Upload your Global Superstore CSV in the sidebar. It must contain `sales` and `order_date` columns; `category`, `region`, and `profit` are optional.

## What the dashboard shows

- Sales KPIs and monthly trend
- Revenue by category and region (if present)
- Three models: Linear Regression, Random Forest, XGBoost
- MAE/RMSE comparison and historical test predictions
- Recursive 1–12 month forecast with a final historical-versus-forecast chart
- CSV downloads

## Important limitations

- The dashboard retrains the same simple models when a CSV is uploaded. It does not execute the notebook or require a saved `.joblib` file.
- The dataset's dates may be estimated. Forecasts are illustrative.
- The model with the lowest test MAE is selected for convenience, so its reported test error is not an unbiased estimate of model-selection performance.
- Each model predicts historical test months using actual prior-month sales, while future forecasting recursively uses its own predictions.
- Empty calendar months are treated as zero revenue. Check that your source contains continuous monthly coverage.
- No SQL or FastAPI is needed.
