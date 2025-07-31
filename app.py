# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Page title
st.set_page_config(page_title="IT Load Forecasting App", layout="wide")
st.title("🔌 IT Load Forecasting with ARIMAX")


# Step 1: Load dataset
@st.cache_data
def load_data():
    # Load time series data
    data = pd.read_csv("TimeSeries_TotalSolarGen_and_Load_IT_2016.csv", parse_dates=['utc_timestamp'], index_col='utc_timestamp')

    # Handle missing values
    data['IT_load_new'] = data['IT_load_new'].interpolate()
    data['IT_solar_generation'] = data['IT_solar_generation'].fillna(0)

    # ✅ Recreate time-based features from index
    data['month'] = data.index.month
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek

    return data


data = load_data()

# Step 2: Define target and exogenous variables
y = data['IT_load_new']
X = data[['IT_solar_generation', 'month', 'hour', 'dayofweek']]


# Step 3: Load the pre-trained ARIMAX model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model_fit = pickle.load(f)
    return model_fit


model_fit = load_model()

# Step 4: Select forecast horizon
st.sidebar.header("⚙️ Forecast Settings")
horizon = st.sidebar.slider("Select forecast horizon (in hours)", min_value=24, max_value=720, step=24, value=168)

# Step 5: Prepare test set for forecasting
X_test = X[-horizon:]
y_test = y[-horizon:]

# Step 6: Forecast
forecast = model_fit.forecast(steps=horizon, exog=X_test)

# Step 7: Plot forecast vs actual
st.subheader("📈 Forecast vs Actual")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_test.index, y_test, label="Actual")
ax.plot(y_test.index, forecast, label="Forecast", color="red")
ax.set_title("IT Load Forecast")
ax.set_xlabel("Time")
ax.set_ylabel("Load (MW)")
ax.legend()
st.pyplot(fig)

# Step 8: Show performance metrics
st.subheader("📊 Forecast Accuracy Metrics")
mae = mean_absolute_error(y_test, forecast)
rmse = mean_squared_error(y_test, forecast) ** 0.5
st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

# Step 9: Allow user to download the forecast
st.subheader("📥 Download Forecast")
forecast_df = pd.DataFrame({
    'timestamp': y_test.index,
    'actual_load': y_test.values,
    'forecast_load': forecast.values
})

csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecast as CSV", data=csv, file_name='forecast_output.csv', mime='text/csv')
