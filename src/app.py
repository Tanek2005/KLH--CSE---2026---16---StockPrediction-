import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Core Scikit-Learn Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# SAFE DYNAMIC IMPORTS FOR HARDWARE/NATIVE DEPENDENCIES
# ---------------------------------------------------------
XGB_AVAILABLE = False
XGB_ERROR_MSG = ""
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception as e:
    XGB_ERROR_MSG = str(e)

CAT_AVAILABLE = False
CAT_ERROR_MSG = ""
try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except Exception as e:
    CAT_ERROR_MSG = str(e)

SHAP_AVAILABLE = False
SHAP_ERROR_MSG = ""
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    SHAP_ERROR_MSG = str(e)

LIME_AVAILABLE = False
LIME_ERROR_MSG = ""
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception as e:
    LIME_ERROR_MSG = str(e)

# Streamlit Page Config
st.set_page_config(
    page_title="Explainable ML Stock Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 1. HELPER FUNCTIONS: DATA FETCHING & FEATURE ENGINEERING
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker, start_date):
    """Fetch daily OHLCV historical data safely via yfinance."""
    try:
        df = yf.download(ticker, start=start_date, end=str(datetime.date.today()), progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Flatten MultiIndex columns if present in newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna(how='all')
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching ticker '{ticker}': {e}")
        return pd.DataFrame()

def engineer_features(df):
    """
    Construct financial indicators while handling edge cases safely:
    - Daily Returns & Lagged Variables
    - Trend: SMA (10, 50) & EMA (12, 26)
    - Momentum: RSI (14) & MACD
    - Volatility & Volume Indicators
    """
    data = df.copy()
    
    # Ensure numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 1. Returns & Lags
    data['Return_1d'] = data['Close'].pct_change()
    data['Lag_Close_1'] = data['Close'].shift(1)
    data['Lag_Return_1'] = data['Return_1d'].shift(1)
    
    # 2. Moving Averages
    data['SMA_10'] = data['Close'].rolling(window=10, min_periods=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50, min_periods=50).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # 3. MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # 4. Relative Strength Index (RSI - 14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
    rs = gain / (loss.replace(0, 1e-8))
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 5. Volatility & Volume
    data['Volatility_10'] = data['Return_1d'].rolling(window=10, min_periods=10).std()
    data['Volume_SMA_10'] = data['Volume'].rolling(window=10, min_periods=10).mean()
    
    # 6. Target Variable (Next Day Closing Price)
    data['Target_Next_Close'] = data['Close'].shift(-1)
    
    # Drop rows containing NaNs created by rolling windows/shifts
    data.dropna(inplace=True)
    return data

# ---------------------------------------------------------
# 2. SIDEBAR CONFIGURATION & STATUS CHECKS
# ---------------------------------------------------------
st.sidebar.title("Configuration & System Check")

# Module Diagnostics in Sidebar
with st.sidebar.expander("System Diagnostic Status", expanded=False):
    st.write(f"**XGBoost:** {'✅ Loaded' if XGB_AVAILABLE else '⚠️ Unavailable'}")
    if not XGB_AVAILABLE:
        st.caption(f"Reason: {XGB_ERROR_MSG}")
    st.write(f"**CatBoost:** {'✅ Loaded' if CAT_AVAILABLE else '⚠️ Unavailable'}")
    if not CAT_AVAILABLE:
        st.caption(f"Reason: {CAT_ERROR_MSG}")
    st.write(f"**SHAP:** {'✅ Loaded' if SHAP_AVAILABLE else '⚠️ Unavailable'}")
    st.write(f"**LIME:** {'✅ Loaded' if LIME_AVAILABLE else '⚠️ Unavailable'}")

stocks = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA"
}

option = st.sidebar.radio("Input Method", ["Top Stocks", "Enter Custom Ticker"])
if option == "Top Stocks":
    selected_stock = st.sidebar.selectbox("Choose a Stock", list(stocks.keys()))
    ticker = stocks[selected_stock]
else:
    ticker = st.sidebar.text_input(
        "Enter Ticker", 
        value="AAPL"
    ).upper().strip()

start_date = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
train_ratio = st.sidebar.slider("Chronological Train Ratio", 0.60, 0.90, 0.80, step=0.05)

st.sidebar.markdown("---")
st.sidebar.caption("💡 *Note for Indian Stocks:* Append `.NS` or `.BO` (e.g., `RELIANCE.NS`).")

# Main Header
st.title("Explainable AI Stock Prediction System")
st.caption(f"Analyzing Stock Ticker: **{ticker}** | Horizon: **{start_date}** to Today")

if not ticker:
    st.warning("Please provide a valid stock ticker.")
    st.stop()

# Load and Validate Data
with st.spinner(f"Loading historical market data for {ticker}..."):
    raw_data = load_stock_data(ticker, start_date)

if raw_data.empty or len(raw_data) < 60:
    st.error(
        f"Unable to fetch sufficient market data for ticker '{ticker}'. "
        "Ensure the ticker is valid (e.g., `AAPL`, or `COALINDIA.NS` for Indian markets) "
        "and the date range spans at least 60 trading days."
    )
    st.stop()

# Feature Engineering
df_features = engineer_features(raw_data)

if df_features.empty or len(df_features) < 20:
    st.error("Insufficient data left after constructing technical indicators. Select an earlier start date.")
    st.stop()

# Feature Definition
feature_cols = [
    'Close', 'Volume', 'Return_1d', 'Lag_Close_1', 'Lag_Return_1',
    'SMA_10', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal',
    'RSI_14', 'Volatility_10', 'Volume_SMA_10'
]

X = df_features[feature_cols]
y = df_features['Target_Next_Close']

# Chronological Train-Test Split (Temporal Leakage-aware)
split_idx = int(len(X) * train_ratio)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

if len(X_train) == 0 or len(X_test) == 0:
    st.error("Invalid train/test split. Adjust the chronological ratio slider.")
    st.stop()

# Initialize Available Models Gracefully
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

# Add Gradient Boosting baseline if XGBoost is missing
if XGB_AVAILABLE:
    models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
else:
    models["Gradient Boosting (Fallback)"] = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)

if CAT_AVAILABLE:
    models["CatBoost"] = CatBoostRegressor(iterations=100, learning_rate=0.05, verbose=0, random_seed=42)

# Train Models & Compute Metrics
results = []
trained_models = {}
predictions_dict = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        # Directional Accuracy
        actual_dir = np.sign(y_test.values - X_test['Close'].values)
        pred_dir = np.sign(preds - X_test['Close'].values)
        dir_acc = np.mean(actual_dir == pred_dir) * 100
        
        results.append({
            "Model": name,
            "MAE ($)": round(float(mae), 3),
            "RMSE ($)": round(float(rmse), 3),
            "R² Score": round(float(r2), 4),
            "Directional Accuracy (%)": round(float(dir_acc), 2)
        })
        
        trained_models[name] = model
        predictions_dict[name] = preds
    except Exception as e:
        st.warning(f"Skipping model '{name}' due to training exception: {e}")

df_results = pd.DataFrame(results)

# ---------------------------------------------------------
# 3. TABBED USER INTERFACE
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Data & EDA", 
    "📊 Comparative Benchmark", 
    "🔮 Next-Day Prediction", 
    "🔍 Explainable AI (SHAP & LIME)"
])

# ---------------------------------------------------------
# TAB 1: DATA & EDA
# ---------------------------------------------------------
with tab1:
    st.subheader("Data Quality & Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cleaned Records", len(df_features))
    col2.metric("Missing Values Cleaned", int(raw_data.isna().sum().sum()))
    col3.metric("Engineered Features Count", len(feature_cols))
    
    st.subheader("Historical Stock Price & Moving Averages")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_features.index, df_features['Close'], label='Closing Price', color='black', alpha=0.8)
    ax.plot(df_features.index, df_features['SMA_10'], label='SMA 10', linestyle='--', alpha=0.7)
    ax.plot(df_features.index, df_features['SMA_50'], label='SMA 50', linestyle='--', alpha=0.7)
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Feature Correlation Heatmap"):
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_features[feature_cols].corr(), annot=False, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

# ---------------------------------------------------------
# TAB 2: COMPARATIVE BENCHMARK
# ---------------------------------------------------------
with tab2:
    st.subheader("Comparative Model Performance Protocol")
    st.dataframe(df_results, use_container_width=True)
    
    st.subheader("Actual vs Predicted Prices (Holdout Test Set)")
    selected_model_tab2 = st.selectbox("Select Model to Visualize", list(trained_models.keys()))
    
    if selected_model_tab2 in predictions_dict:
        fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
        test_dates = df_features.index[split_idx:]
        ax_pred.plot(test_dates, y_test.values, label="Actual Next-Day Close", color="black", alpha=0.8)
        ax_pred.plot(test_dates, predictions_dict[selected_model_tab2], label=f"Predicted ({selected_model_tab2})", color="crimson", alpha=0.7)
        ax_pred.set_ylabel("Price")
        ax_pred.legend()
        st.pyplot(fig_pred)
        plt.close(fig_pred)

# ---------------------------------------------------------
# TAB 3: NEXT-DAY FORECAST
# ---------------------------------------------------------
with tab3:
    st.subheader("Next Trading Day Price Forecasting")
    
    best_model_name = st.selectbox("Select Prediction Model Engine", list(trained_models.keys()))
    selected_model = trained_models[best_model_name]
    
    # Extract latest feature row
    latest_features = X.iloc[-1:].copy()
    latest_close = float(df_features['Close'].iloc[-1])
    
    predicted_next_price = float(selected_model.predict(latest_features)[0])
    price_change = predicted_next_price - latest_close
    pct_change = (price_change / latest_close) * 100
    
    # Calculate Next Trading Day Date (Skipping Weekends)
    last_date = df_features.index[-1].date()
    next_day = last_date + datetime.timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += datetime.timedelta(days=1)
        
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close Price", f"${latest_close:.2f}")
    c2.metric("Predicted Price", f"${predicted_next_price:.2f}", f"${price_change:+.2f} ({pct_change:+.2f}%)")
    c3.metric("Directional Signal", "BULLISH 🚀" if predicted_next_price > latest_close else "BEARISH 📉")
    c4.metric("Next Trading Day", str(next_day))
    
    if predicted_next_price > latest_close:
        st.success(f"**Bullish Forecast**: Model predicts a price increase of **${price_change:.2f}** on {next_day}.")
    else:
        st.warning(f"**Bearish Forecast**: Model predicts a price decline of **${abs(price_change):.2f}** on {next_day}.")

# ---------------------------------------------------------
# TAB 4: EXPLAINABLE AI (SHAP & LIME)
# ---------------------------------------------------------
with tab4:
    st.subheader("Explainable Artificial Intelligence (XAI) Attribution")
    
    xai_model_choice = st.selectbox("Select Model for XAI Analysis", list(trained_models.keys()))
    target_model = trained_models[xai_model_choice]
    
    col_shap, col_lime = st.columns(2)
    
    # --- SHAP EXPLAINER ---
    with col_shap:
        st.markdown("### SHAP (Global Feature Attribution)")
        if not SHAP_AVAILABLE:
            st.warning("SHAP library is not installed or available.")
        else:
            try:
                # Use standard Explainer or TreeExplainer safely
                if hasattr(target_model, "get_booster") or hasattr(target_model, "estimators_"):
                    explainer = shap.TreeExplainer(target_model)
                else:
                    explainer = shap.Explainer(target_model, X_train)
                
                shap_values = explainer(X_test)
                
                fig_shap, ax_shap = plt.subplots(figsize=(6, 5))
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                st.pyplot(fig_shap)
                plt.close(fig_shap)
            except Exception as e:
                st.info(f"Note: Could not calculate SHAP plot for {xai_model_choice}. ({e})")

    # --- LIME EXPLAINER ---
    with col_lime:
        st.markdown("### LIME (Local Instance Explanation)")
        if not LIME_AVAILABLE:
            st.warning("LIME library is not installed or available.")
        else:
            try:
                lime_explainer = LimeTabularExplainer(
                    training_data=np.asarray(X_train, dtype=np.float64),
                    feature_names=list(feature_cols),
                    class_names=['Target_Next_Close'],
                    mode='regression'
                )
                
                latest_instance = X_test.iloc[-1].values.astype(np.float64)
                exp = lime_explainer.explain_instance(
                    data_row=latest_instance,
                    predict_fn=target_model.predict
                )
                
                fig_lime = exp.as_pyplot_figure()
                st.pyplot(fig_lime)
                plt.close(fig_lime)
            except Exception as e:
                st.info(f"Note: Could not calculate LIME explanation for {xai_model_choice}. ({e})")