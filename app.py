import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Punjab Crop Price Forecast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #F8F9FA; }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #2E7D32;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    .metric-value { font-size: 28px; font-weight: 700; color: #1B5E20; }
    .metric-label { font-size: 13px; color: #666; margin-top: 4px; }
    
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #1B5E20;
        border-bottom: 2px solid #E8F5E9;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    .winner-badge {
        background: #E8F5E9;
        color: #2E7D32;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .sidebar .sidebar-content { background: #1B5E20; }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%);
    }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] .stSelectbox label { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CROPS = ['Carrot', 'Gram Black', 'Cauliflower', 'Banana(DOZENS)']
CROP_COLORS = {
    'Carrot': '#FF7043',
    'Gram Black': '#26A69A',
    'Cauliflower': '#EC407A',
    'Banana(DOZENS)': '#FFA726'
}
MAPE_RESULTS = {
    'Carrot':         {'SARIMA': 15.92, 'XGBoost': 14.66, 'LSTM': 10.83},
    'Gram Black':     {'SARIMA': 10.13, 'XGBoost': 27.90, 'LSTM': 6.88},
    'Cauliflower':    {'SARIMA': 20.88, 'XGBoost': 17.61, 'LSTM': 18.69},
    'Banana(DOZENS)': {'SARIMA': 13.90, 'XGBoost': 13.99, 'LSTM': 12.07},
}

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("forecast_dashboard.csv", parse_dates=["Date"])
    return df

@st.cache_resource
def load_models():
    scalers = joblib.load("models/scalers.pkl")
    models = {}
    name_map = {
        'Carrot': 'Carrot',
        'Gram Black': 'Gram_Black',
        'Cauliflower': 'Cauliflower',
        'Banana(DOZENS)': 'BananaDOZENS'
    }
    for crop, fname in name_map.items():
        models[crop] = load_model(f"models/{fname}.keras")
    return models, scalers

# ── Forecast function ─────────────────────────────────────────────────────────
def generate_forecast(crop, months, models, scalers, df):
    ts = df[df['Crop'] == crop][df['Type'] == 'Actual'].set_index('Date')['Price']
    ts = ts.resample('ME').mean().dropna()

    scaler = scalers[crop]
    model = models[crop]

    scaled = scaler.transform(ts.values.reshape(-1, 1))
    seq = scaled[-12:].copy()
    preds = []

    for _ in range(months):
        pred = model.predict(seq.reshape(1, 12, 1), verbose=0)
        preds.append(pred[0, 0])
        seq = np.append(seq[1:], pred, axis=0)

    prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    last_date = ts.index[-1]
    dates = pd.date_range(last_date, periods=months + 1, freq='ME')[1:]
    return dates, prices, ts

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Punjab Crop\nPrice Forecast")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Dashboard", "🔮 Forecast", "📋 Model Results", "ℹ️ About"])
    st.markdown("---")
    st.markdown("**Data:** 138 Punjab Markets")
    st.markdown("**Period:** 2007 – 2022")
    st.markdown("**Best Model:** LSTM")
    st.markdown("**Crops:** 4 commodity types")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = load_data()
    models, scalers = load_models()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Could not load data/models: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# Punjab Agricultural Price Dashboard")
    st.markdown("Daily commodity prices across 138 markets · 2007–2022")
    st.markdown("---")

    if data_loaded:
        actual = df[df['Type'] == 'Actual']

        # ── KPI cards ──────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("845K+", "Price Records"),
            ("138", "Punjab Markets"),
            ("15 Years", "Historical Data"),
            ("6.88%", "Best MAPE (LSTM)"),
        ]
        for col, (val, label) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Price trend ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Price Trends — All Crops</div>', unsafe_allow_html=True)

        fig = go.Figure()
        for crop in CROPS:
            crop_data = actual[actual['Crop'] == crop].groupby('Date')['Price'].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=crop_data['Date'], y=crop_data['Price'],
                name=crop, line=dict(color=CROP_COLORS[crop], width=1.5),
                mode='lines'
            ))
        fig.update_layout(
            height=350, paper_bgcolor='white', plot_bgcolor='#FAFAFA',
            legend=dict(orientation='h', y=-0.2),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='#F0F0F0', title='Avg Price (Rs.)'),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Volatility + Distribution ──────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Price Volatility</div>', unsafe_allow_html=True)
            vol = actual.groupby('Crop')['Price'].std().reset_index()
            vol.columns = ['Crop', 'Std Dev']
            vol['Color'] = vol['Crop'].map(CROP_COLORS)
            fig2 = px.bar(vol, x='Crop', y='Std Dev', color='Crop',
                         color_discrete_map=CROP_COLORS)
            fig2.update_layout(height=280, showlegend=False,
                              paper_bgcolor='white', plot_bgcolor='#FAFAFA',
                              margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(gridcolor='#F0F0F0'))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Data Distribution</div>', unsafe_allow_html=True)
            counts = actual.groupby('Crop')['Price'].count().reset_index()
            fig3 = px.pie(counts, values='Price', names='Crop',
                         color='Crop', color_discrete_map=CROP_COLORS,
                         hole=0.4)
            fig3.update_layout(height=280, paper_bgcolor='white',
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.markdown("# 6-Month Price Forecast")
    st.markdown("LSTM-powered predictions for Punjab commodity markets")
    st.markdown("---")

    col1, col2 = st.columns([1, 3])

    with col1:
        selected_crop = st.selectbox("Select Crop", CROPS)
        forecast_months = st.slider("Forecast Months", 1, 12, 6)
        show_history = st.slider("Historical Months to Show", 6, 36, 24)

    if data_loaded:
        dates, prices, ts = generate_forecast(selected_crop, forecast_months, models, scalers, df)

        with col2:
            fig = go.Figure()

            # Historical
            hist = ts[-show_history:]
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist.values,
                name='Actual', line=dict(color=CROP_COLORS[selected_crop], width=2),
                mode='lines'
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=dates, y=prices,
                name='Forecast', line=dict(color='#1B5E20', width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=6)
            ))

            # Confidence band
            upper = prices * 1.1
            lower = prices * 0.9
            fig.add_trace(go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill='toself', fillcolor='rgba(46,125,50,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±10% Band'
            ))

            fig.update_layout(
                height=400, paper_bgcolor='white', plot_bgcolor='#FAFAFA',
                legend=dict(orientation='h', y=-0.2),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='#F0F0F0', title='Price (Rs.)'),
                margin=dict(l=0, r=0, t=10, b=0),
                title=f"{selected_crop} — {forecast_months}-Month Forecast"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.markdown('<div class="section-header">Forecast Values</div>', unsafe_allow_html=True)
        forecast_df = pd.DataFrame({
            'Month': [d.strftime('%B %Y') for d in dates],
            'Forecasted Price (Rs.)': [f"Rs. {p:,.0f}" for p in prices],
            'Change vs Last Actual': [f"{((p - ts.iloc[-1]) / ts.iloc[-1] * 100):+.1f}%" for p in prices]
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # MAPE badge
        mape = MAPE_RESULTS[selected_crop]['LSTM']
        st.markdown(f'Model accuracy on this crop: <span class="winner-badge">LSTM MAPE: {mape}%</span>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Model Results":
    st.markdown("# Model Performance Comparison")
    st.markdown("SARIMA vs XGBoost vs LSTM — MAPE % across 4 crops")
    st.markdown("---")

    # Build results df
    rows = []
    for crop, results in MAPE_RESULTS.items():
        winner = min(results, key=results.get)
        rows.append({
            'Crop': crop,
            'SARIMA': results['SARIMA'],
            'XGBoost': results['XGBoost'],
            'LSTM': results['LSTM'],
            'Winner': winner
        })
    results_df = pd.DataFrame(rows)

    # Grouped bar chart
    fig = go.Figure()
    model_colors = {'SARIMA': '#78909C', 'XGBoost': '#FFA726', 'LSTM': '#2E7D32'}
    for model in ['SARIMA', 'XGBoost', 'LSTM']:
        fig.add_trace(go.Bar(
            name=model,
            x=results_df['Crop'],
            y=results_df[model],
            marker_color=model_colors[model],
            text=results_df[model].apply(lambda x: f"{x}%"),
            textposition='outside'
        ))

    fig.update_layout(
        barmode='group', height=400,
        paper_bgcolor='white', plot_bgcolor='#FAFAFA',
        yaxis=dict(gridcolor='#F0F0F0', title='MAPE %'),
        xaxis=dict(showgrid=False),
        legend=dict(orientation='h', y=-0.2),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.markdown('<div class="section-header">Full Results Table</div>', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Key insights
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**LSTM wins** on 3 out of 4 crops — best overall model")
    with col2:
        st.warning("**XGBoost** only wins on Cauliflower — high volatility crops")
    with col3:
        st.error("**SARIMA** weakest overall — linear assumptions fail on complex markets")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# About This Project")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📌 Project Summary
        This app presents an ML-based agricultural commodity price 
        forecasting system for Punjab, Pakistan. Three models — SARIMA, 
        XGBoost, and LSTM — are compared across four diverse crop types 
        using 15 years of daily market data from 138 Punjab mandis.
        
        ### 👨‍💻 Author
        **Dilawar Mahar**  
        BS Computer Science, Sukkur IBA University  
        Google Advanced Data Analytics Certificate
        
        ### 📊 Dashboard
        [View on Tableau Public](https://public.tableau.com/authoring/croppriceforecast/Dashboard2#1)
        """)

    with col2:
        st.markdown("""
        ### 📁 Dataset
        - **Source:** Punjab Agricultural Markets (Kaggle)
        - **Period:** May 2007 – December 2022
        - **Records:** 845,878 (after cleaning)
        - **Markets:** 138 cities across Punjab
        - **Crops:** Gram Black, Carrot, Cauliflower, Banana
        
        ### 🤖 Models
        | Model | Type | Best MAPE |
        |---|---|---|
        | SARIMA(1,1,1)(1,1,1,12) | Statistical | 10.13% |
        | XGBoost | ML | 14.66% |
        | LSTM (50 units) | Deep Learning | **6.88%** |
        
        ### 🛠️ Tech Stack
        Python · TensorFlow · Statsmodels · XGBoost  
        Streamlit · Plotly · Pandas · Tableau Public
        """)
