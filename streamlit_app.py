import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ==================== CSS ====================
st.markdown("""
<style>
    h1 {
        color: white;
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 100%;
        border-radius: 10px;
        padding: 12px;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏠 HOUSE PRICE PREDICTOR</h1>", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model_pipeline():
    """Load the full sklearn Pipeline (preprocessor + model)"""
    model_path = "final_house_price_model.pkl"
    if not os.path.exists(model_path):
        return None, "❌ Model file NOT found!"
    
    try:
        model_obj = joblib.load(model_path)
        return model_obj, None
    except Exception as e:
        return None, f"❌ Error loading model: {str(e)}"

model_pipeline, error = load_model_pipeline()

if error:
    st.error(error)
    st.stop()

st.success("✅ Model loaded successfully!")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.info("""
    **Model:** Random Forest
    
    **Test R²:** 0.88
    
    **Features:** 79
    
    **Accuracy:** ±15%
    """)

# ==================== INPUT SECTION ====================
st.markdown("## Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    quality = st.slider("OverallQual (1-10)", 1, 10, 6, key="quality_slider")
    living_area = st.number_input("GrLivArea (sq ft)", 300, 6000, 1500, 50, key="grlivarea_input")
    garage = st.slider("GarageCars", 0, 4, 2, key="garage_slider")

with col2:
    basement = st.number_input("TotalBsmtSF (sq ft)", 0, 5000, 800, 50, key="basement_input")
    year = st.slider("YearBuilt", 1900, 2025, 2000, key="year_slider")
    lot = st.number_input("LotArea (sq ft)", 1000, 200000, 10000, 500, key="lot_input")

# Reset Button - Use session state
if st.button("🔄 Reset to Defaults", use_container_width=True):
    st.session_state.quality_slider = 6
    st.session_state.grlivarea_input = 1500
    st.session_state.garage_slider = 2
    st.session_state.basement_input = 800
    st.session_state.year_slider = 2000
    st.session_state.lot_input = 10000
    st.rerun()

# ==================== PREDICTION ====================
st.markdown("## Get Prediction")

if st.button("🔮 PREDICT PRICE", use_container_width=True):
    try:
        # Create DataFrame with ONLY the 6 raw features
        input_df = pd.DataFrame({
            'OverallQual': [quality],
            'GrLivArea': [living_area],
            'GarageCars': [garage],
            'TotalBsmtSF': [basement],
            'YearBuilt': [year],
            'LotArea': [lot]
        })
        
        # Predict using pipeline (handles preprocessing internally)
        with st.spinner("Making prediction..."):
            log_price = float(model_pipeline.predict(input_df)[0])
            price = float(np.expm1(log_price))
        
        st.success("✅ Prediction Complete!")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Price", f"${price:,.0f}")
        col2.metric("Range (±15%)", f"${price*0.85:,.0f} - ${price*1.15:,.0f}")
        col3.metric("Model R²", "0.88")
        
        # Summary table
        st.markdown("### 📋 Property Summary")
        summary_df = pd.DataFrame({
            'Feature': ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'LotArea'],
            'Your Input': [f'{quality}/10', f'{living_area:,}', f'{garage}', f'{basement:,}', f'{year}', f'{lot:,}']
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Price category
        st.markdown("### 🎯 Property Category")
        if price > 300000:
            st.success(f"🏆 Premium Property - ${price:,.0f}")
        elif price > 200000:
            st.info(f"⭐ Above Average - ${price:,.0f}")
        elif price > 150000:
            st.warning(f"✅ Average Price - ${price:,.0f}")
        else:
            st.info(f"💰 Budget Friendly - ${price:,.0f}")
            
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("Make sure all inputs are valid.")

st.divider()
st.markdown("""
**MLDP Project** | Temasek Polytechnic | Test R² = 0.88
""")