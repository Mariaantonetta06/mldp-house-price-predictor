import streamlit as st
import pandas as pd
import numpy as np
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
        margin-bottom: 6px;
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
    /* small spacing polish */
    .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏠 HOUSE PRICE PREDICTOR</h1>", unsafe_allow_html=True)

# ==================== LOAD MODEL WITH PIPELINE ====================
@st.cache_resource
def load_model_pipeline():
    """
    Loads the FULL sklearn Pipeline (preprocessor + model) saved as final_house_price_model.pkl.
    Uses joblib first; falls back to pickle if needed.
    """
    model_path = "final_house_price_model.pkl"
    if not os.path.exists(model_path):
        return None, "❌ Model file NOT found! (final_house_price_model.pkl)"

    # Try joblib first; if not installed / fails, use pickle
    try:
        import joblib
        model_obj = joblib.load(model_path)
        return model_obj, None
    except Exception:
        try:
            import pickle
            with open(model_path, "rb") as f:
                model_obj = pickle.load(f)
            return model_obj, None
        except Exception as e:
            return None, f"❌ Error loading model: {str(e)}"

model_pipeline, error = load_model_pipeline()

if error:
    st.error(error)
    st.stop()

# --- sanity check: ensure it's a PIPELINE (preprocess + model) ---
# If feature names look like cat__/num__, you loaded the post-encoded model, not the pipeline.
if hasattr(model_pipeline, "feature_names_in_"):
    fitted_cols = list(model_pipeline.feature_names_in_)
    looks_encoded = any(str(c).startswith(("cat__", "num__", "remainder__")) for c in fitted_cols)
else:
    fitted_cols = None
    looks_encoded = False

# If it’s not a sklearn Pipeline, or it looks like encoded columns, prediction will break / be wrong.
is_pipeline = hasattr(model_pipeline, "named_steps")

if (not is_pipeline) or looks_encoded:
    st.error(
        "❌ You loaded a model that expects ONE-HOT encoded feature columns (cat__/num__).\n\n"
        "Fix: re-save the FULL sklearn Pipeline (preprocessor + model) as final_house_price_model.pkl.\n"
        "In your notebook: joblib.dump(<your_pipeline>, 'final_house_price_model.pkl')"
    )
    st.stop()

st.success("✅ Model loaded successfully!")

# ==================== SIDEBAR ====================
with st.sidebar:
    # IMPORTANT: for regression, use R² wording (not “accuracy”).
    st.info(
        "**Model:** Random Forest\n\n"
        "**Test R²:** 0.88\n\n"
        "**Features (raw):** 79\n\n"
        "**Range shown:** ±15% (heuristic)"
    )

# ==================== INPUT ====================
st.markdown("## Enter Property Details")

# Better, more realistic defaults (first impression matters)
DEFAULT_QUALITY = 6
DEFAULT_GRLIVAREA = 1500
DEFAULT_GARAGECARS = 2
DEFAULT_BSMT = 800
DEFAULT_YEARBUILT = 2000
DEFAULT_LOTAREA = 10000

col1, col2 = st.columns(2)

with col1:
    # Use dataset-style naming for professionalism
    quality = st.slider("OverallQual (Overall Quality: 1–10)", 1, 10, DEFAULT_QUALITY)
    living_area = st.number_input("GrLivArea (Above Ground Living Area, sq ft)", 300, 6000, DEFAULT_GRLIVAREA, 50)
    garage = st.slider("GarageCars (Garage Capacity)", 0, 4, DEFAULT_GARAGECARS)

with col2:
    basement = st.number_input("TotalBsmtSF (Total Basement Area, sq ft)", 0, 5000, DEFAULT_BSMT, 50)
    year = st.slider("YearBuilt (Year Built)", 1900, 2025, DEFAULT_YEARBUILT)
    lot = st.number_input("LotArea (Lot Area, sq ft)", 1000, 200000, DEFAULT_LOTAREA, 500)

# Quick reset (nice UX, minimal change)
if st.button("Reset to defaults"):
    st.rerun()

# ==================== PREDICTION ====================
st.markdown("## Get Prediction")

if st.button("🔮 PREDICT PRICE", use_container_width=True):
    try:
        # Build a FULL raw-feature row the pipeline was trained on
        # (Fill everything else with NaN so your imputers handle it)
        raw_cols = list(model_pipeline.feature_names_in_)
        input_row = {c: np.nan for c in raw_cols}

        # Fill in user-selected fields (only if they exist in training cols)
        for k, v in {
            "OverallQual": quality,
            "GrLivArea": living_area,
            "GarageCars": garage,
            "TotalBsmtSF": basement,
            "YearBuilt": year,
            "LotArea": lot
        }.items():
            if k in input_row:
                input_row[k] = v

        input_df = pd.DataFrame([input_row], columns=raw_cols)

        with st.spinner("Predicting..."):
            # Your model predicts log1p(SalePrice)
            log_price = float(model_pipeline.predict(input_df)[0])
            price = float(np.expm1(log_price))

        st.success("✅ Prediction Complete!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Price (USD)", f"${price:,.0f}")
        c2.metric("Estimated Range (±15%)", f"${price*0.85:,.0f} - ${price*1.15:,.0f}")
        c3.metric("Test R²", "0.88")

        # Summary (keep your layout)
        st.dataframe(pd.DataFrame({
            "Feature": ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "LotArea"],
            "Value": [f"{quality}/10", f"{living_area:,}", f"{garage}", f"{basement:,}", f"{year}", f"{lot:,}"]
        }), use_container_width=True, hide_index=True)

        # Category (keep your logic)
        if price > 300000:
            st.success(f"🏆 Premium Property - ${price:,.0f}")
        elif price > 200000:
            st.info(f"⭐ Above Average - ${price:,.0f}")
        elif price > 150000:
            st.warning(f"✅ Average Price - ${price:,.0f}")
        else:
            st.info(f"💰 Budget Friendly - ${price:,.0f}")

        # Optional: show prediction details without cluttering main UI
        with st.expander("See prediction details"):
            st.write(f"Predicted log1p(SalePrice): {log_price:.4f}")
            st.write("Note: Price is converted back using expm1().")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.divider()
st.markdown("""
**MLDP Project** | Temasek Polytechnic | **Test R² ≈ 0.88**
""")
