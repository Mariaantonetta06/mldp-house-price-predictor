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

def get_required_raw_columns(pipeline_obj):
    """
    Returns the list of RAW (pre-encoded) feature columns expected by the pipeline.
    Tries feature_names_in_ first; otherwise extracts from ColumnTransformer transformers_.
    """
    # Best case: sklearn recorded the raw input columns during fit
    if hasattr(pipeline_obj, "feature_names_in_"):
        cols = list(pipeline_obj.feature_names_in_)
        # If they look like already-one-hot-encoded names, it's the wrong artifact
        looks_encoded = any(str(c).startswith(("cat__", "num__", "remainder__")) for c in cols)
        if looks_encoded:
            raise ValueError(
                "You loaded a model that expects ONE-HOT encoded columns (cat__/num__). "
                "Re-save the FULL pipeline (preprocessor + model) as final_house_price_model.pkl."
            )
        return cols

    # Otherwise, extract raw column selectors from the ColumnTransformer inside the pipeline
    if not hasattr(pipeline_obj, "named_steps"):
        raise ValueError("Loaded object is not a sklearn Pipeline. Please save the FULL pipeline.")

    preprocessor = None
    for _, step_obj in pipeline_obj.named_steps.items():
        if "ColumnTransformer" in type(step_obj).__name__:
            preprocessor = step_obj
            break

    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        raise ValueError(
            "Cannot detect required raw columns. Ensure your saved file is the FULL pipeline "
            "(ColumnTransformer preprocessor + model)."
        )

    required = []
    for _, _, cols in preprocessor.transformers_:
        if cols is None or cols == "drop":
            continue
        if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
            required.extend(list(cols))

    required = sorted(set(required))
    if not required:
        raise ValueError("Detected 0 raw columns from preprocessor. Re-check how the pipeline was built/saved.")
    return required

model_pipeline, error = load_model_pipeline()
if error:
    st.error(error)
    st.stop()

# Sanity check: must be Pipeline
if not hasattr(model_pipeline, "named_steps"):
    st.error(
        "❌ Loaded model is not a sklearn Pipeline.\n\n"
        "Fix: save the FULL pipeline (preprocessor + model) into final_house_price_model.pkl."
    )
    st.stop()

# Detect required raw columns (prevents missing-columns errors)
try:
    REQUIRED_COLS = get_required_raw_columns(model_pipeline)
except Exception as e:
    st.error(f"❌ {str(e)}")
    st.stop()

st.success("✅ Model loaded successfully!")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.info(
        "**Model:** Random Forest\n\n"
        "**Test R²:** 0.88\n\n"
        f"**Features (raw):** {len(REQUIRED_COLS)}\n\n"
        "**Range shown:** ±15% (heuristic)"
    )

# ==================== DEFAULTS + RESET (FIXED) ====================
DEFAULTS = {
    "quality": 6,
    "living_area": 1500,
    "garage": 2,
    "basement": 800,
    "year": 2000,
    "lot": 10000,
}

def reset_defaults():
    """
    Proper reset: sets widget keys back to defaults.
    This works because widgets use these keys.
    """
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# Initialize defaults once
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== INPUT ====================
st.markdown("## Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    quality = st.slider(
        "OverallQual (Overall Quality: 1–10)",
        1, 10,
        key="quality"
    )
    living_area = st.number_input(
        "GrLivArea (Above Ground Living Area, sq ft)",
        300, 6000,
        step=50,
        key="living_area"
    )
    garage = st.slider(
        "GarageCars (Garage Capacity)",
        0, 4,
        key="garage"
    )

with col2:
    basement = st.number_input(
        "TotalBsmtSF (Total Basement Area, sq ft)",
        0, 5000,
        step=50,
        key="basement"
    )
    year = st.slider(
        "YearBuilt (Year Built)",
        1900, 2025,
        key="year"
    )
    lot = st.number_input(
        "LotArea (Lot Area, sq ft)",
        1000, 200000,
        step=500,
        key="lot"
    )

# ✅ Reset button that actually works
if st.button("Reset to defaults"):
    reset_defaults()
    st.rerun()

# ==================== PREDICTION ====================
st.markdown("## Get Prediction")

if st.button("🔮 PREDICT PRICE", use_container_width=True):
    try:
        # Build a FULL raw-feature row the pipeline expects (prevents missing-columns errors)
        input_row = {c: np.nan for c in REQUIRED_COLS}

        # Fill only what your UI collects
        user_values = {
            "OverallQual": st.session_state["quality"],
            "GrLivArea": st.session_state["living_area"],
            "GarageCars": st.session_state["garage"],
            "TotalBsmtSF": st.session_state["basement"],
            "YearBuilt": st.session_state["year"],
            "LotArea": st.session_state["lot"],
        }

        for k, v in user_values.items():
            if k in input_row:
                input_row[k] = v

        input_df = pd.DataFrame([input_row], columns=REQUIRED_COLS)

        with st.spinner("Predicting..."):
            # Model predicts log1p(SalePrice)
            log_price = float(model_pipeline.predict(input_df)[0])
            price = float(np.expm1(log_price))

        st.success("✅ Prediction Complete!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Price (USD)", f"${price:,.0f}")
        c2.metric("Estimated Range (±15%)", f"${price*0.85:,.0f} - ${price*1.15:,.0f}")
        c3.metric("Test R²", "0.88")

        st.dataframe(pd.DataFrame({
            "Feature": ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "LotArea"],
            "Value": [
                f'{st.session_state["quality"]}/10',
                f'{st.session_state["living_area"]:,}',
                f'{st.session_state["garage"]}',
                f'{st.session_state["basement"]:,}',
                f'{st.session_state["year"]}',
                f'{st.session_state["lot"]:,}'
            ]
        }), use_container_width=True, hide_index=True)

        if price > 300000:
            st.success(f"🏆 Premium Property - ${price:,.0f}")
        elif price > 200000:
            st.info(f"⭐ Above Average - ${price:,.0f}")
        elif price > 150000:
            st.warning(f"✅ Average Price - ${price:,.0f}")
        else:
            st.info(f"💰 Budget Friendly - ${price:,.0f}")

        with st.expander("See prediction details"):
            st.write(f"Predicted log1p(SalePrice): {log_price:.4f}")
            st.write("Note: Price is converted back using expm1().")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

st.divider()
st.markdown("""
**MLDP Project** | Temasek Polytechnic | **Test R² ≈ 0.88**
""")
