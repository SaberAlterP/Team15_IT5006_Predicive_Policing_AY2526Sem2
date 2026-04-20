"""
app.py — Arrest Prediction POC (Decision Tree)
===============================================
Streamlit web application that serves a trained Decision Tree binary classifier
to predict whether a reported Chicago crime incident is likely to result in arrest.

Deployment target : Streamlit Community Cloud
Model artifact    : decision_tree_bundle.pkl   (serialised sklearn model + feature schema)
Schema artifact   : dt_model_metadata.json     (categorical levels, numeric ranges,
                                                feature list)

Two prediction modes:
  - Tab 1 | Single prediction   : manual form input for one incident at a time.
  - Tab 2 | Batch CSV prediction : upload a CSV to score multiple incidents and
                                   download the results.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file for Streamlit Cloud compatibility
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / 'decision_tree_bundle.pkl'
META_PATH   = BASE_DIR / 'dt_model_metadata.json'

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title='Arrest Prediction — Decision Tree', page_icon='🌳', layout='wide')


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """Load the model bundle and metadata once, cached for the app lifetime.

    Returns
    -------
    model : sklearn DecisionTreeClassifier
        Trained model loaded from the pkl bundle.
    training_columns : list[str]
        One-hot-encoded column names seen during training; used to align
        the encoded inference DataFrame to the training schema.
    meta : dict
        Metadata including feature list, categorical levels, and numeric ranges.
    """
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    bundle = joblib.load(BUNDLE_PATH)
    model            = bundle['model']
    training_columns = bundle['training_columns']
    return model, training_columns, meta


# Load once at startup; subsequent reruns use the cache.
model, training_columns, meta = load_assets()

# ---------------------------------------------------------------------------
# Global constants derived from metadata
# ---------------------------------------------------------------------------
FEATURE_COLS       = meta['feature_cols']
CAT_FEATURES       = set(meta['categorical_features'])
CAT_LEVELS         = meta['categorical_levels']
NUMERIC_RANGES     = meta['numeric_ranges']


# ---------------------------------------------------------------------------
# Input validation & preprocessing
# ---------------------------------------------------------------------------
def coerce_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and preprocess a raw input DataFrame for the Decision Tree model.

    Validation rules
    ----------------
    1. All columns in FEATURE_COLS must be present.
    2. Extra columns are silently dropped.
    3. Categorical columns are checked against allowed values in CAT_LEVELS.
    4. Numeric columns are coerced to numeric; NaN raises a ValueError.

    After validation the function applies the same One-Hot encoding used at
    training time (pd.get_dummies, drop_first=False) and aligns the result
    to training_columns via reindex, filling unseen columns with 0.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input with one row per incident.

    Returns
    -------
    pd.DataFrame
        Encoded and aligned DataFrame ready for model.predict().

    Raises
    ------
    ValueError
        If required columns are missing, categorical values are out of range,
        or numeric columns contain non-numeric / null entries.
    """
    # --- 1. Check for missing required columns ---
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # --- 2. Drop extra columns ---
    clean = df[FEATURE_COLS].copy()

    # --- 3 & 4. Per-column validation ---
    for col in FEATURE_COLS:
        if col in CAT_FEATURES:
            clean[col] = clean[col].astype(str)
            allowed = set(CAT_LEVELS[col])
            bad = sorted(set(clean[col].dropna().unique()) - allowed)
            if bad:
                preview = ', '.join(map(str, bad[:5]))
                raise ValueError(
                    f"Column '{col}' contains unsupported values: {preview}. "
                    f"Allowed values: {', '.join(CAT_LEVELS[col])}."
                )
        else:
            clean[col] = pd.to_numeric(clean[col], errors='coerce')
            if clean[col].isna().any():
                raise ValueError(
                    f"Column '{col}' contains missing or non-numeric values."
                )

    # --- One-Hot encode categorical columns (same as training) ---
    X_encoded = pd.get_dummies(clean, columns=list(CAT_FEATURES), drop_first=False)

    # --- Align to training schema: add missing columns as 0, drop unseen ones ---
    X_encoded = X_encoded.reindex(columns=training_columns, fill_value=0)
    return X_encoded


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run the Decision Tree model on a validated input DataFrame.

    Appends three result columns to the original DataFrame:
      - arrest_probability : P(arrest=1) from predict_proba (4 d.p.)
      - predicted_arrest   : binary label (1 = likely arrest, 0 = unlikely)
      - model              : model name tag for transparency

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame (one row per incident).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with result columns appended.
    """
    X = coerce_input_frame(df)
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    out = df.copy()
    out['arrest_probability'] = probs.round(4)
    out['predicted_arrest']   = preds
    out['model']              = meta['model_name']
    return out


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title('🌳 Arrest Prediction — Decision Tree')
st.caption(
    'POC for predicting whether a Chicago crime incident is likely to result in arrest, '
    'using a trained Decision Tree classifier on Chicago Crime Data (2015–2025).'
)

# Collapsible model summary
with st.expander('Model summary', expanded=False):
    st.write({
        'Model'          : meta['model_name'],
        'Target'         : meta['target'],
        'Dataset'        : meta['dataset'],
        'Split strategy' : meta['split_strategy'],
        'Feature set'    : FEATURE_COLS,
    })

# ---------------------------------------------------------------------------
# UI — Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(['Single prediction', 'Batch CSV prediction'])

# ── Tab 1: Single prediction ────────────────────────────────────────────────
with tab1:
    st.subheader('Single incident prediction')

    # Three columns: crime features | location | temporal
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('**Crime**')
        crime_against = st.selectbox(
            'crime_against',
            CAT_LEVELS['crime_against'],
            index=CAT_LEVELS['crime_against'].index('Society')
        )
        offense_category_name = st.selectbox(
            'offense_category_name',
            CAT_LEVELS['offense_category_name'],
            index=CAT_LEVELS['offense_category_name'].index('Drug/Narcotic Offenses')
        )

    with c2:
        st.markdown('**Location**')
        st.caption('1=Residence  2=Street/Outdoor  3=Transportation Hub  4=Retail/Commercial  '
                   '5=Entertainment  6=Government/Public  7=Medical  8=Workplace  '
                   '9=Parking Lot  10=Unknown/Other')
        unified_location_code = st.number_input(
            'UNIFIED_LOCATION_CODE',
            min_value=NUMERIC_RANGES['UNIFIED_LOCATION_CODE'][0],
            max_value=NUMERIC_RANGES['UNIFIED_LOCATION_CODE'][1],
            value=2,  # Street/Outdoor
            step=1
        )

    with c3:
        st.markdown('**Time**')
        month      = st.number_input('month',    min_value=1,  max_value=12, value=6,  step=1)
        day        = st.number_input('day',      min_value=1,  max_value=31, value=15, step=1)
        hour       = st.number_input('hour',     min_value=0,  max_value=23, value=14, step=1)
        weekday    = st.number_input('weekday (0=Mon, 6=Sun)', min_value=0, max_value=6, value=2, step=1)
        is_weekend = st.selectbox('is_weekend', [0, 1])
        is_holiday = st.selectbox('is_holiday', [0, 1])

    if st.button('Run prediction', type='primary'):
        row = pd.DataFrame([{
            'UNIFIED_LOCATION_CODE': unified_location_code,
            'month'                : month,
            'day'                  : day,
            'hour'                 : hour,
            'weekday'              : weekday,
            'is_weekend'           : is_weekend,
            'is_holiday'           : is_holiday,
            'crime_against'        : crime_against,
            'offense_category_name': offense_category_name,
        }])
        try:
            result = predict_df(row)
            prob = float(result.loc[0, 'arrest_probability'])
            pred = int(result.loc[0, 'predicted_arrest'])
            if pred == 1:
                st.success(f'Prediction: Likely Arrest (probability = {prob:.2%})')
            else:
                st.info(f'Prediction: Unlikely Arrest (probability = {prob:.2%})')
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f'Prediction failed: {e}')

# ── Tab 2: Batch CSV prediction ─────────────────────────────────────────────
with tab2:
    st.subheader('Batch CSV prediction')
    st.write('Upload a CSV containing the following columns:')
    st.code(', '.join(FEATURE_COLS), language='text')

    # Generate a one-row sample template from metadata
    sample = pd.DataFrame([{
        col: (CAT_LEVELS[col][0] if col in CAT_FEATURES else NUMERIC_RANGES[col][0])
        for col in FEATURE_COLS
    }])
    st.download_button(
        'Download sample CSV template',
        data=sample.to_csv(index=False).encode('utf-8'),
        file_name='sample_dt_prediction_template.csv',
        mime='text/csv',
    )

    uploaded = st.file_uploader('Upload CSV file', type=['csv'])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write('Preview of uploaded file:')
            st.dataframe(batch_df.head(), use_container_width=True)

            # Validation and scoring via coerce_input_frame() internally
            result_df = predict_df(batch_df)
            st.success(f'Successfully scored {len(result_df)} rows.')
            st.dataframe(result_df.head(20), use_container_width=True)

            st.download_button(
                'Download prediction results',
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name='dt_prediction_results.csv',
                mime='text/csv',
            )
        except Exception as e:
            # Surfaces validation errors from coerce_input_frame() and
            # any unexpected parsing or runtime errors.
            st.error(f'Upload failed: {e}')

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('---')
st.write('Model bundle fields: `model`, `feature_cols`, `training_columns` · '
         'Schema controlled by `dt_model_metadata.json`')
