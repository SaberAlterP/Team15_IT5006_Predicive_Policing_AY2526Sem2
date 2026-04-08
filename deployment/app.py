"""
app.py — Arrest Prediction Proof-of-Concept (POC)
==================================================
Streamlit web application that serves a trained XGBoost binary classifier
to predict whether a reported crime incident is likely to result in an arrest.

Deployment target : Streamlit Community Cloud
Model artifact    : nibrs_xgb_model.json       (serialised XGBoost model)
Schema artifact   : nibrs_model_metadata.json  (feature list, allowed categorical
                    values, decision threshold, and training hyperparameters)

Two prediction modes are provided:
  - Tab 1 | Single prediction  : manual form input for one incident at a time.
  - Tab 2 | Batch CSV prediction: upload a CSV file to score multiple incidents
                                  and download the results.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve paths relative to this file so the app works both locally and on
# Streamlit Community Cloud regardless of the working directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'nibrs_xgb_model.json'
META_PATH  = BASE_DIR / 'nibrs_model_metadata.json'

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title='Arrest Prediction POC', page_icon='🚓', layout='wide')


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """Load the model and metadata once and cache them for the app lifetime.

    Returns
    -------
    model : xgb.XGBClassifier
        The trained XGBoost classifier loaded from the JSON artifact.
    meta : dict
        Metadata dictionary containing:
          - features            : ordered list of expected input column names
          - categorical_features: subset of features that are categorical
          - categorical_levels  : allowed values for each categorical feature
          - recommended_threshold: decision threshold tuned on the validation set
          - xgboost_params      : hyperparameters used during training
          - test_metrics_*      : evaluation metrics at various thresholds
    """
    # Load schema and training metadata
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # Re-instantiate the classifier with the same hyperparameters used at
    # training time so that load_model() restores weights correctly.
    model = xgb.XGBClassifier(
        n_estimators=meta['xgboost_params']['n_estimators'],
        max_depth=meta['xgboost_params']['max_depth'],
        learning_rate=meta['xgboost_params']['learning_rate'],
        subsample=meta['xgboost_params']['subsample'],
        colsample_bytree=meta['xgboost_params']['colsample_bytree'],
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        enable_categorical=True,  # required to handle pd.Categorical inputs
        tree_method='hist'
    )
    model.load_model(MODEL_PATH)
    return model, meta


# Load assets at startup; cached so subsequent reruns skip disk I/O.
model, meta = load_assets()

# ---------------------------------------------------------------------------
# Global constants derived from metadata
# ---------------------------------------------------------------------------
# Ordered list of feature columns the model expects.
FEATURES = meta['features']

# Set of feature names that must be treated as categorical (pd.Categorical).
CATEGORICAL_FEATURES = set(meta['categorical_features'])

# Mapping from categorical feature name → list of allowed string values.
# Used for both UI dropdowns and input validation.
CATEGORICAL_LEVELS = meta['categorical_levels']

# Classification threshold tuned on the validation set to balance
# precision and recall under class imbalance (arrest rate ≈ 19 %).
# Predictions with predicted_proba >= THRESHOLD are labelled as arrest=1.
THRESHOLD = meta['recommended_threshold']


# ---------------------------------------------------------------------------
# Input validation & preprocessing
# ---------------------------------------------------------------------------
def coerce_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a raw input DataFrame into the format expected by
    the XGBoost model.

    Validation rules
    ----------------
    1. All columns listed in FEATURES must be present; missing columns raise
       a ValueError naming the absent columns.
    2. Extra columns not in FEATURES are silently dropped.
    3. Categorical columns are cast to string, checked against the allowed
       values in CATEGORICAL_LEVELS, then encoded as pd.Categorical with the
       training-time category order preserved.
    4. Numeric columns are coerced with pd.to_numeric; any resulting NaN
       (from missing values or non-numeric strings) raises a ValueError.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input with one row per incident.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with correct dtypes and column ordering,
        ready to be passed to model.predict_proba().

    Raises
    ------
    ValueError
        If required columns are missing, categorical values are out of
        range, or numeric columns contain non-numeric / null entries.
    """
    # --- 1. Check for missing required columns ---
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # --- 2. Drop extra columns not used by the model ---
    extra = [c for c in df.columns if c not in FEATURES]
    if extra:
        df = df[[c for c in df.columns if c in FEATURES]]

    # Work on a copy to avoid mutating the caller's DataFrame.
    clean = df.copy()[FEATURES]

    # --- 3 & 4. Per-column type coercion and value validation ---
    for col in FEATURES:
        if col in CATEGORICAL_FEATURES:
            # Cast to string so mixed-type columns (e.g. int-encoded booleans)
            # are handled uniformly before the allowed-value check.
            clean[col] = clean[col].astype('string')

            allowed = set(CATEGORICAL_LEVELS[col])
            bad = sorted(set(clean[col].dropna().unique()) - allowed)
            if bad:
                # Show at most 5 bad values to keep the error message concise.
                preview = ', '.join(map(str, bad[:5]))
                raise ValueError(
                    f"Column '{col}' contains unsupported values: {preview}. "
                    f"Use one of: {', '.join(CATEGORICAL_LEVELS[col][:10])}..."
                )

            # Encode as pd.Categorical with the exact category order seen
            # during training; this is required by XGBoost's native categorical
            # support (enable_categorical=True).
            clean[col] = pd.Categorical(clean[col], categories=CATEGORICAL_LEVELS[col])

        else:
            # Coerce to numeric; non-parseable strings become NaN.
            clean[col] = pd.to_numeric(clean[col], errors='coerce')
            if clean[col].isna().any():
                raise ValueError(
                    f"Column '{col}' contains missing or non-numeric values."
                )

    return clean


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run the arrest-prediction model on a validated input DataFrame.

    Calls coerce_input_frame() for validation and preprocessing, then
    appends three result columns to the original DataFrame:
      - arrest_probability : model's predicted probability of arrest (float, 4 d.p.)
      - predicted_arrest   : binary label (1 = likely arrest, 0 = unlikely)
                             derived by applying THRESHOLD to the probability.
      - decision_threshold : the threshold value used, for transparency.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame (one row per incident).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with the three result columns appended.
    """
    X = coerce_input_frame(df)

    # predict_proba returns shape (n_samples, 2); column 1 is P(arrest=1).
    probs = model.predict_proba(X)[:, 1]

    # Apply threshold to produce binary labels.
    preds = (probs >= THRESHOLD).astype(int)

    # Append results to a copy of the original (uncoerced) input so the
    # output retains the user's original column values.
    out = df.copy()
    out['arrest_probability'] = probs.round(4)
    out['predicted_arrest']   = preds
    out['decision_threshold'] = THRESHOLD
    return out


# ---------------------------------------------------------------------------
# UI — Header
# ---------------------------------------------------------------------------
st.title('🚓 Arrest Prediction POC')
st.caption(
    'POC for predicting whether an incident is likely to result in arrest. '
    'This demo uses the prepared NIBRS feature schema and a deployment-ready '
    'XGBoost model artifact.'
)

# Collapsible model summary panel for transparency / reproducibility.
with st.expander('Model summary', expanded=False):
    st.write({
        'Target'                              : meta['target'],
        'Feature set'                         : meta['features'],
        'Recommended threshold'               : round(THRESHOLD, 4),
        'Split strategy'                      : meta['split_strategy'],
        'Test metrics @ recommended threshold': meta['test_metrics_recommended_threshold'],
    })
    st.warning(
        'Important: if your team must deploy the exact Chicago-trained model, '
        'replace the model file with the saved Chicago artifact. '
        'The uploaded notebook does not contain a serialized Chicago model, so '
        'this app currently uses the deployment-ready NIBRS model artifact.'
    )

# ---------------------------------------------------------------------------
# UI — Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(['Single prediction', 'Batch CSV prediction'])

# ── Tab 1: Single prediction ────────────────────────────────────────────────
with tab1:
    st.subheader('Single incident prediction')

    # Input widgets are arranged in three columns to match the three feature
    # groups: crime-related (col 1), location-related (col 2), temporal (col 3).
    c1, c2, c3 = st.columns(3)

    with c1:
        # Crime-related categorical features
        cargo_theft_flag      = st.selectbox('cargo_theft_flag',      CATEGORICAL_LEVELS['cargo_theft_flag'])
        incident_status       = st.selectbox('incident_status',       CATEGORICAL_LEVELS['incident_status'])
        attempt_complete_flag = st.selectbox('attempt_complete_flag', CATEGORICAL_LEVELS['attempt_complete_flag'])
        crime_against         = st.selectbox('crime_against',         CATEGORICAL_LEVELS['crime_against'])

    with c2:
        # Location-related categorical features
        location_name             = st.selectbox('location_name',             CATEGORICAL_LEVELS['location_name'],             index=23)  # Grocery/Supermarket
        unified_location_category = st.selectbox('unified_location_category', CATEGORICAL_LEVELS['unified_location_category'], index=5)   # Retail/Commercial
        new_offense_category_name = st.selectbox('New_offense_category_name', CATEGORICAL_LEVELS['New_offense_category_name'], index=5)   # Drug/Narcotic Offenses
        offense_name              = st.selectbox('offense_name',              CATEGORICAL_LEVELS['offense_name'],              index=13)  # Drug/Narcotic Violations

    with c3:
        # Temporal numeric features; min/max constraints provide lightweight
        # client-side validation before the row reaches coerce_input_frame().
        month      = st.number_input('month',      min_value=1,  max_value=12, value=1,  step=1)
        day        = st.number_input('day',        min_value=1,  max_value=31, value=1,  step=1)
        hour       = st.number_input('hour',       min_value=0,  max_value=23, value=12, step=1)
        weekday    = st.number_input('weekday',    min_value=0,  max_value=6,  value=0,  step=1)
        # Binary flags presented as dropdowns to prevent invalid free-text entry.
        is_weekend = st.selectbox('is_weekend', [0, 1])
        is_holiday = st.selectbox('is_holiday', [0, 1])

    if st.button('Run prediction', type='primary'):
        # Assemble the single-row DataFrame from widget values.
        row = pd.DataFrame([{
            'cargo_theft_flag'         : cargo_theft_flag,
            'incident_status'          : incident_status,
            'attempt_complete_flag'    : attempt_complete_flag,
            'crime_against'            : crime_against,
            'location_name'            : location_name,
            'unified_location_category': unified_location_category,
            'New_offense_category_name': new_offense_category_name,
            'offense_name'             : offense_name,
            'month'                    : month,
            'day'                      : day,
            'hour'                     : hour,
            'weekday'                  : weekday,
            'is_weekend'               : is_weekend,
            'is_holiday'               : is_holiday,
        }])

        try:
            result = predict_df(row)
            prob = float(result.loc[0, 'arrest_probability'])
            pred = int(result.loc[0, 'predicted_arrest'])

            # Display result with colour-coded feedback.
            if pred == 1:
                st.success(f'Prediction: Likely Arrest (probability = {prob:.2%})')
            else:
                st.info(f'Prediction: Unlikely Arrest (probability = {prob:.2%})')

            # Show the full result row for transparency.
            st.dataframe(result, use_container_width=True)

        except Exception as e:
            # Surface validation or model errors directly to the user.
            st.error(f'Prediction failed: {e}')

# ── Tab 2: Batch CSV prediction ─────────────────────────────────────────────
with tab2:
    st.subheader('Batch CSV prediction')
    st.write('Upload a CSV containing the following columns:')
    st.code(', '.join(FEATURES), language='text')

    # Generate a one-row sample CSV from the metadata so users can download a
    # correctly structured template without having to infer the schema manually.
    sample = pd.DataFrame([{
        k: (CATEGORICAL_LEVELS[k][0] if k in CATEGORICAL_FEATURES else 0)
        for k in FEATURES
    }])
    st.download_button(
        'Download sample CSV template',
        data=sample.to_csv(index=False).encode('utf-8'),
        file_name='sample_prediction_template.csv',
        mime='text/csv',
    )

    uploaded = st.file_uploader('Upload CSV file', type=['csv'])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)

            # Show a preview so users can confirm the file was parsed correctly
            # before the model runs.
            st.write('Preview of uploaded file:')
            st.dataframe(batch_df.head(), use_container_width=True)

            # Validation and scoring — coerce_input_frame() is called internally
            # and will raise a ValueError for any schema or value violations.
            result_df = predict_df(batch_df)

            st.success(f'Successfully scored {len(result_df)} rows.')

            # Show the first 20 scored rows as a quick sanity check.
            st.dataframe(result_df.head(20), use_container_width=True)

            # Allow the user to download the full scored dataset.
            st.download_button(
                'Download prediction results',
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name='prediction_results.csv',
                mime='text/csv',
            )

        except Exception as e:
            # Catches both schema validation errors from coerce_input_frame()
            # and unexpected parsing/runtime errors.
            st.error(f'Upload failed: {e}')

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('---')
st.write(
    'Suggested deployment target: **Streamlit Community Cloud**. '
    'It is the fastest way to meet the assignment requirements for a live URL, '
    'a basic interface, prediction output, and input validation.'
)
