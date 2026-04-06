import json
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'nibrs_xgb_model.json'
META_PATH = BASE_DIR / 'nibrs_model_metadata.json'

st.set_page_config(page_title='Arrest Prediction POC', page_icon='🚓', layout='wide')

@st.cache_resource
def load_assets():
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    model = xgb.XGBClassifier(
        n_estimators=meta['xgboost_params']['n_estimators'],
        max_depth=meta['xgboost_params']['max_depth'],
        learning_rate=meta['xgboost_params']['learning_rate'],
        subsample=meta['xgboost_params']['subsample'],
        colsample_bytree=meta['xgboost_params']['colsample_bytree'],
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        enable_categorical=True,
        tree_method='hist'
    )
    model.load_model(MODEL_PATH)
    return model, meta

model, meta = load_assets()
FEATURES = meta['features']
CATEGORICAL_FEATURES = set(meta['categorical_features'])
CATEGORICAL_LEVELS = meta['categorical_levels']
THRESHOLD = meta['recommended_threshold']


def coerce_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    extra = [c for c in df.columns if c not in FEATURES]
    if extra:
        df = df[[c for c in df.columns if c in FEATURES]]

    clean = df.copy()[FEATURES]

    for col in FEATURES:
        if col in CATEGORICAL_FEATURES:
            clean[col] = clean[col].astype('string')
            allowed = set(CATEGORICAL_LEVELS[col])
            bad = sorted(set(clean[col].dropna().unique()) - allowed)
            if bad:
                preview = ', '.join(map(str, bad[:5]))
                raise ValueError(
                    f"Column '{col}' contains unsupported values: {preview}. "
                    f"Use one of: {', '.join(CATEGORICAL_LEVELS[col][:10])}..."
                )
            clean[col] = pd.Categorical(clean[col], categories=CATEGORICAL_LEVELS[col])
        else:
            clean[col] = pd.to_numeric(clean[col], errors='coerce')
            if clean[col].isna().any():
                raise ValueError(f"Column '{col}' contains missing or non-numeric values.")

    return clean


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    X = coerce_input_frame(df)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    out = df.copy()
    out['arrest_probability'] = probs.round(4)
    out['predicted_arrest'] = preds
    out['decision_threshold'] = THRESHOLD
    return out


st.title('🚓 Arrest Prediction POC')
st.caption('POC for predicting whether an incident is likely to result in arrest. This demo uses the prepared NIBRS feature schema and a deployment-ready XGBoost model artifact.')

with st.expander('Model summary', expanded=False):
    st.write({
        'Target': meta['target'],
        'Feature set': meta['features'],
        'Recommended threshold': round(THRESHOLD, 4),
        'Split strategy': meta['split_strategy'],
        'Test metrics @ recommended threshold': meta['test_metrics_recommended_threshold']
    })
    st.warning(
        'Important: if your team must deploy the exact Chicago-trained model, replace the model file with the saved Chicago artifact. '
        'The uploaded notebook does not contain a serialized Chicago model, so this app currently uses the deployment-ready NIBRS model artifact.'
    )

tab1, tab2 = st.tabs(['Single prediction', 'Batch CSV prediction'])

with tab1:
    st.subheader('Single incident prediction')
    c1, c2, c3 = st.columns(3)
    with c1:
        cargo_theft_flag = st.selectbox('cargo_theft_flag', CATEGORICAL_LEVELS['cargo_theft_flag'])
        incident_status = st.selectbox('incident_status', CATEGORICAL_LEVELS['incident_status'])
        attempt_complete_flag = st.selectbox('attempt_complete_flag', CATEGORICAL_LEVELS['attempt_complete_flag'])
        crime_against = st.selectbox('crime_against', CATEGORICAL_LEVELS['crime_against'])
    with c2:
        location_name = st.selectbox('location_name', CATEGORICAL_LEVELS['location_name'])
        unified_location_category = st.selectbox('unified_location_category', CATEGORICAL_LEVELS['unified_location_category'])
        new_offense_category_name = st.selectbox('New_offense_category_name', CATEGORICAL_LEVELS['New_offense_category_name'])
        offense_name = st.selectbox('offense_name', CATEGORICAL_LEVELS['offense_name'])
    with c3:
        month = st.number_input('month', min_value=1, max_value=12, value=1, step=1)
        day = st.number_input('day', min_value=1, max_value=31, value=1, step=1)
        hour = st.number_input('hour', min_value=0, max_value=23, value=12, step=1)
        weekday = st.number_input('weekday', min_value=0, max_value=6, value=0, step=1)
        is_weekend = st.selectbox('is_weekend', [0, 1])
        is_holiday = st.selectbox('is_holiday', [0, 1])

    if st.button('Run prediction', type='primary'):
        row = pd.DataFrame([{
            'cargo_theft_flag': cargo_theft_flag,
            'incident_status': incident_status,
            'attempt_complete_flag': attempt_complete_flag,
            'crime_against': crime_against,
            'location_name': location_name,
            'unified_location_category': unified_location_category,
            'New_offense_category_name': new_offense_category_name,
            'offense_name': offense_name,
            'month': month,
            'day': day,
            'hour': hour,
            'weekday': weekday,
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
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

with tab2:
    st.subheader('Batch CSV prediction')
    st.write('Upload a CSV containing the following columns:')
    st.code(', '.join(FEATURES), language='text')

    sample = pd.DataFrame([{k: (CATEGORICAL_LEVELS[k][0] if k in CATEGORICAL_FEATURES else 0) for k in FEATURES}])
    st.download_button(
        'Download sample CSV template',
        data=sample.to_csv(index=False).encode('utf-8'),
        file_name='sample_prediction_template.csv',
        mime='text/csv'
    )

    uploaded = st.file_uploader('Upload CSV file', type=['csv'])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.write('Preview of uploaded file:')
            st.dataframe(batch_df.head(), use_container_width=True)
            result_df = predict_df(batch_df)
            st.success(f'Successfully scored {len(result_df)} rows.')
            st.dataframe(result_df.head(20), use_container_width=True)
            st.download_button(
                'Download prediction results',
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name='prediction_results.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f'Upload failed: {e}')

st.markdown('---')
st.write('Suggested deployment target: **Streamlit Community Cloud**. It is the fastest way to meet the assignment requirements for a live URL, a basic interface, prediction output, and input validation.')
