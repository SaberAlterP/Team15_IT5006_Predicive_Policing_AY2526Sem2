import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title='Arrest Prediction - Decision Tree', layout='wide')

BUNDLE_PATH = 'decision_tree_bundle.pkl'

@st.cache_resource

def load_bundle(path: str):
    return joblib.load(path)


def preprocess_input(df: pd.DataFrame, feature_cols, training_columns):
    missing_required = [c for c in feature_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f'Missing required columns: {missing_required}')

    X = df[feature_cols].copy()

    if 'crime_against' in X.columns:
        X['crime_against'] = X['crime_against'].fillna('Unknown')
    if 'offense_category_name' in X.columns:
        X['offense_category_name'] = X['offense_category_name'].fillna('Unknown')

    X_encoded = pd.get_dummies(
        X,
        columns=['crime_against', 'offense_category_name'],
        drop_first=False
    )

    X_encoded = X_encoded.reindex(columns=training_columns, fill_value=0)
    return X_encoded


def evaluate_predictions(y_true, y_pred, y_prob):
    results = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        results['ROC_AUC'] = roc_auc_score(y_true, y_prob)
    return pd.DataFrame({'Metric': list(results.keys()), 'Score': list(results.values())})


st.title('Chicago Arrest Prediction - Decision Tree')
st.caption('Upload a CSV file with the same raw feature columns used in training.')

try:
    bundle = load_bundle(BUNDLE_PATH)
except FileNotFoundError:
    st.error(
        'decision_tree_bundle.pkl was not found. Please export the model bundle from the notebook and place it in the same folder as app.py.'
    )
    st.stop()

model = bundle['model']
feature_cols = bundle['feature_cols']
training_columns = bundle['training_columns']
model_name = bundle.get('model_name', 'Decision Tree')

with st.expander('Required input columns', expanded=True):
    st.write(feature_cols)
    st.write('Optional label column for evaluation: arrest or Arrest')

uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader('Preview of uploaded data')
    st.dataframe(raw_df.head(), use_container_width=True)

    try:
        X_input = preprocess_input(raw_df, feature_cols, training_columns)
        pred = model.predict(X_input)
        prob = model.predict_proba(X_input)[:, 1] if hasattr(model, 'predict_proba') else None

        result_df = raw_df.copy()
        result_df['prediction'] = pred
        if prob is not None:
            result_df['probability_arrest_1'] = prob

        st.subheader(f'Predictions from {model_name}')
        st.dataframe(result_df.head(20), use_container_width=True)

        label_col = None
        if 'arrest' in raw_df.columns:
            label_col = 'arrest'
        elif 'Arrest' in raw_df.columns:
            label_col = 'Arrest'

        if label_col is not None:
            y_true = raw_df[label_col].astype(int)
            eval_df = evaluate_predictions(y_true, pred, prob)
            st.subheader('Evaluation on uploaded dataset')
            st.dataframe(eval_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            'Download predictions CSV',
            data=csv_bytes,
            file_name='predictions_with_decision_tree.csv',
            mime='text/csv'
        )

    except Exception as exc:
        st.error(str(exc))

st.markdown('---')
st.write('Model bundle fields used by this app: model, feature_cols, training_columns')
