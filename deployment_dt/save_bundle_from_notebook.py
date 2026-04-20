import joblib

# Choose the model you want to deploy.
# If your final model is best_dt_model, keep the line below.
model_to_deploy = best_dt_model

feature_cols = [
    'UNIFIED_LOCATION_CODE',
    'month',
    'day',
    'hour',
    'weekday',
    'is_weekend',
    'is_holiday',
    'crime_against',
    'offense_category_name'
]

bundle = {
    'model': model_to_deploy,
    'model_name': 'Decision Tree',
    'feature_cols': feature_cols,
    'training_columns': list(X_train.columns)
}

joblib.dump(bundle, 'decision_tree_bundle.pkl')
print('Saved: decision_tree_bundle.pkl')
