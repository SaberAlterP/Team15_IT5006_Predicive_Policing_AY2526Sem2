# Decision Tree Streamlit Deployment

This folder is a minimal GitHub-ready deployment package for your Chicago arrest prediction Decision Tree model.

## Files
- `app.py`: Streamlit app
- `requirements.txt`: Python dependencies
- `runtime.txt`: Python version for cloud deployment
- `save_bundle_from_notebook.py`: code to export the trained model bundle

## 1. Export the trained model from your notebook
Run the code inside `save_bundle_from_notebook.py` after you finish training your final Decision Tree model.

This will generate:
- `decision_tree_bundle.pkl`

Put that file in the same folder as `app.py`.

## 2. Push to GitHub
Upload these files to your repo root:
- `app.py`
- `requirements.txt`
- `runtime.txt`
- `decision_tree_bundle.pkl`

## 3. Deploy on Streamlit Cloud
Set:
- Repository: your GitHub repo
- Branch: main
- Main file path: `app.py`

## Expected raw input columns
The uploaded CSV should contain these columns:
- `UNIFIED_LOCATION_CODE`
- `month`
- `day`
- `hour`
- `weekday`
- `is_weekend`
- `is_holiday`
- `crime_against`
- `offense_category_name`

Optional evaluation label column:
- `arrest` or `Arrest`
