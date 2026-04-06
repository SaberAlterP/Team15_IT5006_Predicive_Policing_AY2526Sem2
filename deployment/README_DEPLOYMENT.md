# Arrest Prediction POC - Deployment Notes

## Recommended platform
Use **Streamlit Community Cloud** for the POC because it gives the team a public URL quickly and already matches the assignment requirement for a simple prediction interface.

## Files to keep in the GitHub repo
- `app.py`
- `requirements.txt`
- `nibrs_xgb_model.json`
- `nibrs_model_metadata.json`

## Streamlit Cloud deployment steps
1. Push the above files to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Click **Create app**.
4. Select the GitHub repo and choose `app.py` as the entry point.
5. Deploy the app.
6. After deployment, copy the generated `streamlit.app` URL.

## Four screenshots you should take for the report
1. **Live URL screenshot**  
   Show the deployed page in the browser with the visible URL.
2. **Prediction form screenshot**  
   Show the input widgets before prediction.
3. **Prediction result screenshot**  
   Fill in one example and show the probability plus the predicted result.
4. **Error handling screenshot**  
   Upload a wrong CSV or remove a required column, then capture the validation error message.

## Suggested caption text for the report
- **Live deployment:** The POC was deployed on Streamlit Community Cloud and made accessible via a public URL.
- **User interface:** The application provides both single-record prediction and batch CSV prediction.
- **Core functionality:** The system outputs arrest probability and the final binary prediction.
- **Validation:** The application checks for missing columns, invalid categorical values, and malformed numeric inputs.

## Important note
If the team must deploy the **exact Chicago-trained model**, you still need the saved model artifact (for example, `.pkl`, `.joblib`, or `.json`) and its fitted preprocessing pipeline. The uploaded Chicago notebook shows how the model was trained, but it does not contain a serialized trained model file.
