# Team15_IT5006_Predicive_Policing_AY2526Sem2

## Phase 1 Work (Milestone 1)

This project focuses on the comprehensive analysis of **2.9 million** original Chicago crime records. After deriving key insights through exploratory data analysis, the findings are presented via an interactive dashboard deployed on **Streamlit Cloud**.

### Cloud-safe Implementation
The choice of Streamlit was driven by its free cloud deployment space and easy deployment way, allowing us to focus our efforts on data analysis.

To overcome the resource and caching limitations of the free Streamlit Cloud tier, we have implemented the following strategy:
* **Remote Data Hosting**: The processed dataset is exported to and hosted on **Hugging Face** (`Ayanamikus/chicago-crime`).
* **Optimized Loading**: The application fetches data from Hugging Face and utilizes the `pyarrow` engine to create a local Parquet cache, ensuring high performance on the cloud platform.

---

## Phase 2 Work (Milestone 2)

This milestone focuses on **arrest prediction modeling** using the Chicago crime dataset.  
Based on the cleaned and engineered features from the previous phase, we built and evaluated multiple classification models to predict whether an incident results in an arrest.

### Milestone 2 includes:
* **Feature Engineering**: location, time, and crime-type related variables were transformed into model-ready features.
* **Model Training**: Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting were implemented and compared.
* **Model Evaluation**: performance was assessed using Accuracy, Precision, Recall, F1-score, and AUC-ROC.
* **Robustness Checks**: cross-validation, hyperparameter tuning, and subgroup evaluation by time and location were conducted.
* **Interpretation**: feature importance analysis was used to identify the most influential predictors.

The main notebook for this phase is stored in the `notebooks/` folder.

---

## Project Structure
**`data/`** Used for storing the raw datasets we use.
 
**`src/`** Deployment source code for the Streamlit application, including `Dashboard.py`. 
 
**`notebooks/`** Project notebooks for both milestones, including Phase 1 exploratory data analysis (EDA) and Phase 2 predictive modeling notebooks.
 
**`requirements.txt`** Project dependency list.

---

## Testing Procedures

Follow these steps to test the project in your local environment:

### 1. Clone the repository
```bash
git clone https://github.com/SaberAlterP/Team15_IT5006_Predicive_Policing_AY2526Sem2.git

cd [Project Folder Name]
```

### 2. Install dependencies

It is recommended to use a virtual environment or Conda environment:

```bash
pip install -r requirements.txt
```

### 3. Run the analysis notebooks

Navigate to the `notebooks/` directory and run the `.ipynb` files to review:
* **Milestone 1**: exploratory data analysis of the 2.9 million crime records
* **Milestone 2**: feature engineering, arrest prediction modeling, and evaluation

### 4. Run the application

* **4.1 Local Deployment**:  
run:
```bash
streamlit run src/Dashboard.py
```

* **4.2 Cloud Version**:  
Access our live interactive dashboard here:  
[https://team15it5006predicivepolicingay2526sem2-4oumidjohqqhlhp6p2vvkt.streamlit.app/]
```