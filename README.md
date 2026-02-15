# Team15_IT5006_Predicive_Policing_AY2526Sem2

##  Phase 1 Work (Milestone 1)

This project focuses on the comprehensive analysis of **2.9 million** original Chicago crime records. After deriving key insights through exploratory data analysis, the findings are presented via an interactive dashboard deployed on **Streamlit Cloud**.

### Cloud-safe Implementation
The choice of Streamlit was driven by its free cloud deployment space and easy deployment way, allowing us to focus our efforts on data analysis.

To overcome the resource and caching limitations of the free Streamlit Cloud tier, we have implemented the following strategy:
* **Remote Data Hosting**: The processed dataset is exported to and hosted on **Hugging Face** (`Ayanamikus/chicago-crime`).
* **Optimized Loading**: The application fetches data from Hugging Face and utilizes the `pyarrow` engine to create a local Parquet cache, ensuring high performance on the cloud platform.

---

##  Project Structure
 **`data/`** Used for storing the raw datasets we use.
 **`src/`**  Deployment source code for the Streamlit application, including `app.py`. 
 **`notebooks/`**  Phase 1 data analysis notebooks, including exploratory data analysis (EDA) (e.g., `jzx-EDA.ipynb`). 
 **`requirements.txt`**  Project dependency list.

---

##  Testing Procedures

Follow these steps to test the project in your local environment:

### 1. Clone the repository
```bash
git clone [Your Repository URL]
cd [Project Folder Name]

```

### 2. Install dependencies

It is recommended to use a virtual environment or Conda environment:

```bash
pip install -r requirements.txt

```
### 3. Run the analysis notebooks

Navigate to the `notebook/` directory and run the `.ipynb` files to review our analysis of the 2.9 million records.

### 4. Run the application

* **4.1 Local Deployment**:
run:
```
streamlit run src/Dashboard.py
```
* **4.2 Cloud Version**:
Access our live interactive dashboard here:
[https://team15it5006predicivepolicingay2526sem2-4oumidjohqqhlhp6p2vvkt.streamlit.app/]
