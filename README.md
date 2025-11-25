# Hospital Readmission Prediction for Diabetic Patients

## Project Overview

This project applies the complete data science lifecycle to predict hospital readmissions for diabetic patients using the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository. The dataset includes over 100,000 patient encounters across 130 U.S. hospitals, featuring demographic information, hospital utilization metrics, diagnoses, and medication histories.

**Primary Goal:** Build predictive models to classify whether a patient will be readmitted to the hospital (binary classification: readmitted vs. not readmitted).

**Tools & Technologies:**
- **Python 3.10** (primary programming language)
- **pandas** (data manipulation and analysis)
- **NumPy** (numerical computing)
- **scikit-learn** (machine learning models and evaluation)
- **matplotlib & seaborn** (data visualization)
- **Streamlit** (interactive dashboard for stakeholder findings)
- **Conda** (environment and dependency management)

---

## Environment Setup & Reproducibility

### Prerequisites
- **Conda** (Anaconda or Miniconda installed)
- **Git** (for cloning the repository)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd class_project
   ```

2. **Create the Conda environment from `environment.yml`:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate CAP3764_PROJECT
   ```

4. **Verify installation:**
   ```bash
   python --version  # Should display Python 3.10.x
   conda list        # Check installed packages
   ```

---

## Repository Structure

```
class_project/
│
├── data/
│   └── Diabetes 130 US Hospitals 1999-2008/
│       ├── diabetic_data.csv           # Primary dataset (~102K encounters)
│       └── IDS_mapping.csv             # ID mappings for categorical variables
│
├── notebooks/
│   ├── main.ipynb                       # Final modeling pipeline (Baseline + RF)
│   ├── initial_eda.ipynb                # Exploratory data analysis
│   ├── EDA_initial modeling.ipynb       # Extended EDA with feature insights
│   ├── decision_tree.pdf                # Decision tree visualization (PDF)
│   ├── decision_tree_stakeholder.pdf    # Stakeholder-facing decision tree
│   └── decision_tree.dot                # Graphviz DOT file for decision tree
│
├── utils/
│   ├── data_helper.py                   # Helper functions for loading & encoding data
│   └── __pycache__/                     # Python bytecode cache
│
├── Final_Insights.py                    # Streamlit app for interactive findings
├── visualization_additions.ipynb        # Additional visualizations
├── environment.yml                      # Conda environment specification
└── README.md                            # Project documentation (this file)
```

---

## Data Science Lifecycle

### 1. Data Collection & Cleaning

- **Data Source:** UCI Machine Learning Repository – Diabetes 130-US Hospitals dataset (1999–2008).
- **Size:** 101,766 patient encounters across 50 features.
- **Cleaning Steps:**
  - Loaded raw data using `data_helper.py`
  - Handled missing values (e.g., weight, medical specialty largely missing)
  - Encoded categorical variables using `pd.factorize()` for numerical compatibility
  - Mapped target variable `readmitted` to binary: `0` (not readmitted) vs. `1` (readmitted within or after 30 days)

**Key Files:**
- `utils/data_helper.py` – `load_dataset_and_encode()` function
- `notebooks/initial_eda.ipynb` – Initial data inspection and summary statistics

---

### 2. Exploratory Data Analysis (EDA)

**Objectives:**
- Understand feature distributions and correlations
- Identify potential predictors of readmission
- Detect outliers and rare values

**Key Findings:**
- **Target Class Imbalance:** Dataset is mildly imbalanced (not readmitted vs. readmitted).
- **Top Correlated Features with Readmission:**
  - `number_inpatient` (prior inpatient visits)
  - `number_emergency` (prior emergency visits)
  - Medication changes and combinations (e.g., sulfonylureas, thiazolidinediones)
- **Sparse Features:** Many medication fields have low usage counts, reflecting rare prescriptions in 1999–2008 practice.

**Visualizations:**
- Histograms and boxplots for numeric features
- Correlation heatmap (numeric variables)
- Countplots for categorical distributions (gender, age, race)

**Key Files:**
- `notebooks/initial_eda.ipynb`
- `notebooks/EDA_initial modeling.ipynb`

---

### 3. Feature Engineering

**Engineered Features:**
- `prior_visits` = `number_outpatient` + `number_emergency` + `number_inpatient`
- `emergency_to_inpatient_ratio` = `number_emergency` / (`number_inpatient` + 1)
- `outpatient_proportion` = `number_outpatient` / (`prior_visits` + 1)
- `long_stay` = Binary indicator if `time_in_hospital` > median
- `multiple_medicines` = Binary indicator if `num_medications` > 1
- `lab_intensity` = `num_lab_procedures` / `time_in_hospital`
- `procedures_per_day` = `num_procedures` / `time_in_hospital`

**Selected Features for Modeling:**
- Medication indicators: `acetohexamide`, `tolazamide`, `glimepiride-pioglitazone`, `metformin-pioglitazone`, `metformin-rosiglitazone`, `chlorpropamide`, `miglitol`
- Hospital utilization: `number_emergency`, `number_inpatient`, `prior_visits`
- Engineered ratios: `emergency_to_inpatient_ratio`, `outpatient_proportion`, `lab_intensity`, `procedures_per_day`
- Other: `weight`, `long_stay`, `multiple_medicines`

**Key Files:**
- `notebooks/main.ipynb` (Cells 5–6)

---

### 4. Modeling & Evaluation

**Data Splitting:**
- **Training Set:** 60% (n ≈ 61,000)
- **Validation Set:** 20% (n ≈ 20,000)
- **Test Set:** 20% (n ≈ 20,000)

**Preprocessing:**
- Standardized numeric features using `StandardScaler`
- Applied class balancing (`class_weight='balanced'`) to handle imbalance

---

#### Model 1: Baseline Logistic Regression

**Configuration:**
- Algorithm: Logistic Regression
- Hyperparameters: `class_weight='balanced'`, `max_iter=1000`

**Performance (Test Set):**
| Metric          | Class 0 (No Readmit) | Class 1 (Readmit) |
|-----------------|----------------------|-------------------|
| Precision       | 0.63                 | 0.60              |
| Recall          | 0.70                 | 0.52              |
| F1-Score        | 0.66                 | 0.56              |
| **ROC-AUC**     | **0.642**            |                   |
| **Accuracy**    | **0.62**             |                   |

**Interpretation:**
- Baseline model achieves moderate discrimination (AUC ≈ 0.64).
- Better at identifying non-readmissions (recall 0.70) than readmissions (recall 0.52).
- Top positive coefficients: `number_inpatient`, `emergency_to_inpatient_ratio`
- Top negative coefficients: `procedures_per_day`, `metformin-rosiglitazone`

---

#### Model 2: Random Forest Classifier

**Configuration:**
- Algorithm: Random Forest
- Hyperparameter Tuning: GridSearchCV with 3-fold cross-validation
- Search Space:
  - `n_estimators`: [100, 200]
  - `max_depth`: [5, 10, None]
  - `min_samples_leaf`: [1, 5, 10]

**Best Hyperparameters:**
- Determined via ROC-AUC scoring on validation set (see `notebooks/main.ipynb` Cell 15)

**Performance (Test Set):**
| Metric          | Class 0 (No Readmit) | Class 1 (Readmit) |
|-----------------|----------------------|-------------------|
| Precision       | 0.64                 | 0.59              |
| Recall          | 0.68                 | 0.55              |
| F1-Score        | 0.66                 | 0.57              |
| **ROC-AUC**     | **0.648**            |                   |
| **Accuracy**    | **0.62**             |                   |

**Interpretation:**
- Random Forest slightly outperforms baseline (AUC: 0.648 vs. 0.642).
- Improved recall for readmissions (0.55 vs. 0.52).
- Top features by importance: `number_inpatient`, `prior_visits`, `emergency_to_inpatient_ratio`

---

### 5. Reporting & Conclusions

**Comparison Summary:**

| Model                  | Test ROC-AUC | Accuracy | Recall (Readmit) | F1 (Readmit) |
|------------------------|--------------|----------|------------------|--------------|
| Logistic Regression    | 0.642        | 0.62     | 0.52             | 0.56         |
| **Random Forest**      | **0.648**    | **0.62** | **0.55**         | **0.57**     |

**Key Takeaways:**
- Random Forest provides a **modest but consistent improvement** over the baseline.
- **Prior hospital utilization** (inpatient and emergency visits) is the strongest predictor of readmission.
- **Medication combinations** and **procedures per day** also contribute to risk stratification.
- Both models achieve similar overall accuracy (~62%), indicating room for further optimization (e.g., additional features, ensemble methods, deep learning).

**Visualizations:**
- ROC curves comparing baseline vs. Random Forest
- Confusion matrices for both models
- Feature importance plot for Random Forest

**Key Files:**
- `notebooks/main.ipynb` (Cells 12–20)

---

## Key Findings

### Clinical & Operational Insights

1. **High-Risk Patient Profile:**
   - Patients with **> 15 prior inpatient visits** or **> 40 emergency visits** have significantly higher readmission rates.
   - Increased `emergency_to_inpatient_ratio` signals heavy reliance on acute care.

2. **Protective Factors:**
   - Higher `procedures_per_day` correlates with lower readmission risk (possible indicator of thorough treatment).
   - Certain medication combinations (e.g., `metformin-rosiglitazone`) show slight protective effect.

3. **Medication Patterns:**
   - Many medication fields (e.g., `acetohexamide`, `glimepiride-pioglitazone`) have sparse usage, reflecting rare prescriptions in this historical dataset.
   - Presence of specific medications can flag high-risk patients but requires larger cohorts for robust inference.

4. **Model Performance:**
   - **Baseline Logistic Regression:** AUC ≈ 0.64 (moderate discrimination)
   - **Random Forest:** AUC ≈ 0.65 (slight improvement, better recall for readmissions)
   - Both models demonstrate **modest predictive power**, suggesting that readmission is influenced by complex, multifactorial causes not fully captured by available features.

---

## How to Reproduce Results

### Step 1: Set Up Environment

```bash
# Clone repository
git clone <repository-url>
cd class_project

# Create Conda environment
conda env create -f environment.yml
conda activate CAP3764_PROJECT
```

---

### Step 2: Run Exploratory Data Analysis

```bash
# Launch Jupyter
jupyter notebook

# Open and run:
# - notebooks/initial_eda.ipynb (basic EDA)
# - notebooks/EDA_initial modeling.ipynb (extended EDA)
```

**Expected Outputs:**
- Summary statistics, distributions, correlation heatmaps
- Feature-level insights for readmission prediction

---

### Step 3: Run Main Modeling Pipeline

```bash
# Open notebooks/main.ipynb in Jupyter
jupyter notebook notebooks/main.ipynb

# Execute all cells sequentially
```

**Key Steps in Notebook:**
1. Load and encode data (`data_helper.py`)
2. Apply feature engineering
3. Split data (train/validation/test)
4. Train baseline Logistic Regression
5. Train Random Forest with GridSearchCV
6. Evaluate on test set
7. Generate ROC curves, confusion matrices, feature importances

**Expected Outputs:**
- Baseline Test ROC-AUC: **~0.642**
- Random Forest Test ROC-AUC: **~0.648**
- Classification reports, confusion matrices, ROC curves

---

### Step 4: Launch Interactive Dashboard (Optional)

```bash
# Run the Streamlit app for stakeholder findings
streamlit run Final_Insights.py
```

**Features:**
- Dataset overview and summary statistics
- Interactive visualizations (distributions, correlations)
- "Findings to the Eye" section with conditional readmission rates
- Medical glossary for key terms

---

## Team Information

- **Course:** CAP 3764 – Data Science & Machine Learning
- **Project:** Hospital Readmission Prediction for Diabetic Patients
- **Team:** Team 3
- **Date:** Fall 2025 (November 2025)
- **Dataset:** UCI Machine Learning Repository – Diabetes 130-US Hospitals (1999–2008)

---

## Future Work & Improvements

1. **Feature Expansion:**
   - Incorporate primary diagnosis codes (ICD-9) for disease severity
   - Add temporal features (e.g., seasonality, time since last visit)

2. **Advanced Modeling:**
   - Experiment with XGBoost, LightGBM, or Neural Networks
   - Explore ensemble methods (stacking, blending)

3. **Class Imbalance Handling:**
   - Apply SMOTE (Synthetic Minority Over-sampling Technique)
   - Adjust decision thresholds for precision-recall trade-offs based on healthcare industry standards

---

## References

- **Dataset Source:** Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International. [UCI Repository Link](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **scikit-learn Documentation:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/
- **Streamlit Documentation:** https://docs.streamlit.io/

---

## License

This project is for educational purposes as part of CAP 3764. All data and code are used in accordance with the UCI Machine Learning Repository's terms of use.

---

## Contact

For questions or collaboration inquiries, please contact the team via the repository maintainer.

