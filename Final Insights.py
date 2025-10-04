import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Diabetes 130 US Hospitals 1999-2008/diabetic_data.csv")
    ids_mapping = pd.read_csv("data/Diabetes 130 US Hospitals 1999-2008/IDS_mapping.csv")
    return df, ids_mapping

df, ids_mapping = load_data()

# App title
st.set_page_config(page_title="Diabetes Findings Showcase", layout="wide")
st.title("Diabetes Dataset Findings Showcase")

# Sidebar options
st.sidebar.header("Navigation")
option = st.sidebar.radio(
    "Choose a section:",
    ["Overview & Key Stats", "Distributions & Insights", "Correlation Heatmap"]
)

# --- Section 1: Overview & Key Stats ---
if option == "Overview & Key Stats":
    st.subheader("Dataset Overview")
    st.write("Rows:", df.shape[0], "| Columns:", df.shape[1])
    
    st.write("Sample Data")
    st.dataframe(df.head(10))
    
    st.subheader("Summary Statistics")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    st.dataframe(numeric_df.describe().T)
    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])
    
    st.subheader("Categorical Highlights")
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols[:5]:  # show top 5 categorical columns
        st.write(f"**{col}**")
        st.write(df[col].value_counts().head(5))

# --- Section 2: Distributions & Insights ---
elif option == "Distributions & Insights":
    st.subheader("Distributions of Selected Columns")
    col = st.selectbox("Select a column to visualize", df.columns)
    
    if df[col].dtype == "object":
        st.write(df[col].value_counts().head(20))
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index[:10], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write(df[col].describe())
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[col], kde=True, ax=ax, bins=30)
        st.pyplot(fig)
    
    st.subheader("Top Insights")
    st.markdown("""
    - Most patients are in certain admission types or demographics.
    - Common diagnoses and medications can be highlighted here.
    - Visual patterns in numeric features may indicate trends for further analysis.
    """)

# --- Section 3: Correlation Heatmap ---
elif option == "Correlation Heatmap":
    st.subheader("Correlation Heatmap (Numeric Columns Only)")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation.")
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)
        
        st.markdown("""
        **Key correlations to note:**
        - Strong positive/negative correlations can indicate predictors for outcomes.
        - Use these insights to guide further modeling or feature selection.
        """)
