import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("data/Diabetes 130 US Hospitals 1999-2008/diabetic_data.csv")
    ids_mapping = pd.read_csv("data/Diabetes 130 US Hospitals 1999-2008/IDS_mapping.csv")
    return df, ids_mapping

def build_encoded(df):
    df_encoded = df.copy()
    mappings = {}
    for col in df_encoded.select_dtypes(exclude="number").columns:
        df_encoded[col], uniques = pd.factorize(df_encoded[col])
        mappings[col] = dict(enumerate(uniques))
    return df_encoded, mappings

df, ids_mapping = load_data()
df_encoded, mappings = build_encoded(df)

st.set_page_config(page_title="Diabetes Findings Showcase", layout="wide")
st.title("Diabetes Dataset Findings Showcase")

st.sidebar.header("Navigation")
option = st.sidebar.radio(
    "Choose a section:",
    ["Overview & Key Stats", "Distributions & Insights", "Correlation Heatmap", "Findings to the Eye", "Stakeholder Findings", "Medical Glossary"]
)

df_model_cols = [
    "acetohexamide","tolazamide","glimepiride-pioglitazone","metformin-pioglitazone",
    "metformin-rosiglitazone","weight","number_emergency","number_inpatient",
    "chlorpropamide","miglitol"
]
df_model_cols = [c for c in df_model_cols if c in df_encoded.columns]
df_model = df_encoded[df_model_cols].copy()

def _readmit_flag(series):
    return series.astype(str).str.upper().isin({">30", "<30"})

def _exists_e(colname):
    return colname in df_encoded.columns or colname.replace("-", "_").replace(" ", "_") in df_encoded.columns

def _col_e(colname):
    return colname if colname in df_encoded.columns else colname.replace("-", "_").replace(" ", "_")

def _equals_in_encoded(colname, value):
    if not _exists_e(colname):
        return pd.Series(False, index=df_encoded.index)
    c = _col_e(colname)
    return df_encoded[c] == value

def _not_missing_encoded(colname):
    if not _exists_e(colname):
        return pd.Series(False, index=df_encoded.index)
    c = _col_e(colname)
    return df_encoded[c] != -1

def _summarize_condition(mask, label, non_readmit=False):
    subset_idx = df_encoded.index[mask]
    total = len(subset_idx)
    if total == 0:
        st.metric(label, "n=0", "readmit: n/a" if not non_readmit else "non-readmit: n/a")
        return
    rflag = _readmit_flag(df.loc[subset_idx, "readmitted"]) if "readmitted" in df.columns else pd.Series([None]*total, index=subset_idx)
    if non_readmit:
        val = 1 - rflag.mean()
        st.metric(label, f"n={total}", f"non-readmit: {val:.1%}" if pd.notnull(val) else "non-readmit: n/a")
    else:
        val = rflag.mean()
        st.metric(label, f"n={total}", f"readmit: {val:.1%}" if pd.notnull(val) else "readmit: n/a")

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
    for col in cat_cols[:5]:
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

# --- Section 4: Findings to the Eye ---
elif option == "Findings to the Eye":
    st.subheader("Findings to the Eye (UCI Diabetes Readmissions, after mappings)")
    if "readmitted" not in df.columns:
        st.error("Column 'readmitted' not found.")
    else:
        cols = st.columns(2)
        with cols[0]:
            if "number_emergency" in df.columns:
                mask = df["number_emergency"] > 40
                _summarize_condition(mask, "number_emergency > 40 → readmitted")
            else:
                st.info("number_emergency not found.")
            if "number_inpatient" in df.columns:
                mask = df["number_inpatient"] > 15
                _summarize_condition(mask, "number_inpatient > 15 → readmitted")
            else:
                st.info("number_inpatient not found.")
            if _exists_e("acetohexamide"):
                mask = _equals_in_encoded("acetohexamide", 1)
                _summarize_condition(mask, "acetohexamide == 1 → readmitted")
            else:
                st.info("acetohexamide not found.")
            if _exists_e("tolazamide"):
                mask = _equals_in_encoded("tolazamide", 2)
                _summarize_condition(mask, "tolazamide == 2 → readmitted")
            else:
                st.info("tolazamide not found.")
        with cols[1]:
            if _exists_e("glimepiride-pioglitazone"):
                mask = _equals_in_encoded("glimepiride-pioglitazone", 1)
                _summarize_condition(mask, "glimepiride-pioglitazone == 1 → readmitted")
            else:
                st.info("glimepiride-pioglitazone not found.")
            if _exists_e("metformin-rosiglitazone"):
                mask = _equals_in_encoded("metformin-rosiglitazone", 1)
                _summarize_condition(mask, "metformin-rosiglitazone == 1 → NOT readmitted", non_readmit=True)
            else:
                st.info("metformin-rosiglitazone not found.")
            if _exists_e("metformin-pioglitazone"):
                mask = _equals_in_encoded("metformin-pioglitazone", 1)
                _summarize_condition(mask, "metformin-pioglitazone == 1 → NOT readmitted", non_readmit=True)
            else:
                st.info("metformin-pioglitazone not found.")
            if _exists_e("chlorpropamide") or _exists_e("clorpropamide"):
                cname = "chlorpropamide" if _exists_e("chlorpropamide") else "clorpropamide"
                mask = _equals_in_encoded(cname, 1)
                _summarize_condition(mask, f"{cname} == 1 → NOT readmitted", non_readmit=True)
            else:
                st.info("chlorpropamide/clorpropamide not found.")

        st.markdown("**Weight snapshot**")
        if _exists_e("weight"):
            wmask_any = _not_missing_encoded("weight")
            _summarize_condition(wmask_any, "Weight recorded (any) → readmitted")
            wmask_high = df_encoded[_col_e("weight")] > 8
            _summarize_condition(wmask_high, "Weight code > 8 → readmitted")
        else:
            st.info("weight not found.")

        with st.expander("Low-n feature summary (df_model)"):
            if not df_model.empty:
                counts = {c: df_model[c].value_counts(dropna=False).sort_index() for c in df_model.columns}
                rows = []
                for c, vc in counts.items():
                    rows.append({
                        "feature": c,
                        "unique_codes": int(vc.shape[0]),
                        "n_code_-1(missing)": int(vc.get(-1, 0)),
                        "n_code_0": int(vc.get(0, 0)),
                        "n_code_1": int(vc.get(1, 0)),
                        "n_code_2": int(vc.get(2, 0))
                    })
                st.dataframe(pd.DataFrame(rows))
            cols_show = [
                "weight","acetohexamide","tolazamide","glimepiride-pioglitazone",
                "metformin-rosiglitazone","metformin-pioglitazone","chlorpropamide","clorpropamide","miglitol"
            ]
            map_rows = []
            for c in cols_show:
                key = c if c in mappings else _col_e(c)
                if key in mappings:
                    m = mappings[key]
                    pretty = {int(k): (str(v) if pd.notna(v) else "NaN") for k, v in m.items()}
                    map_rows.append({"Column": c, "Mapping (code → original)": pretty})
            if map_rows:
                st.dataframe(pd.DataFrame(map_rows))
            st.markdown("""
            When certain codes have very small counts, this **suggests** those features are **outliers** in this dataset.
            A plausible medical explanation is that some therapies/combinations were **rare** or **sparsely documented** in 1999–2008 practice.
            This is a **preliminary** analysis and will be improved in the next project submission with larger cohorts and model-based inference.
            """)

# --- Section 5: Stakeholder Findings ---
elif option == "Stakeholder Findings":
    st.subheader("Stakeholder Findings")
    if "readmitted" not in df.columns:
        st.error("Column 'readmitted' not found.")
    else:
        def rate_for(mask, non_readmit=False):
            idx = df_encoded.index[mask]
            if len(idx) == 0:
                return float("nan"), 0
            r = _readmit_flag(df.loc[idx, "readmitted"]).mean()
            if non_readmit:
                return (1 - r), len(idx)
            return r, len(idx)

        cards = st.columns(3)
        with cards[0]:
            m1 = (df["number_emergency"] > 40) if "number_emergency" in df.columns else pd.Series(False, index=df_encoded.index)
            r1, n1 = rate_for(m1)
            st.metric("High ED users (>40)", f"n={n1}", f"readmit: {r1:.1%}" if pd.notnull(r1) else "readmit: n/a")
        with cards[1]:
            m2 = (df["number_inpatient"] > 15) if "number_inpatient" in df.columns else pd.Series(False, index=df_encoded.index)
            r2, n2 = rate_for(m2)
            st.metric("High IP users (>15)", f"n={n2}", f"readmit: {r2:.1%}" if pd.notnull(r2) else "readmit: n/a")
        with cards[2]:
            m3 = _equals_in_encoded("glimepiride-pioglitazone", 1)
            r3, n3 = rate_for(m3)
            st.metric("Glimepiride+Pioglitazone == 1", f"n={n3}", f"readmit: {r3:.1%}" if pd.notnull(r3) else "readmit: n/a")

        st.markdown("**Additional weight signal**")
        if _exists_e("weight"):
            m4 = df_encoded[_col_e("weight")] > 8
            r4, n4 = rate_for(m4)
            st.metric("Weight code > 8", f"n={n4}", f"readmit: {r4:.1%}" if pd.notnull(r4) else "readmit: n/a")
        else:
            st.info("weight not found.")

        st.markdown("""
        **Interpretation (preliminary)**
        - Very small **n** on some medication flags likely reflects **rare utilization**; treat rates as directional only.
        - High prior utilization (ED > 40, IP > 15) flags a cohort for enhanced discharge planning and early follow-up.
        - Encoded **weight > 8** shows a higher readmission share in this project’s mapping; decode the codes in the table above to see which ranges they represent.
        - Findings will be **expanded and revalidated** in the next project submission.
        """)

# --- Section 6: Medical Glossary ---
elif option == "Medical Glossary":
    st.subheader("Medical Glossary (Selected Terms)")
    glossary = pd.DataFrame({
        "Term": [
            "number_emergency",
            "number_inpatient",
            "weight",
            "acetohexamide",
            "tolazamide",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
            "chlorpropamide",
            "miglitol"
        ],
        "Meaning": [
            "Count of prior emergency department encounters during the look-back window.",
            "Count of prior inpatient admissions during the look-back window.",
            "Body weight recorded in the admission record; often sparsely documented in this dataset.",
            "Sulfonylurea antihyperglycemic agent that stimulates pancreatic insulin release.",
            "Sulfonylurea antihyperglycemic agent; similar mechanism to other sulfonylureas.",
            "Combination of sulfonylurea (glimepiride) with thiazolidinedione (pioglitazone).",
            "Combination of biguanide (metformin) with thiazolidinedione (rosiglitazone).",
            "Combination of biguanide (metformin) with thiazolidinedione (pioglitazone).",
            "First-generation sulfonylurea; promotes insulin secretion.",
            "Alpha-glucosidase inhibitor; reduces postprandial glucose absorption."
        ],
        "Encoding in this app": [
            "Numeric integer count (unchanged)",
            "Numeric integer count (unchanged)",
            "Factorized integer code (−1 means missing)",
            "Factorized integer code",
            "Factorized integer code",
            "Factorized integer code",
            "Factorized integer code",
            "Factorized integer code",
            "Factorized integer code",
            "Factorized integer code"
        ]
    })
    st.dataframe(glossary, use_container_width=True)
    st.markdown("""
    **Notes**
    - Many categorical fields were factorized into integer codes using project-specific mappings.
    - Low counts for some codes likely indicate rarer therapies or documentation patterns in 1999–2008 practice.
    - This is preliminary and will be enhanced in the next submission with model-based estimates and uncertainty.
    """)
