"""
Streamlit Dashboard for Hospital Readmission Prediction
Interactive interface for single and batch predictions using FastAPI backend
"""

import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os

# API Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        "Low": "#28a745",
        "Medium": "#ffc107",
        "High": "#dc3545"
    }
    return colors.get(risk_level, "#6c757d")


def display_prediction_card(prediction_data, show_patient_id=False):
    """Display prediction results in a card format"""
    risk_color = get_risk_color(prediction_data['risk_level'])
    
    if show_patient_id and 'patient_id' in prediction_data:
        st.markdown(f"### Patient {prediction_data['patient_id']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Risk Level",
            prediction_data['risk_level'],
            help="Overall risk classification"
        )
    
    with col2:
        st.metric(
            "Random Forest Prediction",
            "Readmit" if prediction_data['rf_prediction'] == 1 else "No Readmit",
            f"{prediction_data['rf_probability']:.1%} probability"
        )
    
    with col3:
        st.metric(
            "Baseline Prediction",
            "Readmit" if prediction_data['baseline_prediction'] == 1 else "No Readmit",
            f"{prediction_data['baseline_probability']:.1%} probability"
        )
    
    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_data['rf_probability'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Readmission Risk (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 60], 'color': "lightyellow"},
                {'range': [60, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def single_prediction_page():
    """Single patient prediction interface"""
    st.title("Single Patient Prediction")
    st.markdown("Enter patient information to predict hospital readmission risk")
    
    with st.form("single_prediction_form"):
        st.subheader("Medication Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            acetohexamide = st.number_input("Acetohexamide", min_value=0.0, max_value=10.0, value=0.0, help="Encoded medication value")
            tolazamide = st.number_input("Tolazamide", min_value=0.0, max_value=10.0, value=0.0)
            chlorpropamide = st.number_input("Chlorpropamide", min_value=0.0, max_value=10.0, value=0.0)
        
        with col2:
            glimepiride_pioglitazone = st.number_input("Glimepiride-Pioglitazone", min_value=0.0, max_value=10.0, value=0.0)
            metformin_pioglitazone = st.number_input("Metformin-Pioglitazone", min_value=0.0, max_value=10.0, value=0.0)
            miglitol = st.number_input("Miglitol", min_value=0.0, max_value=10.0, value=0.0)
        
        with col3:
            metformin_rosiglitazone = st.number_input("Metformin-Rosiglitazone", min_value=0.0, max_value=10.0, value=0.0)
            weight = st.number_input("Weight (encoded)", min_value=0.0, max_value=10.0, value=1.0)
        
        st.subheader("Hospital Utilization")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            number_emergency = st.number_input("Emergency Visits", min_value=0.0, max_value=100.0, value=0.0)
            number_inpatient = st.number_input("Inpatient Visits", min_value=0.0, max_value=50.0, value=0.0)
        
        with col2:
            prior_visits = st.number_input("Total Prior Visits", min_value=0.0, max_value=150.0, value=0.0)
            emergency_to_inpatient_ratio = st.number_input("Emergency/Inpatient Ratio", min_value=0.0, max_value=100.0, value=0.0)
        
        with col3:
            outpatient_proportion = st.number_input("Outpatient Proportion", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        st.subheader("Clinical Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            long_stay = st.selectbox("Long Stay", [0, 1], help="1 if hospital stay > median")
        
        with col2:
            multiple_medicines = st.selectbox("Multiple Medicines", [0, 1], help="1 if taking multiple medicines")
        
        with col3:
            lab_intensity = st.number_input("Lab Intensity", min_value=0.0, max_value=50.0, value=10.0, help="Lab procedures per day")
        
        with col4:
            procedures_per_day = st.number_input("Procedures/Day", min_value=0.0, max_value=10.0, value=0.5)
        
        submitted = st.form_submit_button("Predict Readmission Risk", use_container_width=True)
        
        if submitted:
            # Prepare payload
            payload = {
                "acetohexamide": acetohexamide,
                "tolazamide": tolazamide,
                "glimepiride-pioglitazone": glimepiride_pioglitazone,
                "metformin-pioglitazone": metformin_pioglitazone,
                "metformin-rosiglitazone": metformin_rosiglitazone,
                "weight": weight,
                "number_emergency": number_emergency,
                "number_inpatient": number_inpatient,
                "chlorpropamide": chlorpropamide,
                "miglitol": miglitol,
                "prior_visits": prior_visits,
                "emergency_to_inpatient_ratio": emergency_to_inpatient_ratio,
                "outpatient_proportion": outpatient_proportion,
                "long_stay": long_stay,
                "multiple_medicines": multiple_medicines,
                "lab_intensity": lab_intensity,
                "procedures_per_day": procedures_per_day
            }
            
            with st.spinner("Making prediction..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    response.raise_for_status()
                    prediction = response.json()
                    
                    st.success("Prediction Complete!")
                    st.markdown("---")
                    display_prediction_card(prediction)
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error making prediction: {str(e)}")


def batch_prediction_page():
    """Batch prediction interface"""
    st.title("Batch Prediction")
    st.markdown("Upload a CSV file or enter multiple patients for batch predictions")
    
    tab1, tab2 = st.tabs(["Upload CSV", "Manual Entry"])
    
    with tab1:
        st.subheader("Upload Patient Data")
        st.info("Upload a CSV file with all required features. Download the template below if needed.")
        
        # Template download
        template_data = {
            "acetohexamide": [0.0],
            "tolazamide": [0.0],
            "glimepiride-pioglitazone": [0.0],
            "metformin-pioglitazone": [0.0],
            "metformin-rosiglitazone": [0.0],
            "weight": [1.0],
            "number_emergency": [2.0],
            "number_inpatient": [1.0],
            "chlorpropamide": [0.0],
            "miglitol": [0.0],
            "prior_visits": [4.0],
            "emergency_to_inpatient_ratio": [1.0],
            "outpatient_proportion": [0.25],
            "long_stay": [1],
            "multiple_medicines": [1],
            "lab_intensity": [10.5],
            "procedures_per_day": [0.5]
        }
        template_df = pd.DataFrame(template_data)
        
        st.download_button(
            label="Download CSV Template",
            data=template_df.to_csv(index=False),
            file_name="patient_template.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Option to load pre-saved test data
        st.subheader("Option 1: Use Pre-saved Test Data")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Load a sample batch of 20 patients from the test set")
        with col2:
            load_test_data = st.button("Load Test Data", use_container_width=True)
        
        if load_test_data:
            try:
                test_data_path = "data/readmission_batch_test/readmission_batch_test_sample.csv"
                df = pd.read_csv(test_data_path)
                
                # Remove the true label column if it exists (we don't need it for prediction)
                if 'readmitted_true' in df.columns:
                    true_labels = df['readmitted_true'].copy()
                    df = df.drop(columns=['readmitted_true'])
                    st.session_state['test_data'] = df
                    st.session_state['true_labels'] = true_labels
                else:
                    st.session_state['test_data'] = df
                    st.session_state['true_labels'] = None
                
                st.success(f"Loaded {len(df)} patients from test data")
                
            except FileNotFoundError:
                st.error("Test data file not found. Please ensure 'readmission_batch_test_sample.csv' exists in the data/readmission_batch_test/ folder.")
            except Exception as e:
                st.error(f"Error loading test data: {str(e)}")
        
        # Display and predict on loaded test data
        if 'test_data' in st.session_state:
            df = st.session_state['test_data']
            st.write(f"**Loaded {len(df)} patients from test data**")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("Run Batch Prediction on Test Data", use_container_width=True, key="predict_test"):
                with st.spinner("Processing batch predictions..."):
                    try:
                        # Convert DataFrame to CSV bytes
                        csv_bytes = df.to_csv(index=False).encode('utf-8')
                        response = requests.post(
                            f"{API_URL}/predict/upload",
                            files={'file': ('test_data.csv', csv_bytes, 'text/csv')}
                        )
                        response.raise_for_status()
                        results = response.json()
                        
                        # If we have true labels, add them to the results for comparison
                        if st.session_state.get('true_labels') is not None:
                            st.info("Note: This test data includes true readmission labels for comparison")
                        
                        display_batch_results(results)
                        
                        # Show comparison with true labels if available
                        if st.session_state.get('true_labels') is not None:
                            st.markdown("---")
                            st.subheader("Model Performance on Test Data")
                            
                            predictions_df = pd.DataFrame(results['predictions'])
                            true_labels = st.session_state['true_labels'].values
                            rf_preds = predictions_df['rf_prediction'].values
                            
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Accuracy", f"{accuracy_score(true_labels, rf_preds):.3f}")
                            col2.metric("Precision", f"{precision_score(true_labels, rf_preds):.3f}")
                            col3.metric("Recall", f"{recall_score(true_labels, rf_preds):.3f}")
                            col4.metric("F1-Score", f"{f1_score(true_labels, rf_preds):.3f}")
                    
                    except Exception as e:
                        st.error(f"Error processing test data: {str(e)}")
        
        st.markdown("---")
        
        # Option to upload custom file
        st.subheader("Option 2: Upload Your Own CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Remove true label if present
                if 'readmitted_true' in df.columns:
                    df = df.drop(columns=['readmitted_true'])
                
                st.write(f"**Loaded {len(df)} patients**")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Run Batch Prediction", use_container_width=True, key="predict_upload"):
                    with st.spinner("Processing batch predictions..."):
                        # Convert DataFrame to CSV bytes
                        csv_bytes = df.to_csv(index=False).encode('utf-8')
                        response = requests.post(
                            f"{API_URL}/predict/upload",
                            files={'file': ('data.csv', csv_bytes, 'text/csv')}
                        )
                        response.raise_for_status()
                        results = response.json()
                        
                        display_batch_results(results)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.subheader("Manual Batch Entry")
        st.info("Enter patient data manually (JSON format)")
        
        sample_json = [
            {
                "acetohexamide": 0.0,
                "tolazamide": 0.0,
                "glimepiride-pioglitazone": 0.0,
                "metformin-pioglitazone": 0.0,
                "metformin-rosiglitazone": 0.0,
                "weight": 1.0,
                "number_emergency": 2.0,
                "number_inpatient": 1.0,
                "chlorpropamide": 0.0,
                "miglitol": 0.0,
                "prior_visits": 4.0,
                "emergency_to_inpatient_ratio": 1.0,
                "outpatient_proportion": 0.25,
                "long_stay": 1,
                "multiple_medicines": 1,
                "lab_intensity": 10.5,
                "procedures_per_day": 0.5
            }
        ]
        
        json_input = st.text_area(
            "Enter patient data as JSON array",
            value=json.dumps(sample_json, indent=2),
            height=300
        )
        
        if st.button("Run Batch Prediction (JSON)", use_container_width=True):
            try:
                patients = json.loads(json_input)
                
                with st.spinner("Processing batch predictions..."):
                    response = requests.post(f"{API_URL}/predict/batch", json=patients)
                    response.raise_for_status()
                    results = response.json()
                    
                    display_batch_results(results)
            
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your input.")
            except Exception as e:
                st.error(f"Error: {str(e)}")


def display_batch_results(results):
    """Display batch prediction results"""
    st.success("Batch Prediction Complete!")
    st.markdown("---")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    summary = results['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", summary['total_patients'])
    col2.metric("Predicted Readmissions", summary['predicted_readmissions'])
    col3.metric("Predicted No Readmissions", summary['predicted_no_readmissions'])
    col4.metric("Avg Risk Probability", f"{summary['avg_readmission_probability']:.1%}")
    
    # Risk distribution
    st.subheader("Risk Distribution")
    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk", summary['high_risk_count'], delta=None, delta_color="inverse")
    col2.metric("Medium Risk", summary['medium_risk_count'])
    col3.metric("Low Risk", summary['low_risk_count'], delta=None, delta_color="normal")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk level pie chart
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Count': [summary['low_risk_count'], summary['medium_risk_count'], summary['high_risk_count']]
        })
        fig_pie = px.pie(
            risk_data,
            values='Count',
            names='Risk Level',
            title='Risk Level Distribution',
            color='Risk Level',
            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Prediction distribution
        pred_data = pd.DataFrame({
            'Prediction': ['No Readmit', 'Readmit'],
            'Count': [summary['predicted_no_readmissions'], summary['predicted_readmissions']]
        })
        fig_bar = px.bar(
            pred_data,
            x='Prediction',
            y='Count',
            title='Prediction Distribution',
            color='Prediction',
            color_discrete_map={'No Readmit': '#28a745', 'Readmit': '#dc3545'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Predictions")
    predictions_df = pd.DataFrame(results['predictions'])
    st.dataframe(predictions_df, use_container_width=True)
    
    # Download results
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="batch_predictions.csv",
        mime="text/csv"
    )


def about_page():
    """About page with model information"""
    st.title("About")
    
    st.markdown("""
    ## Hospital Readmission Prediction System
    
    This application predicts hospital readmission risk for diabetic patients using machine learning models.
    
    ### Models
    
    **Baseline Model: Logistic Regression**
    - Test ROC-AUC: 0.642
    - Accuracy: 62%
    - Interpretable linear model
    
    **Advanced Model: Random Forest**
    - Test ROC-AUC: 0.648
    - Accuracy: 62%
    - Better recall for readmissions
    - Captures non-linear relationships
    
    ### Dataset
    
    - **Source:** Diabetes 130-US Hospitals for Years 1999-2008 (UCI Repository)
    - **Size:** 101,766 patient encounters
    - **Features:** 17 selected features including medications and hospital utilization
    
    ### Key Features
    
    **Medication Indicators:**
    - Sulfonylureas (acetohexamide, tolazamide, chlorpropamide)
    - Combination therapies (glimepiride-pioglitazone, metformin combinations)
    - Alpha-glucosidase inhibitors (miglitol)
    
    **Hospital Utilization:**
    - Emergency visits, inpatient visits, prior visits
    - Emergency-to-inpatient ratio
    - Outpatient proportion
    
    **Clinical Indicators:**
    - Long stay indicator
    - Multiple medicines indicator
    - Lab intensity (procedures per day)
    - Procedures per day
    
    ### Risk Levels
    
    - **Low Risk:** < 40% readmission probability
    - **Medium Risk:** 40-60% readmission probability
    - **High Risk:** > 60% readmission probability
    
    ### API Endpoints
    
    - `GET /health` - Check API health
    - `POST /predict` - Single patient prediction
    - `POST /predict/batch` - Batch prediction (JSON)
    - `POST /predict/upload` - Batch prediction (CSV upload)
    - `GET /features` - List required features
    
    ### Team Information
    
    - **Course:** CAP 3764 – Data Science & Machine Learning
    - **Team:** Team 3
    - **Date:** Fall 2025
    
    ---
    
    For questions or issues, please contact the repository maintainer.
    """)


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=80)
        st.title("Navigation")
        
        # Check API status
        api_status = check_api_health()
        if api_status:
            st.success("API Connected")
        else:
            st.error("API Offline")
            st.warning("Please ensure the FastAPI server is running:\n```bash\npython api.py\n```")
        
        page = st.radio(
            "Select Page:",
            ["Single Prediction", "Batch Prediction", "About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        if api_status:
            try:
                response = requests.get(f"{API_URL}/features")
                if response.status_code == 200:
                    features_info = response.json()
                    st.metric("Required Features", features_info['count'])
            except:
                pass
        
        st.markdown("---")
        st.markdown("**Models:**")
        st.markdown("• Logistic Regression (Baseline)")
        st.markdown("• Random Forest (Advanced)")
    
    # Main content
    if not api_status:
        st.error("Cannot connect to API. Please start the FastAPI server first.")
        st.code("python api.py", language="bash")
        return
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "About":
        about_page()


if __name__ == "__main__":
    main()

