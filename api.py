"""
FastAPI Backend for Hospital Readmission Prediction
Serves predictions for both individual and batch requests.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from io import StringIO
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="API for predicting hospital readmissions using Logistic Regression and Random Forest models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
try:
    artifacts = joblib.load("readmission_models.joblib")
    scaler = artifacts["scaler"]
    baseline_model = artifacts["baseline"]
    rf_model = artifacts["rf"]
    feature_cols = artifacts["feature_cols"]
    print(f"Models loaded successfully with {len(feature_cols)} features")
except FileNotFoundError:
    raise RuntimeError("Model file 'readmission_models.joblib' not found. Please ensure the model is saved.")
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")


# Pydantic models for request/response
class PatientInput(BaseModel):
    """Single patient input for prediction"""
    acetohexamide: float = Field(..., description="Acetohexamide medication (encoded)")
    tolazamide: float = Field(..., description="Tolazamide medication (encoded)")
    glimepiride_pioglitazone: float = Field(..., alias="glimepiride-pioglitazone", description="Glimepiride-pioglitazone combination (encoded)")
    metformin_pioglitazone: float = Field(..., alias="metformin-pioglitazone", description="Metformin-pioglitazone combination (encoded)")
    metformin_rosiglitazone: float = Field(..., alias="metformin-rosiglitazone", description="Metformin-rosiglitazone combination (encoded)")
    weight: float = Field(..., description="Weight (encoded)")
    number_emergency: float = Field(..., description="Number of emergency visits")
    number_inpatient: float = Field(..., description="Number of inpatient visits")
    chlorpropamide: float = Field(..., description="Chlorpropamide medication (encoded)")
    miglitol: float = Field(..., description="Miglitol medication (encoded)")
    prior_visits: float = Field(..., description="Total prior visits (outpatient + emergency + inpatient)")
    emergency_to_inpatient_ratio: float = Field(..., description="Ratio of emergency to inpatient visits")
    outpatient_proportion: float = Field(..., description="Proportion of outpatient visits")
    long_stay: int = Field(..., description="Binary indicator for long hospital stay (0 or 1)")
    multiple_medicines: int = Field(..., description="Binary indicator for multiple medicines (0 or 1)")
    lab_intensity: float = Field(..., description="Lab procedures per day")
    procedures_per_day: float = Field(..., description="Procedures per day")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Single prediction response"""
    patient_id: Optional[int] = Field(None, description="Patient identifier")
    baseline_prediction: int = Field(..., description="Baseline model prediction (0=No Readmit, 1=Readmit)")
    baseline_probability: float = Field(..., description="Baseline model probability of readmission")
    rf_prediction: int = Field(..., description="Random Forest prediction (0=No Readmit, 1=Readmit)")
    rf_probability: float = Field(..., description="Random Forest probability of readmission")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    summary: dict


def prepare_features(patient_data: dict) -> pd.DataFrame:
    """Prepare features in the correct order for prediction"""
    # Convert field names (replace underscores with hyphens for the specific fields)
    feature_mapping = {
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "metformin_pioglitazone": "metformin-pioglitazone",
        "metformin_rosiglitazone": "metformin-rosiglitazone"
    }
    
    prepared_data = {}
    for key, value in patient_data.items():
        mapped_key = feature_mapping.get(key, key)
        prepared_data[mapped_key] = value
    
    # Create DataFrame with features in correct order
    df = pd.DataFrame([prepared_data])
    
    # Ensure all required features are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Return features in the same order as training
    return df[feature_cols]


def classify_risk(probability: float) -> str:
    """Classify risk level based on readmission probability"""
    if probability < 0.4:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Hospital Readmission Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "upload": "/predict/upload"
        },
        "models": {
            "baseline": "Logistic Regression",
            "rf": "Random Forest"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": True,
        "features_count": len(feature_cols)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(patient: PatientInput):
    """
    Predict readmission for a single patient
    
    Args:
        patient: Patient data with all required features
        
    Returns:
        PredictionResponse with predictions from both models
    """
    try:
        # Prepare features
        X = prepare_features(patient.dict(by_alias=True))
        
        # Scale features
        X_scaled = X.copy()
        X_scaled[feature_cols] = scaler.transform(X[feature_cols])
        
        # Get predictions from both models
        baseline_pred = int(baseline_model.predict(X_scaled)[0])
        baseline_prob = float(baseline_model.predict_proba(X_scaled)[0, 1])
        
        rf_pred = int(rf_model.predict(X_scaled)[0])
        rf_prob = float(rf_model.predict_proba(X_scaled)[0, 1])
        
        # Use Random Forest probability for risk classification (better model)
        risk_level = classify_risk(rf_prob)
        
        return PredictionResponse(
            baseline_prediction=baseline_pred,
            baseline_probability=round(baseline_prob, 4),
            rf_prediction=rf_pred,
            rf_probability=round(rf_prob, 4),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(patients: List[PatientInput]):
    """
    Predict readmission for multiple patients
    
    Args:
        patients: List of patient data
        
    Returns:
        BatchPredictionResponse with all predictions and summary statistics
    """
    try:
        predictions = []
        
        for idx, patient in enumerate(patients):
            # Prepare features
            X = prepare_features(patient.dict(by_alias=True))
            
            # Scale features
            X_scaled = X.copy()
            X_scaled[feature_cols] = scaler.transform(X[feature_cols])
            
            # Get predictions
            baseline_pred = int(baseline_model.predict(X_scaled)[0])
            baseline_prob = float(baseline_model.predict_proba(X_scaled)[0, 1])
            
            rf_pred = int(rf_model.predict(X_scaled)[0])
            rf_prob = float(rf_model.predict_proba(X_scaled)[0, 1])
            
            risk_level = classify_risk(rf_prob)
            
            predictions.append(
                PredictionResponse(
                    patient_id=idx + 1,
                    baseline_prediction=baseline_pred,
                    baseline_probability=round(baseline_prob, 4),
                    rf_prediction=rf_pred,
                    rf_probability=round(rf_prob, 4),
                    risk_level=risk_level
                )
            )
        
        # Calculate summary statistics
        rf_probs = [p.rf_probability for p in predictions]
        rf_preds = [p.rf_prediction for p in predictions]
        risk_levels = [p.risk_level for p in predictions]
        
        summary = {
            "total_patients": len(predictions),
            "predicted_readmissions": sum(rf_preds),
            "predicted_no_readmissions": len(rf_preds) - sum(rf_preds),
            "avg_readmission_probability": round(np.mean(rf_probs), 4),
            "high_risk_count": risk_levels.count("High"),
            "medium_risk_count": risk_levels.count("Medium"),
            "low_risk_count": risk_levels.count("Low")
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/predict/upload")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predict readmission from uploaded CSV file
    
    Args:
        file: CSV file with patient data (must include all required features)
        
    Returns:
        BatchPredictionResponse with all predictions and summary
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = set(feature_cols)
        provided_cols = set(df.columns)
        missing_cols = required_cols - provided_cols
        
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {list(missing_cols)}"
            )
        
        # Prepare features
        X = df[feature_cols].copy()
        
        # Scale features
        X_scaled = X.copy()
        X_scaled[feature_cols] = scaler.transform(X[feature_cols])
        
        # Get predictions
        baseline_preds = baseline_model.predict(X_scaled)
        baseline_probs = baseline_model.predict_proba(X_scaled)[:, 1]
        
        rf_preds = rf_model.predict(X_scaled)
        rf_probs = rf_model.predict_proba(X_scaled)[:, 1]
        
        # Create predictions list
        predictions = []
        for idx in range(len(df)):
            risk_level = classify_risk(rf_probs[idx])
            
            predictions.append(
                PredictionResponse(
                    patient_id=idx + 1,
                    baseline_prediction=int(baseline_preds[idx]),
                    baseline_probability=round(float(baseline_probs[idx]), 4),
                    rf_prediction=int(rf_preds[idx]),
                    rf_probability=round(float(rf_probs[idx]), 4),
                    risk_level=risk_level
                )
            )
        
        # Calculate summary statistics
        risk_levels = [p.risk_level for p in predictions]
        
        summary = {
            "total_patients": len(predictions),
            "predicted_readmissions": int(rf_preds.sum()),
            "predicted_no_readmissions": int(len(rf_preds) - rf_preds.sum()),
            "avg_readmission_probability": round(float(rf_probs.mean()), 4),
            "high_risk_count": risk_levels.count("High"),
            "medium_risk_count": risk_levels.count("Medium"),
            "low_risk_count": risk_levels.count("Low")
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


@app.get("/features")
def get_features():
    """Get list of required features for prediction"""
    return {
        "features": feature_cols,
        "count": len(feature_cols),
        "description": {
            "acetohexamide": "Sulfonylurea antihyperglycemic medication",
            "tolazamide": "Sulfonylurea antihyperglycemic medication",
            "glimepiride-pioglitazone": "Combination sulfonylurea + thiazolidinedione",
            "metformin-pioglitazone": "Combination biguanide + thiazolidinedione",
            "metformin-rosiglitazone": "Combination biguanide + thiazolidinedione",
            "weight": "Patient weight (encoded)",
            "number_emergency": "Number of emergency visits",
            "number_inpatient": "Number of inpatient visits",
            "chlorpropamide": "First-generation sulfonylurea",
            "miglitol": "Alpha-glucosidase inhibitor",
            "prior_visits": "Total prior visits (engineered)",
            "emergency_to_inpatient_ratio": "Emergency to inpatient ratio (engineered)",
            "outpatient_proportion": "Outpatient proportion (engineered)",
            "long_stay": "Binary indicator for long stay (engineered)",
            "multiple_medicines": "Binary indicator for multiple medicines (engineered)",
            "lab_intensity": "Lab procedures per day (engineered)",
            "procedures_per_day": "Procedures per day (engineered)"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

