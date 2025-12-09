# Deployment Guide: Hospital Readmission Prediction System

This guide provides step-by-step instructions to deploy the Hospital Readmission Prediction system using FastAPI backend and Streamlit frontend.

---

## 📋 Prerequisites

Before starting, ensure you have:
- Python 3.10+ installed
- Conda environment activated (`CAP3764_PROJECT`)
- Model file `readmission_models.joblib` in the project root directory

---

## 🚀 Quick Start

### Option 1: Using Conda Environment (Recommended)

```bash
# Activate the existing Conda environment
conda activate CAP3764_PROJECT

# Install additional deployment dependencies
pip install fastapi uvicorn[standard] python-multipart plotly pydantic

# Navigate to project directory
cd "/Users/dsuniaga/Desktop/CAP 3764/class_project"
```

### Option 2: Using pip (Alternative)

```bash
# Install all deployment requirements
pip install -r requirements_deployment.txt
```

---

## 🎯 Running the Application

### Step 1: Start the FastAPI Backend

Open a terminal and run:

```bash
# Navigate to project directory
cd "/Users/dsuniaga/Desktop/CAP 3764/class_project"

# Start the FastAPI server
python api.py
```

**Expected Output:**
```
✓ Models loaded successfully with 17 features
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The API will be available at: **http://localhost:8000**

**API Documentation:**
- Interactive Docs (Swagger UI): http://localhost:8000/docs
- Alternative Docs (ReDoc): http://localhost:8000/redoc

---

### Step 2: Start the Streamlit Dashboard

Open a **NEW terminal** (keep the API running) and run:

```bash
# Navigate to project directory
cd "/Users/dsuniaga/Desktop/CAP 3764/class_project"

# Activate environment (if not already activated)
conda activate CAP3764_PROJECT

# Start the Streamlit app
streamlit run streamlit_app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The dashboard will automatically open in your browser at: **http://localhost:8501**

---

## 📊 Using the Application

### Single Patient Prediction

1. Navigate to **"🏥 Single Prediction"** page
2. Fill in patient information:
   - **Medication Information**: Encoded medication values
   - **Hospital Utilization**: Emergency visits, inpatient visits, prior visits
   - **Clinical Indicators**: Long stay, multiple medicines, lab intensity, procedures per day
3. Click **"🔍 Predict Readmission Risk"**
4. View results:
   - Risk Level (Low/Medium/High)
   - Random Forest and Baseline predictions
   - Probability scores
   - Interactive risk gauge

### Batch Prediction (CSV Upload)

1. Navigate to **"📊 Batch Prediction"** → **"📤 Upload CSV"** tab
2. Download the CSV template by clicking **"📥 Download CSV Template"**
3. Fill in patient data in the CSV file
4. Upload the completed CSV file
5. Click **"🚀 Run Batch Prediction"**
6. View results:
   - Summary statistics
   - Risk distribution charts
   - Detailed predictions table
   - Download results as CSV

### Batch Prediction (JSON)

1. Navigate to **"📊 Batch Prediction"** → **"✍️ Manual Entry"** tab
2. Enter patient data in JSON format (sample provided)
3. Click **"🚀 Run Batch Prediction (JSON)"**
4. View comprehensive results and analytics

---

## 🔌 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Get Features List
```bash
curl http://localhost:8000/features
```

### Upload CSV for Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@patients.csv"
```

---

## 🧪 Testing the System

### Test 1: API Health Check

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "features_count": 17
}
```

### Test 2: Single Prediction

Use the example payload above or test through the Streamlit interface.

### Test 3: Batch Prediction

Create a test CSV file:

```csv
acetohexamide,tolazamide,glimepiride-pioglitazone,metformin-pioglitazone,metformin-rosiglitazone,weight,number_emergency,number_inpatient,chlorpropamide,miglitol,prior_visits,emergency_to_inpatient_ratio,outpatient_proportion,long_stay,multiple_medicines,lab_intensity,procedures_per_day
0.0,0.0,0.0,0.0,0.0,1.0,2.0,1.0,0.0,0.0,4.0,1.0,0.25,1,1,10.5,0.5
0.0,0.0,0.0,0.0,0.0,2.0,5.0,3.0,0.0,0.0,10.0,1.67,0.2,1,1,15.0,0.8
```

Upload via Streamlit or use cURL:

```bash
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@test_patients.csv"
```

---

## 🛠️ Troubleshooting

### Issue: "Cannot connect to API"

**Solution:**
1. Ensure FastAPI server is running: `python api.py`
2. Check if port 8000 is available: `lsof -i :8000`
3. Verify API health: `curl http://localhost:8000/health`

### Issue: "Model file not found"

**Solution:**
1. Ensure `readmission_models.joblib` exists in project root
2. Verify file was created by running all cells in `notebooks/main.ipynb`
3. Check current directory matches project root

### Issue: "Missing required columns"

**Solution:**
1. Download the CSV template from Streamlit dashboard
2. Ensure all 17 required features are present
3. Check feature names match exactly (including hyphens)

### Issue: "Port already in use"

**Solution:**

For FastAPI (port 8000):
```bash
# Find and kill process
lsof -ti :8000 | xargs kill -9

# Or run on different port
uvicorn api:app --host 0.0.0.0 --port 8001
```

For Streamlit (port 8501):
```bash
# Find and kill process
lsof -ti :8501 | xargs kill -9

# Or run on different port
streamlit run streamlit_app.py --server.port 8502
```

---

## 📦 File Structure

```
class_project/
├── api.py                           # FastAPI backend
├── streamlit_app.py                 # Streamlit dashboard
├── readmission_models.joblib        # Trained models (scaler, baseline, RF)
├── requirements_deployment.txt      # Deployment dependencies
├── DEPLOYMENT.md                    # This file
└── notebooks/
    └── main.ipynb                   # Model training notebook
```

---

## 🔒 Security Considerations

For production deployment:

1. **API Authentication**: Add API key authentication
2. **HTTPS**: Use SSL certificates for encrypted communication
3. **Rate Limiting**: Implement request throttling
4. **Input Validation**: Enhanced validation for all inputs
5. **Logging**: Add comprehensive logging for audit trails
6. **CORS**: Restrict allowed origins in production

Example with API key:

```python
# In api.py
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
```

---

## 🌐 Production Deployment Options

### Option 1: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_deployment.txt .
RUN pip install --no-cache-dir -r requirements_deployment.txt

COPY api.py streamlit_app.py readmission_models.joblib ./

EXPOSE 8000 8501

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t readmission-predictor .
docker run -p 8000:8000 -p 8501:8501 readmission-predictor
```

### Option 2: Cloud Deployment

**FastAPI on Heroku:**
1. Create `Procfile`: `web: uvicorn api:app --host 0.0.0.0 --port $PORT`
2. Deploy: `git push heroku main`

**Streamlit Cloud:**
1. Push code to GitHub
2. Deploy via streamlit.io
3. Configure secrets for API URL

### Option 3: AWS/Azure/GCP

Deploy FastAPI as containerized service and Streamlit as web app.

---

## 📈 Performance Metrics

**API Response Times:**
- Single prediction: ~50-100ms
- Batch prediction (100 patients): ~500ms-1s
- CSV upload (1000 patients): ~2-5s

**Resource Usage:**
- Memory: ~200-300 MB (API) + ~150-200 MB (Streamlit)
- CPU: Minimal (<5% idle, 20-40% during predictions)

---

## 🤝 Support

For issues or questions:
1. Check troubleshooting section above
2. Review API documentation at http://localhost:8000/docs
3. Contact repository maintainer

---

## 📝 Version History

- **v1.0.0** (December 2025) - Initial deployment with FastAPI + Streamlit
  - Single and batch predictions
  - CSV upload support
  - Interactive dashboard
  - Risk classification (Low/Medium/High)

---

## 📄 License

This deployment is part of the CAP 3764 course project and is for educational purposes only.

