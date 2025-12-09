# ⚡ Quick Start Guide

## 🚀 Launch in 3 Steps

### 1️⃣ Save Your Model (if not already done)

Open `notebooks/main.ipynb` and add this cell at the end:

```python
import joblib

artifact = {
    "scaler": scaler,
    "baseline": baseline,
    "rf": best_rf,
    "feature_cols": X_train.columns.tolist()
}

joblib.dump(artifact, "../readmission_models.joblib")
print("✓ Model saved successfully!")
```

### 2️⃣ Install Dependencies

```bash
conda activate CAP3764_PROJECT
pip install fastapi uvicorn[standard] python-multipart plotly pydantic
```

### 3️⃣ Start the System

```bash
cd "/Users/dsuniaga/Desktop/CAP 3764/class_project"
./start_services.sh
```

**That's it!** 🎉

---

## 🌐 Access Points

- 🎨 **Dashboard**: http://localhost:8501
- 📡 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

---

## 🧪 Test It Works

```bash
python test_api.py
```

Expected output: `🎉 All tests passed!`

---

## 🛑 Stop the System

Press `Ctrl+C` in the terminal where you ran `./start_services.sh`

---

## 📖 Need More Help?

- **Detailed Guide**: See `DEPLOYMENT.md`
- **Full Summary**: See `DEPLOYMENT_SUMMARY.md`
- **Troubleshooting**: See `DEPLOYMENT.md` → Troubleshooting section

---

## 🎯 Common Commands

### Start Services Manually

```bash
# Terminal 1 - API
python api.py

# Terminal 2 - Dashboard
streamlit run streamlit_app.py
```

### Check API Health

```bash
curl http://localhost:8000/health
```

### Make a Prediction

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

---

## 🎬 Demo Flow

1. **Start System**: `./start_services.sh`
2. **Open Dashboard**: http://localhost:8501
3. **Single Prediction**: Fill form → Predict → View risk gauge
4. **Batch Prediction**: Download template → Fill CSV → Upload → View charts
5. **API Demo**: Open http://localhost:8000/docs → Try endpoints

---

## ⚠️ Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot connect to API" | Run `python api.py` first |
| "Model file not found" | Save model from `main.ipynb` |
| "Port already in use" | `lsof -ti :8000 \| xargs kill -9` |
| "Missing package" | `pip install -r requirements_deployment.txt` |

---

**Ready to deploy! 🚀**

