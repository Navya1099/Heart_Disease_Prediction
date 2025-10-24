from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
from pathlib import Path
import numpy as np

app = FastAPI(title="Heart Disease Prediction Web App")

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
MODEL_PATH = Path("models/heart_disease_model.pkl")
model_bundle = joblib.load(MODEL_PATH)
pipeline = model_bundle["pipeline"]

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: float = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: float = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):
    # Input data
    X_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]])

    # Predict
    prediction = pipeline.predict(X_input)[0]
    probability = pipeline.predict_proba(X_input)[0][1]
    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result,
        "probability": f"{probability * 100:.2f}%"
    })
