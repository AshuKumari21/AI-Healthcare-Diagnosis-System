
import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Healthcare ML API | Clinical Intelligence")

# Add CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define model paths
MODEL_PATHS = {
    "diabetes": "diabetes_classifier.pkl",
    "heart": "heart_classifier.pkl",
    "liver": "liver_classifier.pkl",
    "kidney": "kidney_classifier.pkl",
    "stroke": "stroke_classifier.pkl",
    "anemia": "anemia_classifier.pkl",
    "obesity": "obesity_classifier.pkl",
    "general_health": "general_health_classifier.pkl",
    "bp": "bp_classifier.pkl",
}

# models load
models = {}
for key, filename in MODEL_PATHS.items():
    path = os.path.join(BASE_DIR, "ml", "models", filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[key] = pickle.load(f)

from typing import Any

# input schema
class InputData(BaseModel):
    features: Any

@app.get("/")
def home():
    return {
        "status": "online",
        "models_loaded": list(models.keys()),
        "message": "Healthcare ML API running successfully"
    }

@app.post("/predict/{disease}")
def predict(disease: str, data: InputData):
    if disease not in models:
        return {"error": f"Model for '{disease}' not found. Available: {list(models.keys())}"}

    model = models[disease]
    
    # Identify expected features from the pipeline's scaler or model
    try:
        if hasattr(model, 'feature_names_in_'):
            expected_names = list(model.feature_names_in_)
            expected_features = len(expected_names)
        elif hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            expected_names = None
        else:
            expected_features = len(data.features)
            expected_names = None
    except:
        expected_features = len(data.features)
        expected_names = None

    # Feature Alignment Logic
    input_features = data.features
    final_features = []

    if isinstance(input_features, dict) and expected_names:
        # Create a lowercase mapping of input features for case-insensitive lookup
        input_lower = {str(k).lower(): v for k, v in input_features.items()}
        # If we have names, map them correctly
        for name in expected_names:
            final_features.append(input_lower.get(name.lower(), 0.0))
    else:
        # If we have a list, pad or truncate
        current_features = list(input_features)
        if len(current_features) < expected_features:
            current_features.extend([0.0] * (expected_features - len(current_features)))
        elif len(current_features) > expected_features:
            current_features = current_features[:expected_features]
        final_features = current_features

    features_array = np.array(final_features).reshape(1, -1)
    
    # Prediction
    try:
        prediction = int(model.predict(features_array)[0])
        
        # Risk Probability (from classifier)
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features_array)[0][1] * 100)
        
        return {
            "disease": disease,
            "prediction": prediction,
            "probability": round(probability, 2),
            "status": "High Risk" if prediction == 1 or probability > 70 else ("Medium Risk" if probability > 30 else "Low Risk"),
            "features_used": expected_features,
            "mapping_status": "named" if isinstance(input_features, dict) else "positional"
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)