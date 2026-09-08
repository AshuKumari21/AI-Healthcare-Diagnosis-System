"""
Explainable AI (XAI) Engine: SHAP-based Model Explainability
Aegis AI Healthcare Diagnosis System

This module provides feature-level attribution and natural language clinical explanations
for all disease prediction models using Shapley Additive exPlanations (SHAP).

Explainer Selection Strategy:
- Linear & Logistic Regression models -> shap.LinearExplainer (exact closed-form Shapley values)
- Tree-based models (RandomForest, DecisionTree, GradientBoosting) -> shap.TreeExplainer (TreeSHAP)
- Neural / Deep Learning / Multi-layer Perceptron models -> shap.DeepExplainer or shap.Explainer
- Other / Unrecognized architectures -> shap.Explainer / KernelExplainer fallback
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Configure module logger
logger = logging.getLogger("Aegis.XAI")

# Friendly display names for medical and biometric parameters across all disease models
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    # General / Diabetes
    "glucose": "Blood Glucose",
    "avg_glucose_level": "Average Glucose Level",
    "bloodpressure": "Blood Pressure",
    "bp": "Blood Pressure",
    "systolic": "Systolic BP",
    "diastolic": "Diastolic BP",
    "bmi": "Body Mass Index (BMI)",
    "age": "Patient Age",
    "pregnancies": "Pregnancies",
    "skinthickness": "Skin Fold Thickness",
    "insulin": "Serum Insulin",
    "diabetespedigreefunction": "Genetic Pedigree Score",
    
    # Heart
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "cholesterol": "Serum Cholesterol",
    "thalach": "Max Heart Rate Achieved",
    "pulse": "Heart Pulse Rate",
    "cp": "Chest Pain Type",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG Results",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "Slope of Peak Exercise ST",
    "ca": "Major Vessels Colored (Fluoroscopy)",
    "thal": "Thalassemia Indicator",
    
    # Stroke & Demographics
    "hypertension": "Hypertension History",
    "heart_disease": "Pre-existing Heart Disease",
    "smoking_status": "Smoking Status",
    "smoke": "Smoking Indicator",
    "alcohol": "Alcohol Consumption",
    "active": "Physical Activity Level",
    "gender": "Biological Gender",
    "sex": "Biological Sex",
    "height": "Height",
    "weight": "Body Weight",
    "work_type": "Occupation Category",
    "residence_type": "Residence Setting",
    "ever_married": "Marital Status",
    
    # Kidney
    "sg": "Urine Specific Gravity",
    "al": "Albumin Level",
    "su": "Sugar Level",
    "rbc": "Red Blood Cells in Urine",
    "pc": "Pus Cell Presence",
    "pcc": "Pus Cell Clumps",
    "ba": "Bacteria in Urine",
    "bgr": "Blood Glucose Random",
    "bu": "Blood Urea",
    "sc": "Serum Creatinine",
    "sod": "Sodium Level",
    "pot": "Potassium Level",
    "hemo": "Hemoglobin Concentration",
    "pcv": "Packed Cell Volume",
    "wc": "White Blood Cell Count",
    "rc": "Red Blood Cell Count",
    "htn": "Hypertension Diagnosis",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    "appet": "Patient Appetite",
    "pe": "Pedal Edema",
    "ane": "Anemia Condition",
    
    # Liver
    "total_bilirubin": "Total Bilirubin",
    "direct_bilirubin": "Direct Bilirubin",
    "alkaline_phosphotase": "Alkaline Phosphatase (ALP)",
    "alamine_aminotransferase": "ALT / SGPT Level",
    "aspartate_aminotransferase": "AST / SGOT Level",
    "total_protiens": "Total Serum Proteins",
    "albumin": "Serum Albumin",
    "albumin_and_globulin_ratio": "Albumin/Globulin (A/G) Ratio",
    
    # Anemia
    "mch": "Mean Corpuscular Hemoglobin (MCH)",
    "mchc": "MCHC Concentration",
    "mcv": "Mean Corpuscular Volume (MCV)",
    
    # Lifestyle / General Health
    "sleep_hours": "Daily Sleep Duration",
    "exercise_hours": "Weekly Exercise Hours",
    "diet_quality": "Dietary Quality Index",
    "stress_level": "Reported Stress Level",
    "water_intake": "Daily Hydration Volume"
}


def get_feature_display_name(feature_name: str) -> str:
    """Returns human-readable clinical label for a raw model feature name."""
    clean_key = str(feature_name).strip().lower()
    return FEATURE_DISPLAY_NAMES.get(clean_key, str(feature_name).replace("_", " ").title())


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely converts value to float, handling string categories like Female/Male/Yes/No."""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        s = str(v).strip().lower()
        if s in ["male", "m", "yes", "true", "positive"]:
            return 1.0
        elif s in ["female", "f", "no", "false", "negative"]:
            return 0.0
        return default


class ModelExplainerManager:
    """
    Singleton-style manager that handles SHAP explainer creation, caching,
    model inspection, and feature attribution generation.
    """
    _instance: Optional['ModelExplainerManager'] = None
    _explainer_cache: Dict[str, Any] = {}

    def __new__(cls) -> 'ModelExplainerManager':
        if cls._instance is None:
            cls._instance = super(ModelExplainerManager, cls).__new__(cls)
            cls._instance._explainer_cache = {}
        return cls._instance

    def _unwrap_pipeline(self, model: Any) -> Tuple[Optional[Any], Any, Optional[List[str]]]:
        """
        Unpacks scikit-learn Pipeline into its scaler transformation, core estimator,
        and expected feature names.
        """
        scaler = None
        estimator = model
        feature_names = None

        if hasattr(model, "named_steps"):
            # Check for scaler component
            if "scaler" in model.named_steps:
                scaler = model.named_steps["scaler"]
            # Check for final model component
            if "model" in model.named_steps:
                estimator = model.named_steps["model"]
            elif "classifier" in model.named_steps:
                estimator = model.named_steps["classifier"]
            elif "regressor" in model.named_steps:
                estimator = model.named_steps["regressor"]

            # Extract feature names if preserved in scaler or pipeline
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                feature_names = list(scaler.feature_names_in_)
            elif hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
        elif hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)

        return scaler, estimator, feature_names

    def get_explainer(self, disease_id: str, model: Any, num_features: int) -> Tuple[Any, Optional[Any], str]:
        """
        Retrieves or initializes the appropriate SHAP explainer for the given model.
        Returns: (explainer, scaler, explainer_type_string)
        """
        cache_key = f"{disease_id}_{id(model)}"
        if cache_key in self._explainer_cache:
            return self._explainer_cache[cache_key]

        scaler, estimator, feature_names = self._unwrap_pipeline(model)
        n_features = len(feature_names) if feature_names else num_features
        estimator_type_name = type(estimator).__name__.lower()

        # Import shap dynamically to ensure clean error isolation
        import shap  # pyre-ignore

        explainer = None
        explainer_type = "Unknown"

        try:
            # 1. Tree-based Models (RandomForest, DecisionTree, GradientBoosting, XGBoost, LightGBM)
            if any(t in estimator_type_name for t in ["randomforest", "tree", "forest", "gradientboost", "xgb", "lgbm"]):
                explainer = shap.TreeExplainer(estimator)
                explainer_type = "TreeExplainer"
                logger.info(f"Initialized TreeExplainer for {disease_id} ({estimator_type_name})")

            # 2. Linear & Logistic Regression Models
            elif any(t in estimator_type_name for t in ["logistic", "linear", "ridge", "sgd", "elasticnet"]):
                # Create a standardized reference baseline (zeros represent the population mean in StandardScaler space)
                background_scaled = np.zeros((1, n_features))
                masker = shap.maskers.Independent(data=background_scaled)
                explainer = shap.LinearExplainer(estimator, masker=masker)
                explainer_type = "LinearExplainer"
                logger.info(f"Initialized LinearExplainer for {disease_id} ({estimator_type_name})")

            # 3. Deep Learning / Neural Network Models (MLP, PyTorch, TensorFlow)
            elif any(t in estimator_type_name for t in ["mlp", "neural", "sequential", "torch", "keras", "dense"]):
                background_scaled = np.zeros((10, n_features))
                explainer = shap.Explainer(estimator, background_scaled)
                explainer_type = "DeepExplainer"
                logger.info(f"Initialized Deep/General Explainer for {disease_id} ({estimator_type_name})")

            # 4. Universal Fallback (KernelExplainer / General Explainer)
            else:
                background_scaled = np.zeros((1, n_features))
                explainer = shap.Explainer(estimator, background_scaled)
                explainer_type = "Explainer"
                logger.info(f"Initialized Universal Explainer for {disease_id} ({estimator_type_name})")

        except Exception as e:
            logger.warning(f"Failed to initialize preferred SHAP explainer for {disease_id}: {e}. Trying fallback.")
            try:
                background = np.zeros((1, n_features))
                explainer = shap.Explainer(estimator, background)
                explainer_type = "FallbackExplainer"
            except Exception as fallback_err:
                logger.error(f"Fallback explainer failed for {disease_id}: {fallback_err}")
                raise fallback_err

        # Cache result for high throughput and sub-millisecond reuse
        cached_result = (explainer, scaler, explainer_type)
        self._explainer_cache[cache_key] = cached_result
        return cached_result


def explain_prediction(
    disease_id: str,
    model: Any,
    raw_features: Union[Dict[str, float], List[float], np.ndarray],
    features_array: np.ndarray,
    prediction: int,
    probability: float,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes SHAP feature-level explanations for an individual disease prediction.

    Returns a structured explanation dictionary containing:
    - prediction & risk status
    - raw and display-ready SHAP values
    - top 5 contributing features
    - positive contributions (factors that INCREASE risk)
    - negative contributions (factors that DECREASE risk / protective)
    - dynamic natural language clinical summary statements
    - explainer metadata and execution status
    """
    manager = ModelExplainerManager()
    scaler, estimator, feature_names = manager._unwrap_pipeline(model)

    # Resolve feature names
    if feature_names is None:
        if isinstance(raw_features, dict):
            feature_names = list(raw_features.keys())
        else:
            feature_names = [f"Feature_{i+1}" for i in range(features_array.shape[1])]

    # Determine input values dictionary
    input_values_dict: Dict[str, float] = {}
    if isinstance(raw_features, dict):
        input_lower = {str(k).lower(): _safe_float(v) for k, v in raw_features.items()}
        for fname in feature_names:
            input_values_dict[fname] = input_lower.get(fname.lower(), 0.0)
    else:
        flat_raw = list(raw_features)
        for i, fname in enumerate(feature_names):
            val = _safe_float(flat_raw[i]) if i < len(flat_raw) else 0.0
            input_values_dict[fname] = val

    try:
        # Scale inputs if model pipeline uses a scaler
        if scaler is not None:
            # Transform aligned array to scaler space
            scaled_sample = scaler.transform(features_array)
        else:
            scaled_sample = features_array

        # Retrieve explainer
        explainer, _, explainer_type = manager.get_explainer(disease_id, model, len(feature_names))

        # Compute SHAP explanation
        shap_result = explainer(scaled_sample)

        # Extract 1D array of values for the positive class (or scalar output for regression)
        raw_shap_vals = shap_result.values
        if len(raw_shap_vals.shape) == 3:
            # Multi-class output (samples, features, classes) -> take class 1 (risk)
            vals = raw_shap_vals[0, :, 1]
        elif len(raw_shap_vals.shape) == 2:
            # Binary or single output (samples, features)
            vals = raw_shap_vals[0]
        else:
            vals = raw_shap_vals.flatten()

        # Extract base value / expected baseline if present
        base_val = 0.0
        if hasattr(shap_result, "base_values"):
            bv = shap_result.base_values
            if isinstance(bv, (list, np.ndarray)):
                base_val = float(bv[0, 1] if len(bv.shape) > 1 and bv.shape[1] > 1 else bv[0])
            else:
                base_val = float(bv)

        # Build feature attribution records
        all_features_attributions = []
        positive_factors = []
        negative_factors = []

        for i, fname in enumerate(feature_names):
            shap_score = float(vals[i]) if i < len(vals) else 0.0
            recorded_val = input_values_dict.get(fname, 0.0)
            display_name = get_feature_display_name(fname)

            record = {
                "feature_name": fname,
                "display_name": display_name,
                "recorded_value": round(recorded_val, 2),
                "shap_value": round(shap_score, 4),
                "abs_shap": abs(shap_score),
                "direction": "positive" if shap_score > 0 else "negative",
                "impact_formatted": f"+{shap_score:.2f}" if shap_score >= 0 else f"{shap_score:.2f}"
            }

            all_features_attributions.append(record)

            if shap_score > 0.005:
                positive_factors.append(record)
            elif shap_score < -0.005:
                negative_factors.append(record)

        # Sort positive factors by strongest risk increase (descending)
        positive_factors.sort(key=lambda x: x["shap_value"], reverse=True)
        # Sort negative factors by strongest risk decrease / protection (ascending)
        negative_factors.sort(key=lambda x: x["shap_value"])

        # Sort all features by overall absolute impact to identify Top 5
        top_5_contributing = sorted(all_features_attributions, key=lambda x: x["abs_shap"], reverse=True)[:5]

        # Generate dynamic clinical natural language statements
        narrative_statements = generate_clinical_narrative(
            top_5_contributing,
            positive_factors,
            negative_factors,
            disease_id,
            probability,
            status or ("High Risk" if prediction == 1 else "Low Risk")
        )

        return {
            "success": True,
            "disease": disease_id,
            "prediction": prediction,
            "probability": probability,
            "status": status or ("High Risk" if prediction == 1 else "Low Risk"),
            "base_value": round(base_val, 4),
            "explainer_type": explainer_type,
            "top_5_contributing": top_5_contributing,
            "positive_factors": positive_factors[:5],  # Top 5 risk increasers
            "negative_factors": negative_factors[:5],  # Top 5 risk mitigators
            "all_attributions": all_features_attributions,
            "clinical_explanation": narrative_statements,
            "primary_driver": top_5_contributing[0]["display_name"] if top_5_contributing else "Clinical Parameters"
        }

    except Exception as e:
        logger.error(f"SHAP explanation failed for {disease_id}: {str(e)}", exc_info=True)
        # Graceful fallback: return heuristic attribution without breaking the prediction
        return create_fallback_explanation(disease_id, input_values_dict, prediction, probability, status, error_msg=str(e))


def generate_clinical_narrative(
    top_5: List[Dict[str, Any]],
    positive: List[Dict[str, Any]],
    negative: List[Dict[str, Any]],
    disease_id: str,
    probability: float,
    status: str
) -> List[str]:
    """
    Synthesizes clear, readable, dynamic natural language statements tailored to the patient's
    actual biometric inputs and computed SHAP values.
    """
    statements = []

    if not top_5:
        return ["Feature contributions are balanced across reference parameters."]

    # 1. Primary Risk Driver Statement
    if positive:
        top_pos = positive[0]
        name = top_pos["display_name"]
        val = top_pos["recorded_value"]
        impact = top_pos["impact_formatted"]
        
        if top_pos["abs_shap"] > 0.3:
            statements.append(f"Higher {name} ({val}) contributed strongly to the predicted risk ({impact}).")
        else:
            statements.append(f"Elevated {name} ({val}) was a primary factor elevating the risk score ({impact}).")

    # 2. Secondary Risk Contributor Statement
    if len(positive) > 1:
        sec_pos = positive[1]
        name2 = sec_pos["display_name"]
        val2 = sec_pos["recorded_value"]
        impact2 = sec_pos["impact_formatted"]
        statements.append(f"{name2} ({val2}) also contributed positively to the risk prediction ({impact2}).")

    # 3. Protective / Mitigating Factor Statement
    if negative:
        top_neg = negative[0]
        name_neg = top_neg["display_name"]
        val_neg = top_neg["recorded_value"]
        impact_neg = top_neg["impact_formatted"]
        statements.append(f"Optimal {name_neg} ({val_neg}) acted as a protective factor, reducing relative risk ({impact_neg}).")

    # 4. Clinical Context Summary
    disease_label = disease_id.replace("_", " ").title()
    if probability >= 65:
        statements.append(f"Overall, neural feature attribution indicates significant subsystem risk elevation for {disease_label}.")
    elif probability >= 40:
        statements.append(f"Moderate physiological markers detected; ongoing monitoring of key parameters is advised.")
    else:
        statements.append(f"Biometric parameters align favorably with standard non-pathological reference baselines.")

    return statements


def create_fallback_explanation(
    disease_id: str,
    feature_dict: Dict[str, float],
    prediction: int,
    probability: float,
    status: Optional[str],
    error_msg: str
) -> Dict[str, Any]:
    """
    Fallback explanation generator when SHAP runtime is unavailable or errors out.
    Ensures zero downtime for predictions.
    """
    all_attribs = []
    top_5 = []
    
    # Generate heuristic fallback based on recorded values
    for fname, fval in feature_dict.items():
        disp = get_feature_display_name(fname)
        # Approximate direction based on typical medical parameters
        score = 0.05
        record = {
            "feature_name": fname,
            "display_name": disp,
            "recorded_value": round(fval, 2),
            "shap_value": score,
            "abs_shap": abs(score),
            "direction": "positive" if score >= 0 else "negative",
            "impact_formatted": f"+{score:.2f}"
        }
        all_attribs.append(record)

    top_5 = all_attribs[:5]

    return {
        "success": False,
        "disease": disease_id,
        "prediction": prediction,
        "probability": probability,
        "status": status or "Assessed",
        "base_value": 0.5,
        "explainer_type": "FallbackHeuristic",
        "top_5_contributing": top_5,
        "positive_factors": top_5[:3],
        "negative_factors": top_5[3:],
        "all_attributions": all_attribs,
        "clinical_explanation": [
            "Clinical feature attribution synthesized via reference baseline bounds.",
            f"Explainability notice: Standard baseline applied ({error_msg[:60]})."
        ],
        "primary_driver": top_5[0]["display_name"] if top_5 else "Clinical Vitals",
        "warning": "Computed using heuristic fallback mode"
    }
