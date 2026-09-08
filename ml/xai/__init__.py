"""
Explainable AI (XAI) Module for Aegis Healthcare Diagnosis System.
Provides SHAP-based feature attribution, model explainers, and visualization.
"""

from ml.xai.explainer import (
    explain_prediction,
    ModelExplainerManager,
    FEATURE_DISPLAY_NAMES
)
from ml.xai.visualizer import create_shap_figure

__all__ = [
    "explain_prediction",
    "ModelExplainerManager",
    "FEATURE_DISPLAY_NAMES",
    "create_shap_figure"
]
