import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from django.conf import settings

@dataclass
class RiskPredictionResult:
    risk_category: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_name: str = "Trained ML Classifier"
    is_fallback: bool = False
    warning: Optional[str] = None

class MLRiskPredictor:
    """
    Inference Service for VerifyX AI Machine Learning Risk Scoring.
    Loads the trained scikit-learn Pipeline and produces calibrated risk category predictions.
    """

    DEFAULT_FEATURE_NAMES = [
        'document_quality',
        'ocr_confidence',
        'face_similarity',
        'tampering_probability',
        'identity_consistency',
        'name_match',
        'dob_match',
        'id_format_valid',
        'address_match'
    ]

    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None):
        base_dir = str(settings.BASE_DIR)
        self.model_path = model_path or os.path.join(base_dir, 'ml_models', 'risk_model.pkl')
        self.metadata_path = metadata_path or os.path.join(base_dir, 'ml_models', 'model_metadata.json')
        
        self.pipeline = None
        self.metadata = {}
        self.feature_names = self.DEFAULT_FEATURE_NAMES
        self.classes = ['LOW_RISK', 'MANUAL_REVIEW', 'HIGH_RISK']
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Safely load pipeline and metadata."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('feature_names', self.DEFAULT_FEATURE_NAMES)
                    self.classes = self.metadata.get('classes', self.classes)
            except Exception:
                pass

        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception:
                self.pipeline = None

    def predict_risk(self, features: Dict[str, Any]) -> RiskPredictionResult:
        """
        Execute risk prediction for a single dictionary of screening features.
        
        Input Example:
        {
            "document_quality": 92,
            "ocr_confidence": 96,
            "face_similarity": 94,
            "tampering_probability": 8,
            "name_match": 1,
            "dob_match": 1,
            "id_format_valid": 1,
            "address_match": 1,
            "identity_consistency": 100
        }
        
        Returns RiskPredictionResult with risk_category and confidence.
        """
        if not isinstance(features, dict):
            raise ValueError("Features input must be a valid dictionary.")

        # Extract and sanitize feature values
        row_dict = {}
        missing_features = []

        for feat in self.feature_names:
            if feat in features:
                val = features[feat]
                # Cast booleans to integers
                if isinstance(val, bool):
                    val = 1 if val else 0
                try:
                    row_dict[feat] = float(val)
                except (ValueError, TypeError):
                    raise ValueError(f"Feature '{feat}' has invalid non-numeric value: {val}")
            else:
                missing_features.append(feat)

        if missing_features:
            raise ValueError(f"Missing required feature(s): {', '.join(missing_features)}")

        # Construct DataFrame to preserve feature names for ColumnTransformer
        X_df = pd.DataFrame([row_dict], columns=self.feature_names)

        # 1. Pipeline Prediction
        if self.pipeline is not None:
            try:
                pred_class = str(self.pipeline.predict(X_df)[0])
                probs = {}
                confidence = 0.85

                if hasattr(self.pipeline, 'predict_proba'):
                    prob_array = self.pipeline.predict_proba(X_df)[0]
                    pipeline_classes = getattr(self.pipeline, 'classes_', self.classes)
                    for cls_name, p_val in zip(pipeline_classes, prob_array):
                        probs[str(cls_name)] = round(float(p_val), 4)

                    confidence = round(float(np.max(prob_array)), 2)

                # Heuristic safety override for severe fraud indicators
                # (e.g. tampering > 70% or severe face/identity discrepancy)
                tamper_prob = row_dict.get('tampering_probability', 0)
                face_sim = row_dict.get('face_similarity', 100)
                doc_q = row_dict.get('document_quality', 100)

                if tamper_prob >= 75.0 or (face_sim < 30.0 and row_dict.get('name_match', 1) == 0):
                    pred_class = 'HIGH_RISK'
                    confidence = 0.95
                elif tamper_prob >= 50.0 or face_sim < 50.0 or doc_q < 45.0:
                    if pred_class == 'LOW_RISK':
                        pred_class = 'MANUAL_REVIEW'
                        confidence = 0.78

                return RiskPredictionResult(
                    risk_category=pred_class,
                    confidence=confidence,
                    probabilities=probs,
                    model_name=self.metadata.get('model_name', 'Trained ML Classifier'),
                    is_fallback=False
                )
            except Exception as e:
                # Fallback to rule engine if pipeline execution fails
                pass

        # 2. Rule-Based Fallback if model artifact is unavailable
        return self._rule_based_fallback(row_dict)

    def _rule_based_fallback(self, row_dict: Dict[str, float]) -> RiskPredictionResult:
        """Deterministic heuristic fallback when pipeline binary is not loaded."""
        tamper = row_dict.get('tampering_probability', 0)
        face_sim = row_dict.get('face_similarity', 100)
        doc_q = row_dict.get('document_quality', 100)
        ocr_conf = row_dict.get('ocr_confidence', 100)
        consistency = row_dict.get('identity_consistency', 100)

        # Risk heuristics
        if tamper >= 70 or (face_sim < 40 and row_dict.get('name_match', 1) == 0):
            category = 'HIGH_RISK'
            conf = 0.92
        elif tamper >= 40 or face_sim < 60 or doc_q < 50 or ocr_conf < 60 or consistency < 75:
            category = 'MANUAL_REVIEW'
            conf = 0.85
        else:
            category = 'LOW_RISK'
            conf = 0.90

        return RiskPredictionResult(
            risk_category=category,
            confidence=conf,
            probabilities={category: conf},
            model_name="Deterministic Rule Engine (Fallback)",
            is_fallback=True,
            warning="ML model artifact not loaded; heuristic engine used."
        )
