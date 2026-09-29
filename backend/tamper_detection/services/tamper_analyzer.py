import os
import abc
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from django.conf import settings

from documents.models import Document
from tamper_detection.models import TamperAnalysis
from .ela_analyzer import ELAAnalyzer, ELAResult
from .artifact_analyzer import ArtifactAnalyzer, ArtifactResult
from .region_detector import RegionAnomalyDetector, RegionDetectionResult

class BaseTamperModel(abc.ABC):
    """
    Abstract Base Class for Document Tampering Detection Engines.
    Enables future pluggable CNN / EfficientNet / Transformer deep models to replace
    the forensic heuristic analyzer with zero architecture refactoring.
    """

    @abc.abstractmethod
    def predict(self, image_input) -> Dict[str, Any]:
        """
        Execute tampering detection.
        Returns:
            {
                "tampering_probability": float (0.00 to 1.00),
                "risk_level": str ("LOW", "MODERATE", "HIGH"),
                "signals": List[str],
                "suspicious_regions": List[Dict[str, Any]],
                "ela_image": Optional[np.ndarray],
                "annotated_image": Optional[np.ndarray]
            }
        """
        pass

class HeuristicTamperModel(BaseTamperModel):
    """
    Multi-Signal Forensic Document Tampering Analysis Model.
    Combines Error Level Analysis (ELA), compression artifact examination,
    boundary gradient tracking, copy-move detection, and metadata analysis.
    """

    def __init__(self):
        self.ela_analyzer = ELAAnalyzer()
        self.artifact_analyzer = ArtifactAnalyzer()
        self.region_detector = RegionAnomalyDetector()

    def predict(self, image_input) -> Dict[str, Any]:
        img_bgr = self._load_image(image_input)
        if img_bgr is None:
            return {
                "tampering_probability": 0.0,
                "risk_level": TamperAnalysis.RISK_LOW,
                "signals": ["Unable to load image file."],
                "suspicious_regions": [],
                "ela_image": None,
                "annotated_image": None,
            }

        # 1. Execute Sub-Analyzers
        ela_res: ELAResult = self.ela_analyzer.analyze(img_bgr)
        art_res: ArtifactResult = self.artifact_analyzer.analyze(image_input if isinstance(image_input, str) else img_bgr)
        reg_res: RegionDetectionResult = self.region_detector.analyze(img_bgr)

        # 2. Merge Suspicious Regions
        suspicious_regions = []
        suspicious_regions.extend(ela_res.suspicious_regions)
        suspicious_regions.extend(art_res.suspicious_regions)
        suspicious_regions.extend(reg_res.suspicious_regions)

        # 3. Aggregate Explainable Signals
        signals = []
        for s in ela_res.signals + reg_res.signals + art_res.signals:
            if s and s not in signals:
                signals.append(s)

        # 4. Multi-Signal Calibration of Tampering Probability (0.00 to 1.00)
        w_ela = 0.35
        w_reg = 0.35
        w_art = 0.30

        raw_prob = (ela_res.ela_score * w_ela) + (reg_res.region_score * w_reg) + (art_res.artifact_score * w_art)

        # Boost probability only if strong independent forensic evidence exists
        software_flagged = bool(art_res.metadata_findings.get('software_flagged'))
        high_conf_ela = [r for r in ela_res.suspicious_regions if r.get('confidence', 0) >= 0.85]
        high_conf_reg = [r for r in reg_res.suspicious_regions if r.get('confidence', 0) >= 0.85]

        if software_flagged and (len(high_conf_ela) > 0 or len(high_conf_reg) > 0):
            raw_prob = max(raw_prob, 0.78)
        elif software_flagged:
            raw_prob = max(raw_prob, 0.65)
        elif len(high_conf_ela) >= 2 and len(high_conf_reg) >= 2:
            raw_prob = max(raw_prob, 0.70)
        elif len(high_conf_ela) >= 1 and len(high_conf_reg) >= 1:
            raw_prob = max(raw_prob, 0.40)

        # Ensure probability is bounded [0.05, 0.98]
        tampering_prob = round(max(0.05, min(0.98, raw_prob)), 2)

        # Determine Risk Level Category
        if tampering_prob >= 0.60:
            risk_level = TamperAnalysis.RISK_HIGH
        elif tampering_prob >= 0.35:
            risk_level = TamperAnalysis.RISK_MODERATE
        else:
            risk_level = TamperAnalysis.RISK_LOW

        # Generate Visual Annotation Heatmap
        annotated_img = self._generate_annotated_image(img_bgr, suspicious_regions, tampering_prob)

        return {
            "tampering_probability": tampering_prob,
            "risk_level": risk_level,
            "signals": signals,
            "suspicious_regions": suspicious_regions,
            "ela_image": ela_res.ela_image_array,
            "annotated_image": annotated_img,
            "metadata": art_res.metadata_findings
        }

    def _generate_annotated_image(self, base_img: np.ndarray, regions: List[Dict[str, Any]], prob: float) -> np.ndarray:
        """Create visual forensic overlay with bounding boxes and suspicious region annotations."""
        annotated = base_img.copy()
        
        # Color based on risk level
        color = (0, 0, 255) if prob >= 0.65 else ((0, 165, 255) if prob >= 0.30 else (0, 255, 0))

        for reg in regions:
            x, y, w, h = reg["x"], reg["y"], reg["width"], reg["height"]
            reg_type = reg.get("type", "anomaly").replace("_", " ").title()
            conf = reg.get("confidence", 0.8)

            # Draw glowing bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Semi-transparent overlay fill
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)

            # Label banner
            label = f"{reg_type} ({int(conf * 100)}%)"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, max(0, y - 20)), (x + text_w + 10, y), (20, 20, 20), -1)
            cv2.putText(annotated, label, (x + 5, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return annotated

    def _load_image(self, image_input) -> Optional[np.ndarray]:
        if hasattr(image_input, 'bgr'):
            return image_input.bgr
        if isinstance(image_input, np.ndarray):
            return image_input
        if not isinstance(image_input, (str, bytes)) and not hasattr(image_input, 'read') and not hasattr(image_input, 'path'):
            return None
        try:
            from documents.services.image_loader import DocumentImageLoader
            ctx = DocumentImageLoader.load(image_input)
            return ctx.bgr
        except Exception:
            return None

class TamperAnalyzerService:
    """
    High-level orchestration service for document tampering screening.
    Performs forensic inspection, persists database records, and routes suspicious documents.
    """

    def __init__(self, model: Optional[BaseTamperModel] = None):
        self.model = model or HeuristicTamperModel()

    def process_document(self, document: Document, doc_ctx: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute document tampering screening for an uploaded Document instance.
        """
        img_input = None
        if doc_ctx is not None and hasattr(doc_ctx, 'fast_bgr') and doc_ctx.fast_bgr is not None:
            img_input = doc_ctx.fast_bgr
        elif document.original_file and hasattr(document.original_file, 'path') and os.path.exists(document.original_file.path):
            img_input = document.original_file.path

        pred = self.model.predict(img_input)

        tampering_prob = float(pred["tampering_probability"])
        risk_level = str(pred["risk_level"])
        signals = list(pred["signals"])
        suspicious_regions = list(pred["suspicious_regions"])

        # Save annotated image and ELA visualization
        annotated_path = None
        ela_path = None

        if pred.get("annotated_image") is not None:
            annotated_path = self._save_visualization(pred["annotated_image"], document.verification_id, "tamper_map")
        
        if pred.get("ela_image") is not None:
            ela_path = self._save_visualization(pred["ela_image"], document.verification_id, "ela")

        # Route document processing status if high tampering risk detected
        if risk_level == TamperAnalysis.RISK_HIGH or tampering_prob >= 0.60:
            document.processing_status = Document.STATUS_MANUAL_REVIEW
            document.save(update_fields=['processing_status'])

        # Persist or update record in database
        TamperAnalysis.objects.update_or_create(
            document=document,
            defaults={
                "tampering_probability": tampering_prob,
                "risk_level": risk_level,
                "signals": signals,
                "suspicious_regions": suspicious_regions,
                "annotated_image": annotated_path,
                "ela_image": ela_path,
                "metadata_findings": pred.get("metadata", {})
            }
        )

        # Calculate Region-Level Scores
        img_bgr = self.model._load_image(img_input)
        img_shape = img_bgr.shape if img_bgr is not None else (600, 900, 3)
        region_scores = self.calculate_region_scores(suspicious_regions, img_shape)

        return {
            "tampering_probability": tampering_prob,
            "risk_level": risk_level,
            "signals": signals,
            "suspicious_regions": suspicious_regions,
            "region_scores": region_scores
        }

    @classmethod
    def calculate_region_scores(cls, suspicious_regions: List[Dict[str, Any]], img_shape: Tuple[int, int, int]) -> Dict[str, int]:
        """
        Phase 8: Map detected anomalies to specific document anatomical regions:
        - photograph (left half / upper-left)
        - name (top center-left)
        - dob (middle center-left)
        - document_number (bottom center)
        - background (remaining canvas)
        """
        h, w = img_shape[:2]
        scores = {
            "photograph": 10,
            "name": 8,
            "dob": 8,
            "document_number": 10,
            "background": 12
        }
        for reg in suspicious_regions:
            rx, ry, rw, rh = reg.get("x", 0), reg.get("y", 0), reg.get("width", 0), reg.get("height", 0)
            conf = reg.get("confidence", 0.8)
            added_points = int(conf * 45)
            
            # Map coordinates to document regions
            if rx < (w * 0.40) and ry < (h * 0.70):
                scores["photograph"] = min(95, scores["photograph"] + added_points)
            elif ry > (h * 0.70):
                scores["document_number"] = min(95, scores["document_number"] + added_points)
            elif (w * 0.25) <= rx <= (w * 0.80) and ry < (h * 0.45):
                scores["name"] = min(95, scores["name"] + added_points)
            elif (w * 0.25) <= rx <= (w * 0.80) and (h * 0.45) <= ry <= (h * 0.70):
                scores["dob"] = min(95, scores["dob"] + added_points)
            else:
                scores["background"] = min(95, scores["background"] + added_points)
        return scores

    def _save_visualization(self, img_array: np.ndarray, verification_id: str, tag: str) -> Optional[str]:
        try:
            rel_dir = os.path.join('crops', 'tamper')
            full_dir = os.path.join(settings.MEDIA_ROOT, rel_dir)
            os.makedirs(full_dir, exist_ok=True)

            filename = f"{verification_id}_{tag}.jpg"
            rel_path = os.path.join(rel_dir, filename).replace('\\', '/')
            full_path = os.path.join(full_dir, filename)

            cv2.imwrite(full_path, img_array)
            return rel_path
        except Exception:
            return None
