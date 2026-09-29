import os
import io
import math
import numpy as np
import cv2
from PIL import Image
import pypdf

class DocumentQualityAnalyzer:
    """
    Comprehensive document quality and readability analysis service.
    Calculates:
    - Resolution & Image Dimensions
    - Blur & Sharpness (Laplacian variance & edge frequency)
    - Brightness & Exposure (Mean luminance)
    - Contrast (Standard deviation / RMS contrast)
    - Skew / Rotation angle
    - Readability & Text Edge Density
    - Composite 0–100 Quality Score & Issues Breakdown
    """

    @staticmethod
    def load_image_bgr(file_or_path):
        """Standardize any file input (path, bytes, PDF, PIL, DocumentContext) into a BGR numpy array."""
        if hasattr(file_or_path, 'bgr'):
            return file_or_path.bgr
        if isinstance(file_or_path, np.ndarray):
            return file_or_path
        from .image_loader import DocumentImageLoader
        ctx = DocumentImageLoader.load(file_or_path)
        return ctx.bgr

        if isinstance(file_or_path, str):
            img = cv2.imread(file_or_path)
            if img is None:
                pil_img = Image.open(file_or_path).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        elif hasattr(file_or_path, 'read'):
            pos = file_or_path.tell() if hasattr(file_or_path, 'tell') else 0
            b = file_or_path.read()
            if hasattr(file_or_path, 'seek'):
                file_or_path.seek(pos)
            nparr = np.frombuffer(b, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                pil_img = Image.open(io.BytesIO(b)).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        elif isinstance(file_or_path, bytes):
            nparr = np.frombuffer(file_or_path, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                pil_img = Image.open(io.BytesIO(file_or_path)).convert('RGB')
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        elif isinstance(file_or_path, Image.Image):
            return cv2.cvtColor(np.array(file_or_path.convert('RGB')), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported image input type: {type(file_or_path)}")

    @classmethod
    def detect_skew_angle(cls, gray):
        """Calculate skew angle using minAreaRect on foreground text contours."""
        try:
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) < 80:
                return 0.0
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle
            return round(float(angle), 2)
        except Exception:
            return 0.0

    @classmethod
    def analyze_quality(cls, file_or_path):
        """
        Execute full quality diagnostics and return:
        - quality_score: int (0–100)
        - issues: list of string warnings
        - checklist: list of passed criteria
        - metrics: detailed breakdown dictionary
        - is_poor_quality: boolean indicating manual review routing
        """
        img_bgr = cls.load_image_bgr(file_or_path)
        if img_bgr is None or img_bgr.size == 0:
            return {
                "quality_score": 0,
                "issues": ["Image file is empty, unreadable, or corrupted"],
                "checklist": [],
                "metrics": {
                    "resolution": "0x0",
                    "width": 0,
                    "height": 0,
                    "blur_score": 0.0,
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "skew_angle": 0.0,
                    "readability": "Unreadable"
                },
                "is_poor_quality": True,
                "recommended_status": "MANUAL_REVIEW"
            }

        height, width = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        issues = []
        checklist = []

        # 1. Resolution & Dimensions Check
        megapixels = round((width * height) / 1_000_000, 2)
        aspect_ratio = round(width / float(max(1, height)), 2)
        
        score_resolution = 100.0
        if width < 500 or height < 350:
            score_resolution = 35.0
            issues.append(f"Low image resolution ({width}x{height}px). Minimum recommended is 800x500px.")
        elif width < 800 or height < 500:
            score_resolution = 70.0
            issues.append(f"Moderate image resolution ({width}x{height}px).")
        else:
            checklist.append("✓ Resolution acceptable")

        # 2. Blur / Sharpness Check (Laplacian Variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_val = float(laplacian.var())
        
        score_blur = 100.0
        if blur_val < 35.0:
            score_blur = 20.0
            issues.append(f"High blur detected (Sharpness index: {blur_val:.1f}). Document may be out of focus.")
        elif blur_val < 75.0:
            score_blur = 65.0
            issues.append(f"Minor blur detected (Sharpness index: {blur_val:.1f}).")
        else:
            checklist.append("✓ Low blur")

        # 3. Brightness & Exposure Check (Mean Luminance in 0–255)
        brightness = float(np.mean(gray))
        score_brightness = 100.0
        
        if brightness < 65.0:
            score_brightness = 30.0
            issues.append(f"Document is underexposed / too dark (Brightness: {brightness:.1f}/255).")
        elif brightness > 225.0:
            score_brightness = 35.0
            issues.append(f"Document is overexposed or has severe glare (Brightness: {brightness:.1f}/255).")
        elif brightness < 90.0 or brightness > 195.0:
            score_brightness = 75.0
            issues.append(f"Sub-optimal lighting conditions (Brightness: {brightness:.1f}/255).")
        else:
            checklist.append("✓ Brightness acceptable")

        # 4. Contrast Check (Standard Deviation of Luminance)
        contrast = float(np.std(gray))
        score_contrast = 100.0
        
        if contrast < 25.0:
            score_contrast = 25.0
            issues.append(f"Low contrast detected (Contrast: {contrast:.1f}). Text separation may be degraded.")
        elif contrast < 40.0:
            score_contrast = 65.0
            issues.append(f"Moderate contrast (Contrast: {contrast:.1f}).")
        else:
            checklist.append("✓ Contrast acceptable")

        # 5. Skew / Rotation Angle
        skew_angle = cls.detect_skew_angle(gray)
        score_skew = 100.0
        
        if abs(skew_angle) > 25.0:
            score_skew = 40.0
            issues.append(f"Severe document rotation/tilt detected ({skew_angle}°).")
        elif abs(skew_angle) > 10.0:
            score_skew = 75.0
            issues.append(f"Document slightly tilted ({skew_angle}°).")
        else:
            checklist.append("✓ Alignment acceptable")

        # 6. Readability Metric (Text Edge Sharpness & Local Gradient Distribution)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        score_readability = 100.0
        if edge_magnitude < 8.0:
            score_readability = 30.0
            readability_label = "Poor"
            issues.append("Low text edge definition; document may lack clear legible characters.")
        elif edge_magnitude < 15.0:
            score_readability = 70.0
            readability_label = "Moderate"
        else:
            readability_label = "High / Crisp"
            checklist.append("✓ Text readable")

        # Weighted composite score (0–100)
        composite_score = (
            (score_resolution * 0.20) +
            (score_blur * 0.30) +
            (score_brightness * 0.15) +
            (score_contrast * 0.15) +
            (score_skew * 0.10) +
            (score_readability * 0.10)
        )
        final_quality_score = round(max(0.0, min(100.0, composite_score)))

        # Quality Category (GOOD / ACCEPTABLE / POOR / INCONCLUSIVE)
        if final_quality_score >= 80:
            quality_status = "GOOD"
            guidance = "Image quality is good for screening analysis."
        elif final_quality_score >= 55:
            quality_status = "ACCEPTABLE"
            guidance = "Image quality is acceptable with minor baseline variations."
        elif final_quality_score >= 35:
            quality_status = "POOR"
            guidance = "Image quality is insufficient for reliable verification. Please upload a clearer image."
        else:
            quality_status = "INCONCLUSIVE"
            guidance = "Image quality is insufficient for reliable verification. Please upload a clearer image."

        # Manual Review Routing Threshold (poor quality triggers low-quality alert rather than fake categorization)
        is_poor_quality = (final_quality_score < 55) or (score_blur <= 20.0) or (score_brightness <= 30.0)
        recommended_status = "MANUAL_REVIEW" if is_poor_quality else "UPLOADED"

        return {
            "quality_score": final_quality_score,
            "status": quality_status,
            "guidance": guidance,
            "issues": issues,
            "checklist": checklist,
            "metrics": {
                "resolution": f"{width}x{height}",
                "width": width,
                "height": height,
                "megapixels": megapixels,
                "aspect_ratio": aspect_ratio,
                "blur": round(blur_val, 2),
                "brightness": round(brightness, 2),
                "contrast": round(contrast, 2),
                "skew_angle": skew_angle,
                "readability": readability_label,
            },
            "is_poor_quality": is_poor_quality,
            "recommended_status": recommended_status
        }
