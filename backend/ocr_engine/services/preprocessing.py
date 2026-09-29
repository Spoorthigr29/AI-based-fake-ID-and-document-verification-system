import os
import io
import math
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, Dict, Any

class ImagePreprocessor:
    """
    Advanced OpenCV-based image preprocessing pipeline for OCR:
    - Auto-deskew / orientation correction
    - Contrast Limited Adaptive Histogram Equalization (CLAHE)
    - Adaptive thresholding & noise reduction
    - Image quality assessment (sharpness, brightness, contrast)
    """

    @staticmethod
    def load_image(image_input):
        """
        Load image from path, bytes, Document, FieldFile, PIL Image, or DocumentContext into a standardized BGR numpy array.
        """
        if hasattr(image_input, 'bgr'):
            return image_input.bgr
        if isinstance(image_input, np.ndarray):
            return image_input
        from documents.services.image_loader import DocumentImageLoader
        return DocumentImageLoader.load(image_input).bgr

    @classmethod
    def calculate_quality_metrics(cls, image_bgr):
        """
        Calculate document visual quality metrics:
        - Sharpness (Laplacian variance)
        - Brightness (mean luminance)
        - Contrast (luminance standard deviation)
        - Overall Quality Index (0.0 to 100.0)
        """
        if image_bgr is None or image_bgr.size == 0:
            return {"sharpness": 0.0, "brightness": 0.0, "contrast": 0.0, "quality_score": 0.0, "is_blurry": True}

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness via Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        
        # 2. Brightness & Contrast
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        
        # Normalize sharpness to 0-100 scale (typical good images have laplacian var > 100)
        norm_sharpness = min(100.0, (sharpness / 200.0) * 100.0)
        # Normalize contrast (typical good contrast > 40)
        norm_contrast = min(100.0, (contrast / 50.0) * 100.0)
        # Normalize brightness (ideal is 110-180)
        brightness_penalty = abs(brightness - 145.0) / 145.0
        norm_brightness = max(0.0, 100.0 * (1.0 - brightness_penalty))

        # Composite score
        quality_score = (norm_sharpness * 0.45) + (norm_contrast * 0.35) + (norm_brightness * 0.20)
        quality_score = round(max(0.0, min(100.0, quality_score)), 2)

        is_blurry = sharpness < 45.0

        return {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "quality_score": quality_score,
            "is_blurry": is_blurry,
            "resolution": f"{image_bgr.shape[1]}x{image_bgr.shape[0]}"
        }

    @classmethod
    def correct_skew(cls, image_bgr, max_angle=45.0):
        """
        Detect skew angle using text contour bounding boxes and rotate image to level.
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr, 0.0

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Invert and binarize
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find all foreground pixel coordinates
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return image_bgr, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle calculation depending on OpenCV rectangle convention
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        if abs(angle) > max_angle or abs(angle) < 0.5:
            return image_bgr, 0.0

        # Rotate image
        (h, w) = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, round(float(angle), 2)

    @classmethod
    def detect_and_warp_document(cls, image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Phase 3: Detect document quad boundary and perform perspective correction.
        Preserves aspect ratio and falls back gracefully to original if boundary is not quadrilateral.
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr, False

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            area = cv2.contourArea(c)
            # Must occupy at least 30% of image area to be considered a document boundary
            if len(approx) == 4 and area > (w * h * 0.30):
                doc_contour = approx
                break

        if doc_contour is None:
            return image_bgr, False

        try:
            # Order 4 corner points: top-left, top-right, bottom-right, bottom-left
            pts = doc_contour.reshape(4, 2).astype("float32")
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)] # Top-left
            rect[2] = pts[np.argmax(s)] # Bottom-right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)] # Top-right
            rect[3] = pts[np.argmax(diff)] # Bottom-left

            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            if maxWidth < 300 or maxHeight < 200:
                return image_bgr, False

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight))
            return warped, True
        except Exception:
            return image_bgr, False

    @classmethod
    def preprocess_for_ocr(cls, image_input, enhance_contrast=True, auto_deskew=True):
        """
        Full OCR preprocessing pipeline:
        1. Load image
        2. Assess quality
        3. Correct skew/rotation
        4. Resize if too small/large
        5. Apply CLAHE contrast enhancement
        6. Generate adaptive thresholded binarized view
        """
        img_bgr = cls.load_image(image_input)
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Invalid or unreadable image data provided to preprocessor.")

        quality_metrics = cls.calculate_quality_metrics(img_bgr)
        
        # Skew correction
        deskew_angle = 0.0
        if auto_deskew:
            img_bgr, deskew_angle = cls.correct_skew(img_bgr)

        # Grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Scale normalization: Ensure optimal DPI / dimensions for OCR (~800-1200 width)
        h, w = gray.shape[:2]
        if w < 700:
            scale = 900.0 / float(w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        elif max(h, w) > 1200:
            scale = 1200.0 / float(max(h, w))
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Contrast enhancement using CLAHE
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
        else:
            enhanced_gray = gray

        # Ultra-fast Gaussian denoising (0.002s vs 0.450s NLM)
        denoised = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)

        # Adaptive thresholding
        binarized = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )

        return {
            "processed_bgr": img_bgr,
            "processed_gray": enhanced_gray,
            "binarized": binarized,
            "deskew_angle": deskew_angle,
            "quality_metrics": quality_metrics,
        }
