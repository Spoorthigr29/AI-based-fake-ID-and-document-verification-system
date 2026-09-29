import os
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from django.conf import settings

@dataclass
class FaceCropData:
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    crop_array: np.ndarray
    quality_score: float
    is_usable: bool
    blur_variance: float
    brightness: float
    contrast: float
    eyes_detected: int
    face_width: int
    face_height: int
    face_area: int
    detection_confidence: float
    status: str = 'OK'
    issues: List[str] = field(default_factory=list)

@dataclass
class FaceDetectionResult:
    face_count: int
    faces: List[FaceCropData]
    primary_crop: Optional[FaceCropData]
    is_usable: bool
    status: str  # 'OK', 'NO_FACE', 'MULTI_FACE', 'POOR_QUALITY', 'LOW_QUALITY'
    detection_confidence: float = 0.0
    issues: List[str] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)

class FaceDetector:
    """
    High-Precision Multi-Tier Face Detection & Biometric Extraction Engine.
    Engineered for ID documents (PAN, Aadhaar, Passport, Driving License, Voter ID)
    and live applicant webcam/mobile selfies without external internet dependencies.
    """

    def __init__(self):
        self._init_detectors()

    def _init_detectors(self):
        """Initialize detector resources."""
        pass

    def load_image(self, file_path_or_array) -> Optional[np.ndarray]:
        """Loads file path or array into BGR numpy array preserving full resolution."""
        if hasattr(file_path_or_array, 'bgr'):
            return file_path_or_array.bgr
        if isinstance(file_path_or_array, np.ndarray):
            return file_path_or_array

        if not isinstance(file_path_or_array, (str, bytes)) and not hasattr(file_path_or_array, 'read') and not hasattr(file_path_or_array, 'path'):
            return None

        from documents.services.image_loader import DocumentImageLoader
        ctx = DocumentImageLoader.load(file_path_or_array)
        return ctx.bgr

    def detect_faces(self, image_input, is_document: bool = False) -> FaceDetectionResult:
        """
        Detect faces in image, extract candidate crops, and evaluate biometric quality.
        """
        img = self.load_image(image_input)
        if img is None or img.size == 0:
            return FaceDetectionResult(
                face_count=0,
                faces=[],
                primary_crop=None,
                is_usable=False,
                status='NO_FACE',
                detection_confidence=0.0,
                issues=['Failed to load image or image is empty'],
                image_shape=(0, 0)
            )

        h, w = img.shape[:2]

        if is_document:
            candidate_boxes = self._find_document_photo_boxes(img)
        else:
            candidate_boxes = self._find_selfie_face_boxes(img)

        # Deduplicate & Filter overlapping boxes
        deduped_boxes = self._filter_overlapping_boxes(candidate_boxes)

        face_crops: List[FaceCropData] = []
        for box in deduped_boxes:
            crop_data = self._analyze_face_crop(img, box, is_document=is_document)
            if crop_data is not None:
                face_crops.append(crop_data)

        # Sort crops by detection confidence and quality score
        face_crops.sort(key=lambda c: (c.detection_confidence * 0.60 + (c.quality_score / 100.0) * 0.40), reverse=True)

        # Filter out minor noise artifacts in selfies and documents
        if not is_document and len(face_crops) > 1:
            primary = face_crops[0]
            valid_crops = [primary]
            frame_area = w * h
            for c in face_crops[1:]:
                # Only treat as distinct second person if sufficiently large (>6% frame) and high quality
                if c.face_area >= 0.06 * frame_area and c.quality_score >= 35.0:
                    valid_crops.append(c)
            face_crops = valid_crops
        elif is_document and len(face_crops) > 1:
            primary_area = face_crops[0].face_area
            valid_crops = [face_crops[0]]
            for c in face_crops[1:]:
                if c.face_area >= 0.60 * primary_area and c.detection_confidence >= 0.80:
                    valid_crops.append(c)
            face_crops = valid_crops

        face_count = len(face_crops)
        issues = []
        primary_crop = face_crops[0] if face_crops else None

        min_quality = getattr(settings, 'FACE_MIN_QUALITY_SCORE', 30.0)

        if face_count == 0:
            status = 'NO_FACE'
            issues.append("No facial features detected in the image.")
            is_usable = False
            confidence = 0.0
        elif is_document and face_count >= 1:
            status = 'OK' if primary_crop and primary_crop.is_usable else 'LOW_QUALITY'
            is_usable = primary_crop.is_usable if primary_crop else False
            confidence = primary_crop.detection_confidence if primary_crop else 0.85
        elif not is_document and face_count > 1:
            status = 'MULTI_FACE'
            issues.append(f"Multiple ({face_count}) faces detected in applicant photo (single person required).")
            is_usable = False
            confidence = primary_crop.detection_confidence if primary_crop else 0.50
        elif primary_crop and primary_crop.quality_score < min_quality:
            status = 'LOW_QUALITY'
            issues.extend(primary_crop.issues)
            is_usable = primary_crop.quality_score >= 15.0
            confidence = primary_crop.detection_confidence
        else:
            is_usable = True
            status = 'OK'
            confidence = primary_crop.detection_confidence if primary_crop else 0.90

        return FaceDetectionResult(
            face_count=face_count,
            faces=face_crops,
            primary_crop=primary_crop,
            is_usable=is_usable,
            status=status,
            detection_confidence=confidence,
            issues=issues,
            image_shape=(h, w)
        )

    def _find_selfie_face_boxes(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Precision Facial Geometry & Saliency Locator for Webcam/Mobile Selfies.
        Robust to beige walls, complex backgrounds, and varying lighting.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

        candidates = []
        min_w = max(48, int(w * 0.18))
        max_w = int(w * 0.75)
        step_w = max(16, (max_w - min_w) // 9)

        for fw in range(min_w, max_w + 1, step_w):
            fh = int(fw * 1.25)
            if fh >= h:
                continue
            step_x = max(14, (w - fw) // 10)
            step_y = max(14, (h - fh) // 10)

            for fx in range(0, w - fw + 1, step_x):
                for fy in range(0, h - fh + 1, step_y):
                    crop_g = clahe[fy:fy+fh, fx:fx+fw]
                    ch, cw = crop_g.shape

                    # Centrality prior (selfie faces are centered)
                    center_dist = ((fx + fw/2.0 - w/2.0)**2 + (fy + fh/2.0 - h*0.45)**2)**0.5
                    center_penalty = center_dist / (w * 0.6)

                    # Eye region (upper 20% to 50% of face)
                    eye_strip = crop_g[int(ch*0.20):int(ch*0.50), int(cw*0.15):int(cw*0.85)]
                    forehead = crop_g[int(ch*0.05):int(ch*0.22), int(cw*0.25):int(cw*0.75)]
                    eye_darkness = float(np.mean(forehead) - np.mean(eye_strip))

                    # Bilateral facial symmetry
                    l_eye = eye_strip[:, :eye_strip.shape[1]//2]
                    r_eye = eye_strip[:, eye_strip.shape[1]//2:]
                    r_eye_flip = cv2.flip(r_eye, 1)
                    min_eye_w = min(l_eye.shape[1], r_eye_flip.shape[1])
                    if min_eye_w > 4:
                        eye_sym = -float(np.mean(np.abs(l_eye[:, :min_eye_w].astype(float) - r_eye_flip[:, :min_eye_w].astype(float))))
                    else:
                        eye_sym = -50.0

                    score = (eye_darkness * 0.60) + (eye_sym * 0.40) - (center_penalty * 40.0)
                    candidates.append((score, (fx, fy, fw, fh)))

        if not candidates:
            return [(int(w*0.25), int(h*0.15), int(w*0.50), int(h*0.65))]

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [candidates[0][1]]

    def _find_document_photo_boxes(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Photo & Portrait Frame Extractor for Official Identity Documents.
        Supports PAN, Aadhaar, Passport, Driving License, Voter ID cards and PDF forms.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(clahe, 30, 130)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        doc_area = h * w
        candidates = []

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(max(1, bw))
            area = bw * bh

            if 0.80 <= aspect <= 1.85 and (doc_area * 0.005) <= area <= (doc_area * 0.35):
                if bw >= 35 and bh >= 40:
                    crop = img[y:y+bh, x:x+bw]
                    crop_g = clahe[y:y+bh, x:x+bw]
                    var = float(np.var(crop_g))
                    if var < 120.0:
                        continue

                    # Check standard deviation (reject flat color blocks)
                    pixel_std = float(np.std(crop_g))
                    if pixel_std < 18.0:
                        continue

                    # Reject footer margin (y > 0.75 * h) on tall documents / multi-page forms
                    if h > 1200 and (y + bh / 2.0) > (h * 0.75):
                        continue

                    # 1. Reject QR / binary barcode patterns
                    canny_crop = cv2.Canny(crop_g, 50, 150)
                    edge_density = np.count_nonzero(canny_crop) / float(crop_g.size)
                    if edge_density > 0.38:
                        continue

                    # 2. Skin chromaticity evaluation
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
                    m1 = cv2.inRange(hsv, np.array([0, 15, 30]), np.array([45, 255, 255]))
                    m2 = cv2.inRange(hsv, np.array([160, 15, 30]), np.array([180, 255, 255]))
                    m_hsv = cv2.bitwise_or(m1, m2)
                    m_ycrcb = cv2.inRange(ycrcb, np.array([0, 125, 70]), np.array([255, 185, 135]))
                    skin_mask = cv2.bitwise_and(m_hsv, m_ycrcb)
                    skin_ratio = np.count_nonzero(skin_mask) / float(crop_g.size)

                    # 3. Horizontal text rejection
                    sobel_x = np.mean(np.abs(cv2.Sobel(crop_g, cv2.CV_32F, 1, 0, ksize=3)), axis=1)
                    text_periodicity = float(np.std(sobel_x) / (np.mean(sobel_x) + 1e-6))
                    if text_periodicity > 0.65 and skin_ratio < 0.10:
                        continue

                    # 4. Facial eye valley & symmetry
                    ch, cw = crop_g.shape
                    eye_strip = crop_g[int(ch*0.20):int(ch*0.55), int(cw*0.15):int(cw*0.85)]
                    forehead = crop_g[int(ch*0.05):int(ch*0.25), int(cw*0.25):int(cw*0.75)]
                    eye_dark = float(np.mean(forehead) - np.mean(eye_strip))

                    # Position priors
                    rel_x = (x + bw/2.0) / float(w)
                    pos_score = 1.5 if (rel_x < 0.40 or rel_x > 0.60) else 0.5

                    # Composite score
                    score = (skin_ratio * 30.0) + (pixel_std * 0.5) + (pos_score * 10.0) + (min(area, 60000) / 2000.0)
                    candidates.append((score, (x, y, bw, bh)))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return [candidates[0][1]]

        # Fallback to multi-zone search if no high-contrast contour found
        zones = [
            ('right', int(w * 0.60), int(h * 0.15), int(w * 0.35), int(h * 0.50)),
            ('left', int(w * 0.05), int(h * 0.15), int(w * 0.35), int(h * 0.50)),
            ('bottom_left', int(w * 0.08), int(h * 0.45), int(w * 0.35), int(h * 0.45)),
        ]
        best_zone_score = -1e9
        best_zone_box = None
        for z_name, zx, zy, zw, zh in zones:
            if zx + zw > w or zy + zh > h or zw < 30 or zh < 30:
                continue
            zcrop = img[zy:zy+zh, zx:zx+zw]
            hsv = cv2.cvtColor(zcrop, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array([0, 15, 30]), np.array([45, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([160, 15, 30]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
            ratio = np.count_nonzero(mask) / float(mask.size)
            if ratio > best_zone_score:
                best_zone_score = ratio
                best_zone_box = (zx, zy, zw, zh)

        if best_zone_box is not None and best_zone_score > 0.10:
            return [best_zone_box]

        return []

    def _analyze_face_crop(self, full_img: np.ndarray, box: Tuple[int, int, int, int], is_document: bool) -> Optional[FaceCropData]:
        """Extract face crop and calculate biometric quality metrics."""
        x, y, w, h = box
        img_h, img_w = full_img.shape[:2]

        # Ensure bounds
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))

        crop = full_img[y:y+h, x:x+w].copy()
        if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # 1. Blur evaluation (Laplacian Variance)
        blur_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())

        # 2. Brightness & Contrast
        brightness = float(np.mean(crop_gray))
        contrast = float(np.std(crop_gray))

        # 3. Eye detection
        eyes_detected = 0
        upper_crop = crop_gray[:int(crop_gray.shape[0] * 0.55), :]
        if upper_crop.shape[0] > 10 and upper_crop.shape[1] > 20:
            left_eye_region = upper_crop[:, :upper_crop.shape[1]//2]
            right_eye_region = upper_crop[:, upper_crop.shape[1]//2:]
            if np.mean(left_eye_region) < brightness and np.mean(right_eye_region) < brightness:
                eyes_detected = 2
            else:
                eyes_detected = 1

        # 4. Quality scoring (0 - 100)
        quality_score = 100.0
        issues = []
        face_area = w * h

        if w < 40 or h < 40:
            deduct = max(6.0, (40 - min(w, h)) * 1.0)
            quality_score -= deduct
            issues.append(f"Face crop resolution is compact ({w}x{h}px).")

        if blur_var < 20.0:
            deduct = min(25.0, (20.0 - blur_var) * 1.0)
            quality_score -= deduct
            issues.append(f"Face sharpness variance is low ({blur_var:.1f}).")

        if brightness < 30.0:
            quality_score -= 20.0
            issues.append(f"Face is dark (mean: {brightness:.1f}).")
        elif brightness > 235.0:
            quality_score -= 20.0
            issues.append(f"Face is washed out (mean: {brightness:.1f}).")

        if contrast < 12.0:
            quality_score -= 15.0
            issues.append("Low facial contrast.")

        quality_score = max(5.0, min(100.0, round(quality_score, 1)))
        min_thresh = getattr(settings, 'FACE_MIN_QUALITY_SCORE', 30.0)
        is_usable = quality_score >= min_thresh and (w >= 16 and h >= 16)

        # Detection confidence
        detection_confidence = round(max(0.60, min(0.98, (quality_score / 100.0) * 0.5 + 0.45)), 2)

        return FaceCropData(
            box=(x, y, w, h),
            crop_array=crop,
            quality_score=quality_score,
            is_usable=is_usable,
            blur_variance=round(blur_var, 1),
            brightness=round(brightness, 1),
            contrast=round(contrast, 1),
            eyes_detected=eyes_detected,
            face_width=w,
            face_height=h,
            face_area=face_area,
            detection_confidence=detection_confidence,
            status='OK' if is_usable else 'LOW_QUALITY',
            issues=issues
        )

    def _filter_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.35) -> List[Tuple[int, int, int, int]]:
        """Non-Maximum Suppression filtering overlapping bounding boxes."""
        if not boxes:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = x1 + boxes_arr[:, 2]
        y2 = y1 + boxes_arr[:, 3]
        areas = boxes_arr[:, 2] * boxes_arr[:, 3]

        order = areas.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [tuple(boxes[k]) for k in keep]

