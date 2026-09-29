import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class RegionDetectionResult:
    region_score: float  # 0.0 to 1.0
    boundary_anomaly_score: float
    copy_move_score: float
    text_patch_score: float
    signals: List[str] = field(default_factory=list)
    suspicious_regions: List[Dict[str, Any]] = field(default_factory=list)

class RegionAnomalyDetector:
    """
    Forensic Region and Boundary Anomaly Detection engine for VerifyX AI.
    Detects spliced photo boundaries, copy-move clone regions, and text patch anomalies.
    """

    def analyze(self, image_bgr: np.ndarray) -> RegionDetectionResult:
        """Execute boundary gradient analysis, copy-move detection, and text patch scanning."""
        if image_bgr is None or image_bgr.size == 0:
            return RegionDetectionResult(
                region_score=0.0,
                boundary_anomaly_score=0.0,
                copy_move_score=0.0,
                text_patch_score=0.0,
                signals=["Empty image provided for region anomaly analysis."],
                suspicious_regions=[]
            )

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Detect Photo-Region Boundary & Spliced Rectangular Edge Discontinuities
        boundary_score, boundary_regions, boundary_signals = self._detect_boundary_anomalies(gray, image_bgr)

        # 2. Detect Copy-Move / Duplicated Region Patches
        copy_score, copy_regions, copy_signals = self._detect_copy_move_regions(gray)

        # 3. Detect Text-Region Whiteout / Patching Inconsistencies
        patch_score, patch_regions, patch_signals = self._detect_text_patch_anomalies(gray)

        signals = []
        signals.extend(boundary_signals)
        signals.extend(copy_signals)
        signals.extend(patch_signals)

        all_regions = []
        all_regions.extend(boundary_regions)
        all_regions.extend(copy_regions)
        all_regions.extend(patch_regions)

        # Composite regional score
        raw_score = (boundary_score * 0.40) + (copy_score * 0.35) + (patch_score * 0.25)
        region_score = round(max(0.05, min(0.95, raw_score)), 2)

        if not signals:
            signals.append("No suspicious photo boundary")
            signals.append("No copy-move anomaly detected")

        return RegionDetectionResult(
            region_score=region_score,
            boundary_anomaly_score=round(boundary_score, 2),
            copy_move_score=round(copy_score, 2),
            text_patch_score=round(patch_score, 2),
            signals=signals,
            suspicious_regions=all_regions
        )

    def _detect_boundary_anomalies(self, gray: np.ndarray, color_img: np.ndarray) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Detect unnatural rectangular boundary steps characteristic of pasted/spliced photos or stamps.
        """
        h, w = gray.shape
        signals = []
        regions = []

        # Compute Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)

        # Look for straight horizontal & vertical step edges forming closed rectangular contours
        grad_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, edge_thresh = cv2.threshold(grad_norm, 80, 255, cv2.THRESH_BINARY)

        # Detect rectangular contours with high edge sharpness
        contours, _ = cv2.findContours(edge_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        anomaly_count = 0

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                bx, by, bw, bh = cv2.boundingRect(approx)
                area = bw * bh
                aspect = bh / float(bw + 1e-5)

                # Focus on portrait photo or badge sized rectangles (10% to 50% of document area)
                if (h * w * 0.03) < area < (h * w * 0.50) and (0.6 <= aspect <= 1.8):
                    # Check if perimeter has sharp edge discrepancy vs interior
                    interior = gray[by+5:by+bh-5, bx+5:bx+bw-5] if bw > 10 and bh > 10 else None
                    if interior is not None and interior.size > 0:
                        border_grad = np.mean(magnitude[by:by+bh, bx]) + np.mean(magnitude[by:by+bh, bx+bw-1])
                        interior_grad = np.mean(magnitude[by+5:by+bh-5, bx+5:bx+bw-5])
                        
                        if border_grad > (interior_grad * 2.8) and border_grad > 60:
                            anomaly_count += 1
                            regions.append({
                                "x": int(bx),
                                "y": int(by),
                                "width": int(bw),
                                "height": int(bh),
                                "type": "suspicious_image_boundary",
                                "confidence": 0.86
                            })

        score = min(1.0, anomaly_count * 0.45)
        if anomaly_count > 0:
            signals.append("suspicious image boundary")
            signals.append("unnatural rectangular edge gradient discontinuity")

        return float(score), regions, signals

    def _detect_copy_move_regions(self, gray: np.ndarray) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Detect cloned visual patches (copy-move forgery) using ORB keypoint descriptor matching.
        """
        h, w = gray.shape
        signals = []
        regions = []

        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) < 20:
            return 0.0, [], []

        # Match keypoints with themselves to find duplicate clusters
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            matches = matcher.knnMatch(descriptors, descriptors, k=3)
        except Exception:
            return 0.0, [], []

        clone_pairs = []
        min_spatial_dist = 40 # Cloned regions must be at least 40px apart

        for m_list in matches:
            if len(m_list) >= 2:
                # 0 is the keypoint matching itself; check 1
                m = m_list[1]
                if m.distance < 25: # Very strong descriptor match
                    kp1 = keypoints[m.queryIdx].pt
                    kp2 = keypoints[m.trainIdx].pt
                    dist = np.sqrt((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)

                    if dist >= min_spatial_dist:
                        clone_pairs.append((kp1, kp2))

        score = 0.0
        if len(clone_pairs) >= 6:
            score = min(1.0, 0.40 + (len(clone_pairs) * 0.05))
            signals.append("potential copy/move duplicated patch detected")

            # Create bounding region around dense pair cluster
            pts = np.array([p[0] for p in clone_pairs] + [p[1] for p in clone_pairs])
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)

            regions.append({
                "x": int(max(0, x_min - 10)),
                "y": int(max(0, y_min - 10)),
                "width": int(min(w, x_max - x_min + 20)),
                "height": int(min(h, y_max - y_min + 20)),
                "type": "copy_move_cloned_region",
                "confidence": round(min(0.92, 0.50 + len(clone_pairs) * 0.04), 2)
            })

        return float(score), regions, signals

    def _detect_text_patch_anomalies(self, gray: np.ndarray) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Detect solid paint-over / white-out patches inserted behind altered text fields.
        """
        h, w = gray.shape
        signals = []
        regions = []

        # Look for unnaturally flat luminance patches in textured backgrounds
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        abs_lap = np.abs(laplacian)

        # 24x24 grid cells
        cell_size = 24
        rows = h // cell_size
        cols = w // cell_size

        patch_count = 0
        for r in range(rows):
            for c in range(cols):
                y1 = r * cell_size
                y2 = (r + 1) * cell_size
                x1 = c * cell_size
                x2 = (c + 1) * cell_size

                cell_lap = abs_lap[y1:y2, x1:x2]
                cell_gray = gray[y1:y2, x1:x2]

                mean_val = np.mean(cell_gray)
                var_val = np.var(cell_gray)

                # Pure uniform solid patch in document body (near-zero texture variance)
                if var_val < 1.0 and (30 < mean_val < 245):
                    patch_count += 1
                    if patch_count <= 4:
                        regions.append({
                            "x": int(x1),
                            "y": int(y1),
                            "width": int(cell_size),
                            "height": int(cell_size),
                            "type": "text_patch_anomaly",
                            "confidence": 0.78
                        })

        score = min(1.0, patch_count * 0.15)
        if patch_count >= 2:
            signals.append("unnatural flat texture patch detected")

        return float(score), regions, signals
