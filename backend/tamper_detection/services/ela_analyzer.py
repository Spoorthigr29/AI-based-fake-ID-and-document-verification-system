import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

@dataclass
class ELAResult:
    ela_score: float  # 0.0 to 1.0
    mean_error: float
    max_error: float
    error_variance: float
    signals: List[str] = field(default_factory=list)
    suspicious_regions: List[Dict[str, Any]] = field(default_factory=list)
    ela_image_array: Optional[np.ndarray] = None

class ELAAnalyzer:
    """
    Error Level Analysis (ELA) engine for detecting digital image manipulation.
    Identifies variations in JPEG compression error levels across image regions.
    """

    def __init__(self, quality: int = 90, scale_factor: float = 15.0):
        self.quality = quality
        self.scale_factor = scale_factor

    def analyze(self, image_input) -> ELAResult:
        """
        Execute Error Level Analysis on image input (file path, PIL Image, or BGR numpy array).
        """
        pil_img = self._to_pil_image(image_input)
        if pil_img is None:
            return ELAResult(
                ela_score=0.0,
                mean_error=0.0,
                max_error=0.0,
                error_variance=0.0,
                signals=["Unable to process image for ELA analysis."],
                suspicious_regions=[]
            )

        # 1. Recompress in memory at specified JPEG quality
        buffer = io.BytesIO()
        pil_img.save(buffer, 'JPEG', quality=self.quality)
        buffer.seek(0)
        recompressed_img = Image.open(buffer)

        # 2. Compute absolute difference image
        orig_arr = np.array(pil_img, dtype=np.float32)
        recomp_arr = np.array(recompressed_img, dtype=np.float32)

        # Handle size mismatch if any
        if orig_arr.shape != recomp_arr.shape:
            orig_arr = cv2.resize(orig_arr, (recomp_arr.shape[1], recomp_arr.shape[0]))

        diff = np.abs(orig_arr - recomp_arr)
        
        # 3. Enhance difference for visualization (scale dynamic range)
        ela_visual = np.clip(diff * self.scale_factor, 0, 255).astype(np.uint8)

        # 4. Statistical Analysis of Error Levels
        gray_diff = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(diff.shape) == 3 else diff.astype(np.uint8)
        
        mean_err = float(np.mean(gray_diff))
        max_err = float(np.max(gray_diff))
        err_var = float(np.var(gray_diff))

        # 5. Grid-based Local Inconsistency Detection (32x32 blocks)
        h, w = gray_diff.shape
        block_size = 32
        grid_rows = max(1, h // block_size)
        grid_cols = max(1, w // block_size)

        block_means = []
        block_coords = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                y1 = r * block_size
                y2 = min(h, (r + 1) * block_size)
                x1 = c * block_size
                x2 = min(w, (c + 1) * block_size)

                block = gray_diff[y1:y2, x1:x2]
                b_mean = float(np.mean(block))
                block_means.append(b_mean)
                block_coords.append((x1, y1, x2 - x1, y2 - y1))

        block_means_arr = np.array(block_means)
        global_block_mean = np.mean(block_means_arr) if len(block_means_arr) > 0 else 0.0
        global_block_std = np.std(block_means_arr) if len(block_means_arr) > 0 else 1.0

        # Detect anomalous high-error outlier clusters (> 2.8 std deviations above mean)
        suspicious_regions = []
        signals = []
        outlier_count = 0

        # Morphological clustering of high error mask
        thresh_val = global_block_mean + (3.0 * global_block_std)
        if thresh_val > 8.0:
            _, high_err_mask = cv2.threshold(gray_diff, int(thresh_val), 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed_mask = cv2.morphologyEx(high_err_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 400: # Minimum cluster size for genuine splices
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    # Exclude entire full image borders
                    if bw < w * 0.95 and bh < h * 0.95:
                        outlier_count += 1
                        conf = min(0.95, round(float(area) / (area + 500) + 0.4, 2))
                        suspicious_regions.append({
                            "x": bx,
                            "y": by,
                            "width": bw,
                            "height": bh,
                            "type": "inconsistent_compression_region",
                            "confidence": conf
                        })

        # Calculate ELA Tampering Score (0.00 to 1.00)
        # Spliced regions create localized high variance and distinct error hotspots
        variance_factor = min(1.0, err_var / 35.0)
        outlier_factor = min(1.0, outlier_count * 0.25)
        
        raw_ela_score = (variance_factor * 0.45) + (outlier_factor * 0.55)
        ela_score = round(max(0.05, min(0.95, raw_ela_score)), 2)

        if outlier_count > 0:
            signals.append("inconsistent compression region")
            if outlier_count > 2:
                signals.append("localized ELA error discrepancy detected")
        else:
            signals.append("uniform compression error levels across document")

        return ELAResult(
            ela_score=ela_score,
            mean_error=round(mean_err, 2),
            max_error=round(max_err, 2),
            error_variance=round(err_var, 2),
            signals=signals,
            suspicious_regions=suspicious_regions,
            ela_image_array=ela_visual
        )

    def _to_pil_image(self, img_input) -> Optional[Image.Image]:
        if hasattr(img_input, 'pil'):
            return img_input.pil
        if isinstance(img_input, Image.Image):
            return img_input.convert('RGB')
        if isinstance(img_input, np.ndarray):
            rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB) if len(img_input.shape) == 3 else img_input
            return Image.fromarray(rgb)
        from documents.services.image_loader import DocumentImageLoader
        try:
            return DocumentImageLoader.load(img_input).pil
        except Exception:
            return None
