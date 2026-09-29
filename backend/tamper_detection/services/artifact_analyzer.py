import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class ArtifactResult:
    artifact_score: float  # 0.0 to 1.0
    noise_variance_discrepancy: float
    block_boundary_discontinuity: float
    signals: List[str] = field(default_factory=list)
    suspicious_regions: List[Dict[str, Any]] = field(default_factory=list)
    metadata_findings: Dict[str, Any] = field(default_factory=dict)

class ArtifactAnalyzer:
    """
    Forensic Artifact, Compression, and Metadata Analysis engine for VerifyX AI.
    Detects noise inconsistencies, double compression block artifacts, and editing software signatures.
    """

    SUSPICIOUS_SOFTWARE_KEYWORDS = [
        'photoshop', 'gimp', 'canva', 'paint.net', 'pixlr', 'corel',
        'lightroom', 'snapseed', 'facetune', 'affinity', 'illustrator'
    ]

    def analyze(self, image_path_or_array) -> ArtifactResult:
        """Execute artifact, noise, and metadata forensic inspection."""
        metadata = self._extract_metadata(image_path_or_array)
        img = self._load_cv2_image(image_path_or_array)

        if img is None:
            return ArtifactResult(
                artifact_score=0.0,
                noise_variance_discrepancy=0.0,
                block_boundary_discontinuity=0.0,
                signals=["Unable to read image for artifact inspection."],
                suspicious_regions=[],
                metadata_findings=metadata
            )

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # 1. High-Frequency Noise Inconsistency Analysis
        noise_discrepancy, noise_regions = self._detect_noise_inconsistency(gray)

        # 2. Block Boundary Discontinuity (JPEG 8x8 Grid Artifacts)
        block_discontinuity = self._detect_block_grid_discontinuity(gray)

        # 3. Metadata Software Signature Check
        signals = []
        meta_score = 0.0
        if metadata.get('software_flagged'):
            meta_score = 0.45
            software_name = metadata.get('software_name', 'Image Editing Software')
            signals.append(f"Image editor metadata tag detected ({software_name})")

        if noise_discrepancy > 0.40:
            signals.append("inconsistent image noise distribution")
            signals.append("localized noise level discrepancy across document")

        if block_discontinuity > 0.45:
            signals.append("double compression block artifact detected")

        suspicious_regions = []
        suspicious_regions.extend(noise_regions)

        # Compute composite artifact score
        raw_score = (noise_discrepancy * 0.45) + (block_discontinuity * 0.25) + meta_score
        artifact_score = round(max(0.05, min(0.95, raw_score)), 2)

        if not signals:
            signals.append("No major artifact detected")

        return ArtifactResult(
            artifact_score=artifact_score,
            noise_variance_discrepancy=round(noise_discrepancy, 2),
            block_boundary_discontinuity=round(block_discontinuity, 2),
            signals=signals,
            suspicious_regions=suspicious_regions,
            metadata_findings=metadata
        )

    def _extract_metadata(self, image_input) -> Dict[str, Any]:
        """Inspect EXIF data and software header tags for editing tools."""
        findings = {
            'has_exif': False,
            'software_flagged': False,
            'software_name': None,
            'tags': {}
        }

        if not isinstance(image_input, str) or not os.path.exists(image_input):
            return findings

        try:
            pil_img = Image.open(image_input)
            exif_data = pil_img.getexif()
            if exif_data:
                findings['has_exif'] = True
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    findings['tags'][tag_name] = str(value)

                    if tag_name.lower() in ['software', 'processingsoftware', 'imagedescription']:
                        val_str = str(value).lower()
                        for kw in self.SUSPICIOUS_SOFTWARE_KEYWORDS:
                            if kw in val_str:
                                findings['software_flagged'] = True
                                findings['software_name'] = str(value)
                                break
        except Exception:
            pass

        return findings

    def _detect_noise_inconsistency(self, gray_img: np.ndarray) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Isolate high-frequency noise residual and compute local noise variance across patches.
        """
        h, w = gray_img.shape
        if h < 64 or w < 64:
            return 0.1, []

        # Noise residual = Original - MedianFilter(Original)
        median_filtered = cv2.medianBlur(gray_img, 3)
        noise_residual = cv2.absdiff(gray_img, median_filtered)

        # Partition into grid cells (e.g. 48x48)
        cell_size = 48
        grid_rows = h // cell_size
        grid_cols = w // cell_size

        variances = []
        regions = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                y1 = r * cell_size
                y2 = min(h, (r + 1) * cell_size)
                x1 = c * cell_size
                x2 = min(w, (c + 1) * cell_size)

                cell = noise_residual[y1:y2, x1:x2]
                var = float(np.var(cell))
                variances.append((var, x1, y1, x2 - x1, y2 - y1))

        if not variances:
            return 0.1, []

        var_vals = [v[0] for v in variances]
        mean_var = np.mean(var_vals)
        std_var = np.std(var_vals)

        # Detect patches with radically mismatched noise (> 2.5 std devs)
        suspicious = []
        for var, x, y, width, height in variances:
            if std_var > 1.0 and (var > mean_var + (2.5 * std_var)):
                suspicious.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                    "type": "noise_variance_anomaly",
                    "confidence": round(min(0.92, (var - mean_var) / (3.0 * std_var + 1e-5)), 2)
                })

        discrepancy_score = min(1.0, len(suspicious) * 0.20 + (std_var / (mean_var + 1e-5)) * 0.3)
        return float(discrepancy_score), suspicious

    def _detect_block_grid_discontinuity(self, gray_img: np.ndarray) -> float:
        """
        Detect 8x8 block boundary discontinuities characteristic of misaligned double JPEG compression.
        """
        h, w = gray_img.shape
        if h < 32 or w < 32:
            return 0.0

        # Differences across 8-pixel boundaries vs interior pixel differences
        diff_h = np.abs(gray_img[1:, :] - gray_img[:-1, :])
        diff_w = np.abs(gray_img[:, 1:] - gray_img[:, :-1])

        # Sample grid boundaries (every 8th line)
        grid_h_indices = [i for i in range(7, diff_h.shape[0], 8)]
        non_grid_h_indices = [i for i in range(diff_h.shape[0]) if (i + 1) % 8 != 0]

        if not grid_h_indices or not non_grid_h_indices:
            return 0.1

        mean_grid_diff = np.mean(diff_h[grid_h_indices, :])
        mean_non_grid_diff = np.mean(diff_h[non_grid_h_indices, :])

        ratio = (mean_grid_diff + 1e-5) / (mean_non_grid_diff + 1e-5)
        # Ratio significantly > 1.25 indicates sharp block boundary artifacts
        score = max(0.0, min(1.0, (ratio - 1.0) * 2.5))
        return float(score)

    def _load_cv2_image(self, img_input) -> Optional[np.ndarray]:
        if hasattr(img_input, 'bgr'):
            return img_input.bgr
        if isinstance(img_input, np.ndarray):
            return img_input
        from documents.services.image_loader import DocumentImageLoader
        try:
            return DocumentImageLoader.load(img_input).bgr
        except Exception:
            return None
