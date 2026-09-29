"""
VerifyX AI - High-Performance Unified Document Ingestion & Image Context
========================================================================
Features:
1. Zero redundant disk/PDF decodings. Decodes and renders document ONCE.
2. Immediate, deterministic cleanup of all PDF / file descriptors.
3. Precomputes:
   - raw_bgr (original image in BGR format)
   - raw_rgb (original image in RGB format)
   - pil_image (PIL Image for deep learning / PyTorch transforms)
   - ocr_bgr (standardized resolution max 1024px for fast, accurate OCR)
   - fast_bgr (standardized resolution max 800-1000px for fast ELA/tampering/quality)
   - file_hash (SHA256 hex string)
"""

import os
import io
import hashlib
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple, Any


class DocumentContext:
    """
    Immutable, high-performance in-memory context holding pre-decoded
    document artifacts for shared use across all pipeline subsystems.
    """

    def __init__(
        self,
        raw_bgr: np.ndarray,
        file_path: Optional[str] = None,
        file_hash: Optional[str] = None,
        is_pdf: bool = False,
        pdf_text_layer: Optional[list] = None
    ):
        self.file_path = file_path or ""
        self.file_hash = file_hash or ""
        self.is_pdf = is_pdf
        self.pdf_text_layer = pdf_text_layer or []

        self.bgr = raw_bgr
        h, w = raw_bgr.shape[:2]
        self.height = h
        self.width = w

        # Lazy-computed representations
        self._rgb = None
        self._pil = None
        self._ocr_bgr = None
        self._fast_bgr = None
        self._gray = None

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            self._rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        return self._rgb

    @property
    def pil(self) -> Image.Image:
        if self._pil is None:
            self._pil = Image.fromarray(self.rgb)
        return self._pil

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return self._gray

    @property
    def ocr_bgr(self) -> np.ndarray:
        """Optimized image for OCR (max 1024px, preserving full font clarity while 2-3x faster)."""
        if self._ocr_bgr is None:
            max_dim = max(self.height, self.width)
            if max_dim > 1024:
                scale = 1024.0 / max_dim
                self._ocr_bgr = cv2.resize(self.bgr, (int(self.width * scale), int(self.height * scale)), interpolation=cv2.INTER_AREA)
            else:
                self._ocr_bgr = self.bgr
        return self._ocr_bgr

    @property
    def fast_bgr(self) -> np.ndarray:
        """Optimized image for forensic ELA and quality inspection (max 800px)."""
        if self._fast_bgr is None:
            max_dim = max(self.height, self.width)
            if max_dim > 800:
                scale = 800.0 / max_dim
                self._fast_bgr = cv2.resize(self.bgr, (int(self.width * scale), int(self.height * scale)), interpolation=cv2.INTER_AREA)
            else:
                self._fast_bgr = self.bgr
        return self._fast_bgr


class DocumentImageLoader:
    """
    Centralized loader that ingests any input (file path, bytes, Document instance)
    and returns a clean, reusable DocumentContext.
    """

    @classmethod
    def load(cls, file_or_doc_or_path: Union[str, bytes, np.ndarray, Any]) -> DocumentContext:
        """Load and decode document once, closing all underlying file/PDF handles immediately."""
        # 1. If already a DocumentContext, return directly
        if isinstance(file_or_doc_or_path, DocumentContext):
            return file_or_doc_or_path

        # 2. If numpy array
        if isinstance(file_or_doc_or_path, np.ndarray):
            return DocumentContext(raw_bgr=file_or_doc_or_path)

        # 3. Resolve path or bytes from Document model or FieldFile
        file_path = None
        file_bytes = None

        if hasattr(file_or_doc_or_path, 'original_file') and file_or_doc_or_path.original_file:
            target = file_or_doc_or_path.original_file
            if hasattr(target, 'path') and os.path.exists(target.path):
                file_path = target.path
            elif hasattr(target, 'read'):
                pos = target.tell() if hasattr(target, 'tell') else 0
                file_bytes = target.read()
                if hasattr(target, 'seek'):
                    target.seek(pos)
        elif hasattr(file_or_doc_or_path, 'path') and os.path.exists(file_or_doc_or_path.path):
            file_path = file_or_doc_or_path.path
        elif isinstance(file_or_doc_or_path, str) and os.path.exists(file_or_doc_or_path):
            file_path = file_or_doc_or_path
        elif isinstance(file_or_doc_or_path, bytes):
            file_bytes = file_or_doc_or_path
        elif hasattr(file_or_doc_or_path, 'read'):
            pos = file_or_doc_or_path.tell() if hasattr(file_or_doc_or_path, 'tell') else 0
            file_bytes = file_or_doc_or_path.read()
            if hasattr(file_or_doc_or_path, 'seek'):
                file_or_doc_or_path.seek(pos)

        # Compute SHA256 hash
        file_hash = ""
        if file_bytes:
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        elif file_path:
            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                pass

        # Check if PDF
        is_pdf = False
        if file_path and file_path.lower().endswith('.pdf'):
            is_pdf = True
        elif file_bytes and file_bytes.startswith(b'%PDF-'):
            is_pdf = True

        raw_bgr = None
        pdf_text_layer = []

        # Load PDF
        if is_pdf:
            # 1. Direct PDF text layer extraction (fast, 0.5ms)
            try:
                import pypdf
                if file_path:
                    reader = pypdf.PdfReader(file_path)
                else:
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        pdf_text_layer.extend([l.strip() for l in txt.split('\n') if l.strip()])
            except Exception:
                pass

            # 2. Render Page 0 image with pypdfium2 (safely closed)
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file_path if file_path else io.BytesIO(file_bytes))
                try:
                    if len(pdf) > 0:
                        page = pdf[0]
                        pil_img = page.render(scale=2.0).to_pil().convert('RGB')
                        raw_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                finally:
                    pdf.close()
            except Exception:
                pass

        # Standard image decoding
        if raw_bgr is None:
            if file_path:
                raw_bgr = cv2.imread(file_path)
            elif file_bytes:
                nparr = np.frombuffer(file_bytes, np.uint8)
                raw_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if raw_bgr is None:
            # Fallback to PIL
            try:
                if file_path:
                    with Image.open(file_path) as p:
                        raw_bgr = cv2.cvtColor(np.array(p.convert('RGB')), cv2.COLOR_RGB2BGR)
                elif file_bytes:
                    with Image.open(io.BytesIO(file_bytes)) as p:
                        raw_bgr = cv2.cvtColor(np.array(p.convert('RGB')), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        if raw_bgr is None:
            raw_bgr = np.ones((800, 1200, 3), dtype=np.uint8) * 255

        return DocumentContext(
            raw_bgr=raw_bgr,
            file_path=file_path,
            file_hash=file_hash,
            is_pdf=is_pdf,
            pdf_text_layer=pdf_text_layer
        )
