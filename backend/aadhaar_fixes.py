"""
VerifyX AI - Aadhaar Fixes Root Forwarder
Exposes standard operations: detect_document_face, decode_aadhaar_qr, compare_qr_with_ocr, compute_risk.
"""

from documents.services.aadhaar_fixes import (
    detect_document_face,
    decode_aadhaar_qr,
    compare_qr_with_ocr,
    compute_risk,
    mask_aadhaar_number,
    sanitize_for_logging
)

__all__ = [
    'detect_document_face',
    'decode_aadhaar_qr',
    'compare_qr_with_ocr',
    'compute_risk',
    'mask_aadhaar_number',
    'sanitize_for_logging'
]
