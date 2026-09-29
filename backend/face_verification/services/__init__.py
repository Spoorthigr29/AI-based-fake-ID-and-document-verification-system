"""
Face Verification Services Module for VerifyX AI.
Provides modular face detection, crop extraction, biometric embedding generation,
and explainable comparison against configurable screening thresholds.
"""
from .face_detector import FaceDetector, FaceDetectionResult
from .face_embedder import FaceEmbedder, BaseFaceEmbedder
from .face_service import FaceVerificationService

__all__ = [
    'FaceDetector',
    'FaceDetectionResult',
    'FaceEmbedder',
    'BaseFaceEmbedder',
    'FaceVerificationService',
]
