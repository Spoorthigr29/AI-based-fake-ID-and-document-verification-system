"""
Document Tampering Detection Services for VerifyX AI.
Provides modular Error Level Analysis (ELA), compression artifact inspection,
copy-move and boundary anomaly detection, and explainable tampering probability scoring.
"""
from .ela_analyzer import ELAAnalyzer, ELAResult
from .artifact_analyzer import ArtifactAnalyzer, ArtifactResult
from .region_detector import RegionAnomalyDetector, RegionDetectionResult
from .tamper_analyzer import TamperAnalyzerService, BaseTamperModel, HeuristicTamperModel

__all__ = [
    'ELAAnalyzer',
    'ELAResult',
    'ArtifactAnalyzer',
    'ArtifactResult',
    'RegionAnomalyDetector',
    'RegionDetectionResult',
    'TamperAnalyzerService',
    'BaseTamperModel',
    'HeuristicTamperModel',
]
