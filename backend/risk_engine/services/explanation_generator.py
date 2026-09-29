"""
VerifyX AI - Explainable Risk Explanation Generator
===================================================
Translates deterministic risk factors and category determinations into
clear, transparent, and auditable natural language explanations.
"""

from typing import List, Dict, Any
from .risk_factors import RiskFactorItem

class ExplanationGenerator:
    """
    Generates plain-language, transparent audit narratives for verification results.
    """

    @staticmethod
    def generate(risk_score: int, category: str, factors: List[RiskFactorItem]) -> str:
        """
        Generate a concise, human-readable explanation of the risk assessment.
        """
        # Filter notable risk contributors (contribution > 0)
        risk_contributors = [f for f in factors if f.contribution > 0]
        critical_factors = [f for f in factors if f.severity in ['CRITICAL', 'HIGH']]

        if category == 'LOW_RISK' or risk_score <= 30:
            if not risk_contributors:
                return "No major risk signals detected. All biometric, forensic, OCR, and identity checks passed within acceptable limits."
            
            top_factors = sorted(risk_contributors, key=lambda x: x.contribution, reverse=True)[:2]
            top_reasons = ", ".join([f"{f.factor_name.lower()} ({f.factor_value})" for f in top_factors])
            return f"Low overall risk (Score: {risk_score}/100). Minor baseline variations noted in {top_reasons}, but overall document authenticity appears high."

        elif category == 'MANUAL_REVIEW' or 31 <= risk_score <= 70:
            if not risk_contributors:
                return f"Manual review recommended (Score: {risk_score}/100). Borderline screening confidence across multiple verification signals."
            
            top_factors = sorted(risk_contributors, key=lambda x: x.contribution, reverse=True)[:3]
            reasons = "; ".join([f"{f.factor_name}: {f.description}" for f in top_factors])
            return f"Manual officer review recommended (Score: {risk_score}/100). Notable screening flags detected: {reasons}"

        else:  # HIGH_RISK (71 - 100)
            if critical_factors:
                top_critical = sorted(critical_factors, key=lambda x: x.contribution, reverse=True)[:3]
                reasons = "; ".join([f"{f.factor_name} ({f.factor_value}): {f.description}" for f in top_critical])
                return f"Elevated anomaly risk flagged (Score: {risk_score}/100). Mandatory inspection required. Critical signals: {reasons}"
            else:
                top_factors = sorted(risk_contributors, key=lambda x: x.contribution, reverse=True)[:3]
                reasons = "; ".join([f"{f.factor_name} ({f.factor_value})" for f in top_factors])
                return f"High risk score ({risk_score}/100) accumulated from multiple compound discrepancies: {reasons}. Further verification required."
