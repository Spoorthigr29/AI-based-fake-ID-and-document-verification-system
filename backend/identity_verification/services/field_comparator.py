import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from .normalization import (
    normalize_name,
    normalize_date,
    normalize_address,
    normalize_gender,
    normalize_doc_number,
)

@dataclass
class FieldComparisonResult:
    is_match: bool
    similarity_score: float  # 0.0 to 1.0
    field_name: str
    val_a: Optional[str]
    val_b: Optional[str]
    normalized_a: Optional[str]
    normalized_b: Optional[str]
    note: str

class FieldComparator:
    """
    Algorithmic and Fuzzy Field Comparison Engine for VerifyX AI.
    Executes cross-document field matching with explainable confidence scoring.
    """

    @staticmethod
    def compute_levenshtein_ratio(s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein string similarity ratio (0.0 to 1.0)."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + cost # Substitution
                )

        max_len = max(m, n)
        return max(0.0, 1.0 - (dp[m][n] / max_len)) if max_len > 0 else 1.0

    @classmethod
    def compare_names(cls, name_a: Optional[str], name_b: Optional[str], threshold: float = 0.82) -> FieldComparisonResult:
        """
        Compare personal names with fuzzy matching, token permutation invariance, and case folding.
        """
        norm_a = normalize_name(name_a)
        norm_b = normalize_name(name_b)

        if not norm_a or not norm_b:
            return FieldComparisonResult(
                is_match=False,
                similarity_score=0.0,
                field_name="name",
                val_a=name_a,
                val_b=name_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Name missing in one or both sources"
            )

        # 1. Exact Normalized Match
        if norm_a == norm_b:
            return FieldComparisonResult(
                is_match=True,
                similarity_score=1.0,
                field_name="name",
                val_a=name_a,
                val_b=name_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Exact normalized name match"
            )

        # 2. Token Set Permutation Match (e.g. 'Rahul Kumar' vs 'Kumar Rahul')
        tokens_a = sorted(norm_a.split())
        tokens_b = sorted(norm_b.split())
        if tokens_a == tokens_b:
            return FieldComparisonResult(
                is_match=True,
                similarity_score=0.96,
                field_name="name",
                val_a=name_a,
                val_b=name_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Name token order permutation match"
            )

        # 3. Levenshtein Fuzzy Ratio
        ratio = cls.compute_levenshtein_ratio(norm_a, norm_b)
        
        # Token sort ratio
        sorted_a = ' '.join(tokens_a)
        sorted_b = ' '.join(tokens_b)
        sorted_ratio = cls.compute_levenshtein_ratio(sorted_a, sorted_b)
        best_score = max(ratio, sorted_ratio)

        is_match = best_score >= threshold
        note = f"Fuzzy name match (score: {best_score:.2f})" if is_match else f"Name discrepancy (score: {best_score:.2f})"

        return FieldComparisonResult(
            is_match=is_match,
            similarity_score=round(best_score, 2),
            field_name="name",
            val_a=name_a,
            val_b=name_b,
            normalized_a=norm_a,
            normalized_b=norm_b,
            note=note
        )

    @classmethod
    def compare_dobs(cls, dob_a: Optional[str], dob_b: Optional[str]) -> FieldComparisonResult:
        """
        Compare Dates of Birth normalized to standard ISO YYYY-MM-DD.
        """
        norm_a = normalize_date(dob_a)
        norm_b = normalize_date(dob_b)

        if not norm_a or not norm_b:
            return FieldComparisonResult(
                is_match=False,
                similarity_score=0.0,
                field_name="dob",
                val_a=dob_a,
                val_b=dob_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Date of birth could not be parsed in one or both records"
            )

        if norm_a == norm_b:
            return FieldComparisonResult(
                is_match=True,
                similarity_score=1.0,
                field_name="dob",
                val_a=dob_a,
                val_b=dob_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Exact Date of Birth match"
            )

        # Partial match analysis (Year and Month match)
        parts_a = norm_a.split('-')
        parts_b = norm_b.split('-')

        if parts_a[0] == parts_b[0] and parts_a[1] == parts_b[1]:
            return FieldComparisonResult(
                is_match=False,
                similarity_score=0.65,
                field_name="dob",
                val_a=dob_a,
                val_b=dob_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note=f"Day mismatch in DOB ({norm_a} vs {norm_b})"
            )
        elif parts_a[0] == parts_b[0]:
            return FieldComparisonResult(
                is_match=False,
                similarity_score=0.40,
                field_name="dob",
                val_a=dob_a,
                val_b=dob_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note=f"Year matched but month/day mismatch ({norm_a} vs {norm_b})"
            )

        return FieldComparisonResult(
            is_match=False,
            similarity_score=0.0,
            field_name="dob",
            val_a=dob_a,
            val_b=dob_b,
            normalized_a=norm_a,
            normalized_b=norm_b,
            note=f"DOB mismatch ({norm_a} vs {norm_b})"
        )

    @classmethod
    def compare_document_numbers(cls, doc_a: Optional[str], doc_b: Optional[str]) -> FieldComparisonResult:
        """
        Compare Document Identification Numbers with digit normalization and OCR confusion tolerance.
        """
        norm_a = normalize_doc_number(doc_a)
        norm_b = normalize_doc_number(doc_b)

        if not norm_a or not norm_b:
            return FieldComparisonResult(
                is_match=False,
                similarity_score=0.0,
                field_name="document_number",
                val_a=doc_a,
                val_b=doc_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Document ID number missing in one or both records"
            )

        if norm_a == norm_b:
            return FieldComparisonResult(
                is_match=True,
                similarity_score=1.0,
                field_name="document_number",
                val_a=doc_a,
                val_b=doc_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Exact Document Number match"
            )

        # Check Levenshtein ratio with common OCR character substitutions (0/O, 1/I/L, 5/S, 8/B)
        sub_a = norm_a.replace('O', '0').replace('I', '1').replace('L', '1').replace('S', '5').replace('B', '8')
        sub_b = norm_b.replace('O', '0').replace('I', '1').replace('L', '1').replace('S', '5').replace('B', '8')

        if sub_a == sub_b:
            return FieldComparisonResult(
                is_match=True,
                similarity_score=0.92,
                field_name="document_number",
                val_a=doc_a,
                val_b=doc_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Document number matched with typical OCR character substitution"
            )

        ratio = cls.compute_levenshtein_ratio(norm_a, norm_b)
        is_match = ratio >= 0.88
        note = "Document Number format valid" if is_match else f"Document Number mismatch ({norm_a} vs {norm_b})"

        return FieldComparisonResult(
            is_match=is_match,
            similarity_score=round(ratio, 2),
            field_name="document_number",
            val_a=doc_a,
            val_b=doc_b,
            normalized_a=norm_a,
            normalized_b=norm_b,
            note=note
        )

    @classmethod
    def compare_addresses(cls, addr_a: Optional[str], addr_b: Optional[str], threshold: float = 0.65) -> FieldComparisonResult:
        """
        Compare physical addresses using token-set Jaccard overlap and PIN code matching.
        """
        norm_a = normalize_address(addr_a)
        norm_b = normalize_address(addr_b)

        if not norm_a or not norm_b:
            return FieldComparisonResult(
                is_match=False,
                similarity_score=0.0,
                field_name="address",
                val_a=addr_a,
                val_b=addr_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Address missing in one or both records"
            )

        if norm_a == norm_b:
            return FieldComparisonResult(
                is_match=True,
                similarity_score=1.0,
                field_name="address",
                val_a=addr_a,
                val_b=addr_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Exact normalized address match"
            )

        tokens_a = set(norm_a.split())
        tokens_b = set(norm_b.split())

        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)

        jaccard = len(intersection) / len(union) if union else 0.0
        overlap = len(intersection) / min(len(tokens_a), len(tokens_b)) if tokens_a and tokens_b else 0.0

        # Extract 6-digit Indian PIN codes if present
        pin_a = re.findall(r'\b\d{6}\b', addr_a or '')
        pin_b = re.findall(r'\b\d{6}\b', addr_b or '')
        pin_matched = False
        if pin_a and pin_b:
            pin_matched = (pin_a[0] == pin_b[0])

        score = (jaccard * 0.5) + (overlap * 0.5)
        if pin_matched:
            score = min(1.0, score + 0.15)

        is_match = score >= threshold
        note = "Address matched" if is_match else f"Address mismatch (similarity: {score:.2f})"

        return FieldComparisonResult(
            is_match=is_match,
            similarity_score=round(score, 2),
            field_name="address",
            val_a=addr_a,
            val_b=addr_b,
            normalized_a=norm_a,
            normalized_b=norm_b,
            note=note
        )

    @classmethod
    def compare_genders(cls, gen_a: Optional[str], gen_b: Optional[str]) -> FieldComparisonResult:
        """Compare gender values."""
        norm_a = normalize_gender(gen_a)
        norm_b = normalize_gender(gen_b)

        if not norm_a or not norm_b:
            return FieldComparisonResult(
                is_match=True,
                similarity_score=1.0,
                field_name="gender",
                val_a=gen_a,
                val_b=gen_b,
                normalized_a=norm_a,
                normalized_b=norm_b,
                note="Gender not specified in one or both records"
            )

        is_match = (norm_a == norm_b)
        return FieldComparisonResult(
            is_match=is_match,
            similarity_score=1.0 if is_match else 0.0,
            field_name="gender",
            val_a=gen_a,
            val_b=gen_b,
            normalized_a=norm_a,
            normalized_b=norm_b,
            note="Gender matched" if is_match else f"Gender mismatch ({norm_a} vs {norm_b})"
        )
