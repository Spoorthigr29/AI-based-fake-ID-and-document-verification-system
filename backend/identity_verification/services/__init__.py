"""
Identity Consistency Verification Services Module for VerifyX AI.
Provides field normalization, fuzzy string/date/address comparison,
and cross-document consistency assessment.
"""
from .normalization import (
    normalize_name,
    normalize_date,
    normalize_address,
    normalize_gender,
    normalize_doc_number,
)
from .field_comparator import FieldComparator, FieldComparisonResult
from .consistency_engine import IdentityConsistencyEngine

__all__ = [
    'normalize_name',
    'normalize_date',
    'normalize_address',
    'normalize_gender',
    'normalize_doc_number',
    'FieldComparator',
    'FieldComparisonResult',
    'IdentityConsistencyEngine',
]
