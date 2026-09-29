"""
VerifyX AI - Centralized Document Photo & Verification Capabilities Configuration
==================================================================================
Defines document-type capabilities, photo presence, selfie requirements, and enabled
forensic screening pipelines across all synthetic and standard identity templates.

Centralized to prevent hardcoding document-specific logic in multiple places.
"""

from typing import Dict, Any, Optional

DOCUMENT_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "sample_aadhaar": {
        "has_document_photo": True,
        "requires_live_selfie": True,
        "requires_selfie": True,
        "enable_face_matching": True,
        "face_matching_enabled": True,
        "display_name": "Sample Aadhaar Card",
        "supported_fields": ["name", "dob", "gender", "aadhaar_number", "address"],
        "id_format_regex": r"^\d{4}\s\d{4}\s\d{4}$",
        "description": "Standard synthetic Aadhaar identity card with embedded photograph and QR block."
    },
    "sample_pan": {
        "has_document_photo": True,
        "requires_live_selfie": True,
        "requires_selfie": True,
        "enable_face_matching": True,
        "face_matching_enabled": True,
        "display_name": "Sample PAN Card (Permanent Account Number)",
        "supported_fields": ["name", "father_name", "dob", "pan_number"],
        "id_format_regex": r"^[A-Z]{5}[0-9]{4}[A-Z]$",
        "description": "Permanent Account Number card template with applicant photograph and signature."
    },
    "sample_passport": {
        "has_document_photo": True,
        "requires_live_selfie": True,
        "requires_selfie": True,
        "enable_face_matching": True,
        "face_matching_enabled": True,
        "display_name": "Sample Passport Data Page",
        "supported_fields": ["name", "passport_number", "dob", "expiry_date", "nationality", "mrz"],
        "id_format_regex": r"^[A-Z][0-9]{7}$",
        "description": "International passport bio-data page with portrait photo and Machine Readable Zone (MRZ)."
    },
    "sample_driving_license": {
        "has_document_photo": True,
        "requires_live_selfie": True,
        "requires_selfie": True,
        "enable_face_matching": True,
        "face_matching_enabled": True,
        "display_name": "Sample Driving License",
        "supported_fields": ["name", "license_number", "dob", "validity", "address"],
        "id_format_regex": r"^[A-Z]{2}[-\s]?\d{2}[-\s]?\d{11}$",
        "description": "State motor vehicle driving license with applicant portrait and vehicle authorizations."
    },
    "sample_voter_id": {
        "has_document_photo": True,
        "requires_live_selfie": True,
        "requires_selfie": True,
        "enable_face_matching": True,
        "face_matching_enabled": True,
        "display_name": "Sample Voter ID (EPIC)",
        "supported_fields": ["name", "voter_id_number", "dob", "gender", "address"],
        "id_format_regex": r"^[A-Z]{3}[0-9]{7}$",
        "description": "Election Commission synthetic voter identity card with applicant portrait."
    },
    "other_synthetic": {
        "has_document_photo": False,
        "requires_live_selfie": True,
        "requires_selfie": True,
        "enable_face_matching": True,
        "face_matching_enabled": True,
        "display_name": "Other Synthetic Identity Document",
        "supported_fields": ["name", "id_number", "dob"],
        "id_format_regex": r"^.*$",
        "description": "Generic document template with live selfie capture and OCR forensic checks."
    }
}


def normalize_document_type(document_type: Optional[str]) -> str:
    """
    Normalize various document type identifiers (e.g. 'AADHAAR', 'sample_pan', 'PAN')
    to standard canonical capability keys.
    """
    if not document_type:
        return "other_synthetic"

    dt = document_type.strip().lower()

    if "aadhaar" in dt or dt in ["aadhaar", "sample_aadhaar", "sample_identity_card"]:
        return "sample_aadhaar"
    elif "pan" in dt or dt in ["pan", "sample_pan", "sample_pan_document"]:
        return "sample_pan"
    elif "passport" in dt or dt in ["passport", "sample_passport"]:
        return "sample_passport"
    elif "driving" in dt or "dl" in dt or dt in ["driving_license", "sample_driving_license"]:
        return "sample_driving_license"
    elif "voter" in dt or "epic" in dt or dt in ["voter_id", "sample_voter_id"]:
        return "sample_voter_id"
    elif dt in DOCUMENT_CAPABILITIES:
        return dt
    else:
        return "other_synthetic"


def get_document_capabilities(document_type: Optional[str]) -> Dict[str, Any]:
    """
    Retrieve the capabilities configuration for a given document type.
    """
    canonical_key = normalize_document_type(document_type)
    return DOCUMENT_CAPABILITIES.get(canonical_key, DOCUMENT_CAPABILITIES["other_synthetic"])


def has_document_photo(document_type: Optional[str]) -> bool:
    """Check if the document template contains a photograph."""
    return get_document_capabilities(document_type).get("has_document_photo", False)


def requires_live_selfie(document_type: Optional[str]) -> bool:
    """Check if the document intake workflow requires an applicant live selfie."""
    return get_document_capabilities(document_type).get("requires_live_selfie", False)


def is_face_matching_enabled(document_type: Optional[str]) -> bool:
    """Check if biometric facial matching is active for this document type."""
    return get_document_capabilities(document_type).get("enable_face_matching", False)


def get_all_document_capabilities() -> Dict[str, Dict[str, Any]]:
    """Return the entire capability configuration mapping."""
    return DOCUMENT_CAPABILITIES
