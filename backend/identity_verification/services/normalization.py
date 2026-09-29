import re
from datetime import datetime
from typing import Optional

HONORIFIC_PREFIXES = [
    r'\bmr\b\.?', r'\bmrs\b\.?', r'\bms\b\.?', r'\bdr\b\.?',
    r'\bshri\b\.?', r'\bsmt\b\.?', r'\bprof\b\.?', r'\bmaster\b\.?',
    r'\bkumari\b\.?', r'\bmiss\b\.?'
]

ADDRESS_ABBREVIATIONS = {
    r'\bst\b\.?': 'street',
    r'\brd\b\.?': 'road',
    r'\bave\b\.?': 'avenue',
    r'\bapt\b\.?': 'apartment',
    r'\bflr\b\.?': 'floor',
    r'\bblk\b\.?': 'block',
    r'\bdist\b\.?': 'district',
    r'\bp\.?o\.?\b': 'post office',
    r'\bnr\b\.?': 'near',
    r'\bopp\b\.?': 'opposite',
    r'\bsec\b\.?': 'sector',
    r'\bbldg\b\.?': 'building',
    r'\bh\.?no\.?\b': 'house',
    r'\bste\b\.?': 'suite',
    r'\bln\b\.?': 'lane',
    r'\bmarg\b': 'road',
    r'\bdr\b\.?': 'drive',
}

MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
    'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}

def normalize_name(name_str: Optional[str]) -> str:
    """
    Standardize personal names:
    - Case folding (e.g. 'RAHUL KUMAR', 'Rahul Kumar', 'rahul kumar' -> 'rahul kumar')
    - Removal of honorific titles (Mr., Mrs., Dr., Shri, Smt.)
    - Punctuation stripping and whitespace normalization
    """
    if not name_str or not isinstance(name_str, str):
        return ""

    text = name_str.strip().lower()

    # Strip honorific titles
    for prefix_pat in HONORIFIC_PREFIXES:
        text = re.sub(prefix_pat, '', text, flags=re.IGNORECASE)

    # Remove dots between acronym letters (e.g. ph.d -> phd)
    text = re.sub(r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', '', text)

    # Remove all non-alpha characters except spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Collapse multiple consecutive whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """
    Parse varied date formats and normalize to ISO-8601 'YYYY-MM-DD'.
    Supports: DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, DD.MM.YYYY, DD Month YYYY, etc.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    cleaned = date_str.strip()
    # Normalize separators
    cleaned = re.sub(r'[\./]', '-', cleaned)

    # 1. Standard ISO Format: YYYY-MM-DD
    match_iso = re.search(r'\b(19\d{2}|20\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b', cleaned)
    if match_iso:
        y, m, d = match_iso.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    # 2. Standard Indian / European Format: DD-MM-YYYY
    match_dmy = re.search(r'\b(0?[1-9]|[12]\d|3[01])-(0?[1-9]|1[0-2])-(19\d{2}|20\d{2})\b', cleaned)
    if match_dmy:
        d, m, y = match_dmy.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    # 3. Text Month Format: e.g. 15 Aug 1995 or August 15, 1995
    match_text_dmy = re.search(r'\b(0?[1-9]|[12]\d|3[01])\s+([a-zA-Z]+)\s+(19\d{2}|20\d{2})\b', date_str)
    if match_text_dmy:
        d, m_text, y = match_text_dmy.groups()
        m_num = MONTH_MAP.get(m_text.lower())
        if m_num:
            return f"{int(y):04d}-{m_num:02d}-{int(d):02d}"

    match_text_mdy = re.search(r'\b([a-zA-Z]+)\s+(0?[1-9]|[12]\d|3[01]),?\s+(19\d{2}|20\d{2})\b', date_str)
    if match_text_mdy:
        m_text, d, y = match_text_mdy.groups()
        m_num = MONTH_MAP.get(m_text.lower())
        if m_num:
            return f"{int(y):04d}-{m_num:02d}-{int(d):02d}"

    # 4. Fallback: Parse Year only
    match_year = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if match_year:
        return f"{match_year.group(1)}-01-01"

    return None

def normalize_address(address_str: Optional[str]) -> str:
    """
    Standardize physical addresses:
    - Whitespace normalization
    - Punctuation removal
    - Common abbreviation expansion (St. -> street, Rd. -> road, Apt. -> apartment, etc.)
    """
    if not address_str or not isinstance(address_str, str):
        return ""

    text = address_str.strip().lower()

    # Replace punctuation with spaces
    text = re.sub(r'[,;\.:/\\#\-\(\)]', ' ', text)

    # Expand standardized abbreviations
    for abbrev_pat, full_word in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(abbrev_pat, full_word, text, flags=re.IGNORECASE)

    # Clean non-alphanumeric except space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_gender(gender_str: Optional[str]) -> Optional[str]:
    """Standardize gender to 'MALE', 'FEMALE', or 'OTHER'."""
    if not gender_str or not isinstance(gender_str, str):
        return None

    g = gender_str.strip().upper()
    if g in ['M', 'MALE', 'MAN', 'BOY', 'PURUSH']:
        return 'MALE'
    elif g in ['F', 'FEMALE', 'WOMAN', 'GIRL', 'MAHILA', 'STREE']:
        return 'FEMALE'
    elif g in ['O', 'OTHER', 'TRANSGENDER', 'T']:
        return 'OTHER'
    return None

def normalize_doc_number(doc_str: Optional[str]) -> str:
    """Strip spaces, hyphens, and uppercase document tracking numbers."""
    if not doc_str or not isinstance(doc_str, str):
        return ""

    return re.sub(r'[\s\-_/\.]', '', doc_str).upper()
