"""
VerifyX AI - Aadhaar QR Code Verification & Forensic Consistency Engine
========================================================================
Implements Phase 5 & Phase 6 of the Aadhaar Verification Pipeline:
1. QR Code localization and bounding box extraction.
2. Multi-engine decoding (zxing-cpp, OpenCV QRCodeDetector with adaptive pre-processing).
3. Payload format detection:
   - Secure Decompressed V2/V3 Format (UIDAI Compressed Byte Stream / Deflated Stream)
   - Legacy XML V1 Format (<PrintLetterBarcodeData ... />)
   - Structured JSON / Key-Value Synthetic Prototype Format
   - Raw Text Format
4. Cryptographic Signature Status reporting:
   - Explicitly returns 'Official QR signature validation unavailable' unless authorized offline public keys are available.
5. Cross-field consistency verification between QR data and OCR-extracted fields:
   - Name matching (Exact & Fuzzy Token Match)
   - Date of Birth matching
   - Gender matching
   - Aadhaar Number / Last 4 Digits matching
   - Embedded QR Photo vs Document Photo Biometric Matching (where technically present)
"""

import os
import re
import io
import zlib
import json
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class AadhaarQRService:
    """
    Dedicated Aadhaar QR Code Detection, Payload Decoding,
    and Document Consistency Verification Service.
    """

    # Aadhaar legacy XML regex
    RE_XML_BARCODE = re.compile(r'<PrintLetterBarcodeData[^>]+/>', re.IGNORECASE | re.DOTALL)
    RE_12_DIGIT = re.compile(r'\b(\d{12}|\d{4}\s\d{4}\s\d{4})\b')
    RE_4_DIGIT_MASK = re.compile(r'(?:XXXX\s*XXXX\s*|\b)(\d{4})\b')

    @classmethod
    def detect_and_decode_qr(cls, image_np_or_path: Any) -> Dict[str, Any]:
        """
        Locate and decode QR code from document image using multi-pass strategies.
        
        Returns:
            {
                "qr_detected": bool,
                "qr_decoded": bool,
                "raw_text": str,
                "bbox": [x, y, w, h] or None,
                "format_type": str,
                "extracted_data": dict,
                "has_photo": bool,
                "photo_np": np.ndarray or None,
                "signature_status": str,
                "issues": list
            }
        """
        # Load image array
        if isinstance(image_np_or_path, str):
            img = cv2.imread(image_np_or_path)
            if img is None and os.path.exists(image_np_or_path):
                try:
                    pil_img = Image.open(image_np_or_path).convert('RGB')
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
        elif isinstance(image_np_or_path, bytes):
            nparr = np.frombuffer(image_np_or_path, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_np_or_path, np.ndarray):
            img = image_np_or_path
        elif isinstance(image_np_or_path, Image.Image):
            img = cv2.cvtColor(np.array(image_np_or_path), cv2.COLOR_RGB2BGR)
        elif hasattr(image_np_or_path, 'path') and os.path.exists(image_np_or_path.path):
            img = cv2.imread(image_np_or_path.path)
        else:
            return cls._empty_qr_result("Invalid image input format")

        if img is None or img.size == 0:
            return cls._empty_qr_result("Empty or unreadable image array")

        # Multi-pass decoding strategies
        decoded_text = None
        decoded_bytes = None
        bbox = None
        engine_used = None

        # Pass 1: zxing-cpp on original image
        try:
            import zxingcpp
            # zxing-cpp expects BGR or Grayscale
            results = zxingcpp.read_barcodes(img)
            for res in results:
                if res.format in [zxingcpp.BarcodeFormat.QRCode, zxingcpp.BarcodeFormat.MicroQRCode]:
                    decoded_text = res.text
                    decoded_bytes = res.bytes
                    engine_used = "zxing-cpp"
                    if res.position:
                        # Convert points to bounding box
                        pts = [(p.x, p.y) for p in [res.position.top_left, res.position.top_right, res.position.bottom_right, res.position.bottom_left]]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    break
        except Exception as e:
            logger.debug(f"zxing-cpp pass 1 error: {e}")

        # Pass 2: OpenCV QRCodeDetector
        if not decoded_text:
            try:
                detector = cv2.QRCodeDetector()
                val, pts, _ = detector.detectAndDecode(img)
                if val:
                    decoded_text = val
                    engine_used = "OpenCV QRCodeDetector"
                    if pts is not None and len(pts) > 0:
                        pts = pts.reshape(-1, 2)
                        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
                        bbox = [x, y, w, h]
            except Exception as e:
                logger.debug(f"OpenCV QRCodeDetector pass error: {e}")

        # Pass 3: Preprocessed localized crops (Adaptive thresholding & high contrast on quadrants)
        if not decoded_text:
            h, w = img.shape[:2]
            # Try specific candidate regions where Aadhaar QR codes typically reside (top-right, bottom-right, left)
            candidate_regions = [
                img[0:int(h * 0.7), int(w * 0.5):w], # Top-Right (Standard Aadhaar)
                img[int(h * 0.3):h, int(w * 0.5):w], # Bottom-Right
                img[0:int(h * 0.6), 0:int(w * 0.5)], # Top-Left
                img # Full image with contrast enhancement
            ]

            for region in candidate_regions:
                if region.size == 0:
                    continue
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                try:
                    import zxingcpp
                    res = zxingcpp.read_barcode(enhanced)
                    if res and res.text:
                        decoded_text = res.text
                        decoded_bytes = res.bytes
                        engine_used = "zxing-cpp (Enhanced Region)"
                        break
                except Exception:
                    pass

        if not decoded_text:
            return {
                "qr_detected": False,
                "qr_decoded": False,
                "raw_text": "",
                "bbox": None,
                "format_type": "NONE",
                "extracted_data": {},
                "has_photo": False,
                "photo_np": None,
                "signature_status": "No QR Code Detected",
                "engine_used": None,
                "issues": ["No legible QR code found on the document."]
            }

        # Parse the decoded QR content
        parsed = cls.parse_aadhaar_payload(decoded_text, raw_bytes=decoded_bytes)
        
        return {
            "qr_detected": True,
            "qr_decoded": True,
            "raw_text": decoded_text,
            "bbox": bbox,
            "format_type": parsed.get("format_type", "UNKNOWN"),
            "extracted_data": parsed.get("data", {}),
            "has_photo": parsed.get("has_photo", False),
            "photo_np": parsed.get("photo_np"),
            "signature_status": parsed.get("signature_status", "Official QR signature validation unavailable"),
            "signature_present": parsed.get("signature_present", False),
            "engine_used": engine_used,
            "issues": parsed.get("issues", [])
        }

    @classmethod
    def parse_aadhaar_payload(cls, raw_text: str, raw_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Parse Aadhaar payload from string or decompressed byte stream into structured fields.
        """
        issues = []
        data = {}
        format_type = "UNKNOWN"
        has_photo = False
        photo_np = None
        signature_present = False

        # Strategy 1: Aadhaar Legacy XML Format
        if "<PrintLetterBarcodeData" in raw_text or "<printletterbarcodedata" in raw_text.lower():
            format_type = "AADHAAR_XML_LEGACY"
            try:
                # Extract XML tag even if wrapped with extraneous text
                xml_match = cls.RE_XML_BARCODE.search(raw_text)
                xml_str = xml_match.group(0) if xml_match else raw_text
                root = ET.fromstring(xml_str)
                
                # Attribute extraction
                attr = root.attrib
                uid_raw = attr.get('uid', '') or attr.get('UID', '')
                name = attr.get('name', '') or attr.get('NAME', '')
                gender = attr.get('gender', '') or attr.get('GENDER', '')
                yob = attr.get('yob', '') or attr.get('YOB', '')
                dob = attr.get('dob', '') or attr.get('DOB', '')
                
                # Build address components
                addr_parts = [
                    attr.get('house', ''), attr.get('street', ''), attr.get('lm', ''),
                    attr.get('loc', ''), attr.get('vtc', ''), attr.get('po', ''),
                    attr.get('subdist', ''), attr.get('dist', ''), attr.get('state', ''),
                    attr.get('pc', '')
                ]
                full_address = ", ".join([p.strip() for p in addr_parts if p and p.strip()])

                data = {
                    "uid": cls.mask_aadhaar(uid_raw),
                    "uid_raw_last4": uid_raw[-4:] if len(uid_raw) >= 4 else uid_raw,
                    "name": name.strip(),
                    "dob": dob.strip() if dob else yob.strip(),
                    "gender": "FEMALE" if gender.upper().startswith("F") else ("MALE" if gender.upper().startswith("M") else gender.upper()),
                    "address": full_address,
                    "pincode": attr.get('pc', ''),
                    "state": attr.get('state', ''),
                    "district": attr.get('dist', ''),
                    "care_of": attr.get('co', '')
                }
            except Exception as e:
                issues.append(f"XML parse warning: {e}")

        # Strategy 2: JSON Structured Payload (Modern / Synthetic prototype)
        elif raw_text.strip().startswith("{") and raw_text.strip().endswith("}"):
            format_type = "AADHAAR_JSON_STRUCTURED"
            try:
                parsed_json = json.loads(raw_text)
                uid_val = str(parsed_json.get("uid") or parsed_json.get("aadhaar_number") or parsed_json.get("document_number") or "")
                data = {
                    "uid": cls.mask_aadhaar(uid_val),
                    "uid_raw_last4": uid_val[-4:] if len(uid_val) >= 4 else uid_val,
                    "name": str(parsed_json.get("name") or "").strip(),
                    "dob": str(parsed_json.get("dob") or parsed_json.get("date_of_birth") or parsed_json.get("yob") or "").strip(),
                    "gender": str(parsed_json.get("gender") or "").strip().upper(),
                    "address": str(parsed_json.get("address") or "").strip(),
                    "pincode": str(parsed_json.get("pincode") or parsed_json.get("zip") or "").strip(),
                    "state": str(parsed_json.get("state") or "").strip(),
                    "district": str(parsed_json.get("district") or "").strip()
                }
            except Exception as e:
                issues.append(f"JSON parse error: {e}")

        # Strategy 3: Modern UIDAI Compressed Big-Integer / Deflated Byte Stream (V2 / V3)
        elif raw_bytes or (raw_text.isdigit() and len(raw_text) > 200):
            format_type = "AADHAAR_SECURE_V2_V3"
            try:
                # Attempt big-integer decompression
                if raw_text.isdigit():
                    num = int(raw_text)
                    # Convert to byte array
                    byte_len = (num.bit_length() + 7) // 8
                    byte_data = num.to_bytes(byte_len, byteorder='big')
                else:
                    byte_data = raw_bytes or raw_text.encode('latin1')

                # Decompress with zlib
                decompressed = None
                for wbits in [zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 16]:
                    try:
                        decompressed = zlib.decompress(byte_data, wbits)
                        if decompressed:
                            break
                    except Exception:
                        continue

                if decompressed:
                    signature_present = True
                    # Parse delimiter-separated fields (255 delimiter in V2 format)
                    parts = decompressed.split(b'\xff')
                    if len(parts) >= 15:
                        data = {
                            "reference_id": parts[0].decode('latin1', errors='ignore'),
                            "name": parts[1].decode('utf-8', errors='ignore').strip(),
                            "dob": parts[2].decode('utf-8', errors='ignore').strip(),
                            "gender": parts[3].decode('utf-8', errors='ignore').strip(),
                            "care_of": parts[4].decode('utf-8', errors='ignore').strip(),
                            "district": parts[5].decode('utf-8', errors='ignore').strip(),
                            "landmark": parts[6].decode('utf-8', errors='ignore').strip(),
                            "house": parts[7].decode('utf-8', errors='ignore').strip(),
                            "location": parts[8].decode('utf-8', errors='ignore').strip(),
                            "pincode": parts[9].decode('utf-8', errors='ignore').strip(),
                            "post_office": parts[10].decode('utf-8', errors='ignore').strip(),
                            "state": parts[11].decode('utf-8', errors='ignore').strip(),
                            "street": parts[12].decode('utf-8', errors='ignore').strip(),
                            "sub_district": parts[13].decode('utf-8', errors='ignore').strip(),
                            "vtc": parts[14].decode('utf-8', errors='ignore').strip(),
                        }
                        # Check for embedded JPEG photo in subsequent byte slices
                        for p in parts[15:]:
                            if len(p) > 200 and p.startswith(b'\xff\xd8'):
                                has_photo = True
                                try:
                                    nparr = np.frombuffer(p, np.uint8)
                                    photo_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                except Exception:
                                    pass
                                break
                    else:
                        issues.append("Secure QR payload decompressed but field count atypical.")
                else:
                    format_type = "RAW_NUMERIC_STREAM"
                    issues.append("Uncompressed raw numeric QR stream detected.")
            except Exception as e:
                issues.append(f"Secure QR decoding fallback: {e}")

        # Strategy 4: Raw Text Key-Value Parsing Fallback
        else:
            format_type = "RAW_TEXT_BARCODE"
            # Extract any UID or name patterns
            lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
            for line in lines:
                if "name" in line.lower() and ":" in line:
                    data["name"] = line.split(":", 1)[-1].strip()
                if "dob" in line.lower() and ":" in line:
                    data["dob"] = line.split(":", 1)[-1].strip()
                if "gender" in line.lower() and ":" in line:
                    data["gender"] = line.split(":", 1)[-1].strip().upper()
                if "uid" in line.lower() and ":" in line:
                    u = line.split(":", 1)[-1].strip()
                    data["uid"] = cls.mask_aadhaar(u)
                    data["uid_raw_last4"] = u[-4:]

        return {
            "format_type": format_type,
            "data": data,
            "has_photo": has_photo,
            "photo_np": photo_np,
            "signature_present": signature_present,
            "signature_status": "Official QR signature validation unavailable (Screening Prototype)",
            "issues": issues
        }

    @classmethod
    def verify_qr_document_consistency(
        cls,
        qr_data: Dict[str, Any],
        ocr_fields: Dict[str, Any],
        doc_face_np: Optional[np.ndarray] = None,
        qr_photo_np: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Phase 6: Compare verified QR demographic & photo data against visible OCR document data.
        Exposes document cloning, name alteration, and photo tampering scenarios.
        
        Returns:
            {
                "consistency_status": "CONSISTENT" | "INCONSISTENT" | "NOT_AVAILABLE",
                "consistency_score": int (0 to 100),
                "matched_fields": list,
                "mismatched_fields": list,
                "reasons": list,
                "photo_match_status": str,
                "photo_similarity": float or None
            }
        """
        if not qr_data or not isinstance(qr_data, dict) or len(qr_data) == 0:
            return {
                "consistency_status": "NOT_AVAILABLE",
                "consistency_score": 100,
                "matched_fields": [],
                "mismatched_fields": [],
                "reasons": ["No readable QR code data available for cross-comparison."],
                "photo_match_status": "NOT_APPLICABLE",
                "photo_similarity": None
            }

        matched_fields = []
        mismatched_fields = []
        reasons = []
        points = 100

        # 1. Compare Name
        qr_name = cls._clean_text(qr_data.get("name", ""))
        ocr_name = cls._clean_text(cls._get_field_val(ocr_fields, "name"))
        
        if qr_name and ocr_name:
            if qr_name == ocr_name or qr_name in ocr_name or ocr_name in qr_name:
                matched_fields.append("Name")
                reasons.append("✓ Name in QR matches visible document OCR text.")
            else:
                # Token overlap check
                qr_tokens = set(qr_name.split())
                ocr_tokens = set(ocr_name.split())
                if qr_tokens.intersection(ocr_tokens):
                    matched_fields.append("Name (Partial Token)")
                    reasons.append("✓ Name in QR closely matches document OCR text.")
                else:
                    mismatched_fields.append("Name")
                    points -= 40
                    reasons.append(f"✕ DATA INCONSISTENCY: Document name '{ocr_name}' differs from QR record '{qr_name}'.")

        # 2. Compare Date of Birth
        qr_dob = cls._clean_date(qr_data.get("dob", ""))
        ocr_dob = cls._clean_date(cls._get_field_val(ocr_fields, "dob") or cls._get_field_val(ocr_fields, "date_of_birth"))
        
        if qr_dob and ocr_dob:
            if qr_dob == ocr_dob or (len(qr_dob) == 4 and qr_dob in ocr_dob) or (len(ocr_dob) == 4 and ocr_dob in qr_dob):
                matched_fields.append("DOB")
                reasons.append("✓ DOB in QR matches visible document.")
            else:
                mismatched_fields.append("DOB")
                points -= 25
                reasons.append(f"✕ DATA INCONSISTENCY: Document DOB '{ocr_dob}' differs from QR record '{qr_dob}'.")

        # 3. Compare Gender
        qr_gender = (qr_data.get("gender") or "").upper().strip()
        ocr_gender = (cls._get_field_val(ocr_fields, "gender") or "").upper().strip()
        
        if qr_gender and ocr_gender:
            if qr_gender[0] == ocr_gender[0]:
                matched_fields.append("Gender")
                reasons.append("✓ Gender matches QR record.")
            else:
                mismatched_fields.append("Gender")
                points -= 15
                reasons.append(f"✕ Gender mismatch: Document indicates '{ocr_gender}' but QR records '{qr_gender}'.")

        # 4. Compare Aadhaar Number / Last 4 Digits
        qr_uid = str(qr_data.get("uid_raw_last4") or qr_data.get("uid") or "")[-4:]
        ocr_uid_raw = cls._get_field_val(ocr_fields, "document_number") or cls._get_field_val(ocr_fields, "aadhaar_number") or ""
        ocr_uid = ocr_uid_raw.replace(" ", "").replace("-", "")[-4:] if ocr_uid_raw else ""

        if qr_uid and ocr_uid and len(qr_uid) == 4 and len(ocr_uid) == 4:
            if qr_uid == ocr_uid:
                matched_fields.append("Aadhaar Number (Last 4 Digits)")
                reasons.append("✓ Aadhaar UID sequence matches QR record.")
            else:
                mismatched_fields.append("Aadhaar Number")
                points -= 45
                reasons.append(f"✕ CRITICAL INCONSISTENCY: Printed UID sequence (..{ocr_uid}) contradicts QR digital record (..{qr_uid}).")

        # 5. Compare QR Embedded Photo vs Document Printed Photo (Cloning & Photo-Replacement Detection)
        photo_match_status = "NOT_APPLICABLE"
        photo_similarity = None

        if qr_photo_np is not None and doc_face_np is not None:
            try:
                from face_verification.services.face_comparator import FaceComparator
                comparator = FaceComparator()
                sim, status = comparator.compare(doc_face_np, qr_photo_np)
                photo_similarity = round(float(sim), 3)
                if status == 'MATCH':
                    photo_match_status = "MATCH"
                    matched_fields.append("Embedded QR Photo")
                    reasons.append("✓ Printed document photograph biometrically matches QR digital photo record.")
                else:
                    photo_match_status = "MISMATCH"
                    mismatched_fields.append("Embedded QR Photo")
                    points -= 50
                    reasons.append(f"✕ PHOTO REPLACEMENT DETECTED: Printed photograph biometrically contradicts the digital photo embedded in the secure QR ({int(sim*100)}% match).")
            except Exception as e:
                logger.debug(f"QR photo comparison error: {e}")

        # Final consistency scoring
        final_score = max(0, min(100, points))
        if mismatched_fields:
            status_str = "INCONSISTENT"
        else:
            status_str = "CONSISTENT"

        return {
            "consistency_status": status_str,
            "consistency_score": final_score,
            "matched_fields": matched_fields,
            "mismatched_fields": mismatched_fields,
            "reasons": reasons,
            "photo_match_status": photo_match_status,
            "photo_similarity": photo_similarity
        }

    @staticmethod
    def mask_aadhaar(uid_str: str) -> str:
        """Mask Aadhaar number to XXXX XXXX 1234 per privacy guidelines."""
        if not uid_str:
            return ""
        clean = re.sub(r'[^0-9]', '', uid_str)
        if len(clean) == 12:
            return f"XXXX XXXX {clean[-4:]}"
        elif len(clean) >= 4:
            return f"XXXX-{clean[-4:]}"
        return "XXXX"

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r'[^A-Z0-9\s]', '', text.upper()).strip()

    @staticmethod
    def _clean_date(date_str: str) -> str:
        if not date_str:
            return ""
        date_clean = date_str.strip().replace('/', '-')
        for fmt in ('%d-%m-%Y', '%d-%m-%y', '%Y-%m-%d', '%m-%d-%Y', '%Y'):
            try:
                from datetime import datetime
                dt = datetime.strptime(date_clean, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        digits = re.sub(r'[^0-9]', '', date_str)
        return digits

    @staticmethod
    def _get_field_val(fields: Dict[str, Any], key: str) -> str:
        if not fields or not isinstance(fields, dict):
            return ""
        val = fields.get(key, "")
        if isinstance(val, dict):
            return str(val.get("value", ""))
        return str(val)

    @classmethod
    def _empty_qr_result(cls, reason: str) -> Dict[str, Any]:
        return {
            "qr_detected": False,
            "qr_decoded": False,
            "raw_text": "",
            "bbox": None,
            "format_type": "NONE",
            "extracted_data": {},
            "has_photo": False,
            "photo_np": None,
            "signature_status": "No QR Code Detected",
            "engine_used": None,
            "issues": [reason]
        }
