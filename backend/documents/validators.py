import io
import os
from PIL import Image
from django.core.exceptions import ValidationError
import pypdf

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_DOC_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.pdf'}
ALLOWED_SELFIE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/webp',
    'application/pdf',
    'application/x-pdf',
    'application/octet-stream',
}

ALLOWED_SELFIE_MIME_TYPES = {
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/webp',
    'application/octet-stream',
}

# Magic byte signatures
MAGIC_BYTES = {
    'jpeg': b'\xff\xd8\xff',
    'png': b'\x89PNG\r\n\x1a\n',
    'pdf': b'%PDF-',
}

def is_valid_webp_header(header: bytes) -> bool:
    """Validate standard WebP container header (RIFF....WEBP)."""
    return len(header) >= 12 and header.startswith(b'RIFF') and header[8:12] == b'WEBP'

def validate_uploaded_document(file_obj):
    """
    Strict multi-layer validation for identity documents (PDF, JPG, JPEG, PNG, WEBP).
    Validates file presence, size, extension, MIME type, magic bytes, and file integrity.
    """
    if not file_obj:
        raise ValidationError("No document file was uploaded.")

    # 1. Check size limit (10 MB maximum)
    if file_obj.size == 0:
        raise ValidationError("The uploaded document file is empty (0 bytes).")

    if file_obj.size > MAX_FILE_SIZE_BYTES:
        raise ValidationError("File size exceeds the 10 MB limit. Please upload a smaller document.")

    # 2. Check extension
    filename = getattr(file_obj, 'name', '') or ''
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise ValidationError("Unsupported file format. Please upload PDF, JPG, JPEG, PNG or WEBP.")

    # 3. Check MIME type if available
    content_type = getattr(file_obj, 'content_type', None)
    if content_type:
        mime = content_type.lower().split(';')[0].strip()
        if mime not in ALLOWED_MIME_TYPES:
            raise ValidationError("Unsupported file format. Please upload PDF, JPG, JPEG, PNG or WEBP.")

    # 4. Read initial chunk for magic bytes inspection
    pos = file_obj.tell() if hasattr(file_obj, 'tell') else 0
    file_obj.seek(0)
    header = file_obj.read(16)
    file_obj.seek(pos)

    is_jpeg = header.startswith(MAGIC_BYTES['jpeg'])
    is_png = header.startswith(MAGIC_BYTES['png'])
    is_pdf = header.startswith(MAGIC_BYTES['pdf'])
    is_webp = is_valid_webp_header(header)

    if not (is_jpeg or is_png or is_pdf or is_webp):
        raise ValidationError("Unsupported file format. Please upload PDF, JPG, JPEG, PNG or WEBP.")

    # 5. Check extension matches detected signature
    if (ext in ['.jpg', '.jpeg'] and not is_jpeg) or \
       (ext == '.png' and not is_png) or \
       (ext == '.webp' and not is_webp) or \
       (ext == '.pdf' and not is_pdf):
        raise ValidationError(f"File extension '{ext}' does not match the actual file binary format.")

    # 6. Content integrity verification
    file_obj.seek(0)
    file_bytes = file_obj.read()
    file_obj.seek(pos)

    if is_jpeg or is_png or is_webp:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            img = Image.open(io.BytesIO(file_bytes))
            img.load()
        except Exception as e:
            raise ValidationError(f"Corrupted image: The uploaded image file could not be decoded ({str(e)}).")
    elif is_pdf:
        try:
            if b'%PDF-' not in file_bytes[:1024]:
                raise ValidationError("Invalid PDF header.")
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                raise ValidationError("Password-protected PDFs are not supported. Please upload an unencrypted document.")
            if len(reader.pages) == 0:
                raise ValidationError("The uploaded PDF has no pages.")
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Corrupted or unreadable PDF document: {str(e)}")

    return True

def validate_uploaded_selfie(file_obj):
    """
    Strict validation for applicant selfie image (JPG, JPEG, PNG, WEBP).
    Validates file presence, size, extension, MIME type, magic bytes, and image integrity with PIL.
    """
    if not file_obj:
        raise ValidationError("No selfie file was uploaded.")

    # 1. Check size limit
    if file_obj.size == 0:
        raise ValidationError("The uploaded selfie file is empty (0 bytes).")

    if file_obj.size > MAX_FILE_SIZE_BYTES:
        raise ValidationError("File size exceeds the 10 MB limit. Please upload a smaller document.")

    # 2. Check extension
    filename = getattr(file_obj, 'name', '') or ''
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_SELFIE_EXTENSIONS:
        raise ValidationError("Unsupported file format. Please upload PDF, JPG, JPEG, PNG or WEBP.")

    # 3. Check MIME type if available
    content_type = getattr(file_obj, 'content_type', None)
    if content_type:
        mime = content_type.lower().split(';')[0].strip()
        if mime not in ALLOWED_SELFIE_MIME_TYPES:
            raise ValidationError("Unsupported file format. Please upload PDF, JPG, JPEG, PNG or WEBP.")

    # 4. Magic bytes
    pos = file_obj.tell() if hasattr(file_obj, 'tell') else 0
    file_obj.seek(0)
    header = file_obj.read(16)
    file_obj.seek(pos)

    is_jpeg = header.startswith(MAGIC_BYTES['jpeg'])
    is_png = header.startswith(MAGIC_BYTES['png'])
    is_webp = is_valid_webp_header(header)

    if not (is_jpeg or is_png or is_webp):
        raise ValidationError("Unsupported file format. Please upload PDF, JPG, JPEG, PNG or WEBP.")

    # 5. Content integrity verification with Pillow
    try:
        file_obj.seek(0)
        file_bytes = file_obj.read()
        file_obj.seek(pos)
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
    except Exception as e:
        raise ValidationError(f"Corrupted image: The uploaded selfie image could not be decoded ({str(e)}).")

    return True
