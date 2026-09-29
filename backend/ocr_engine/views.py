from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from documents.models import Document
from .models import OCRAnalysis, ExtractedField
from .services.ocr_service import OCREngineService
from audit.models import AuditLog

def ocr_results_view(request, verification_id):
    """
    View structured OCR extraction results and forensic image quality metrics.
    If OCR has not yet run for this document, it triggers execution automatically.
    """
    document = get_object_or_404(Document, verification_id=verification_id)

    # Check if OCR analysis exists, otherwise run pipeline
    ocr_record = getattr(document, 'ocr_analysis', None)
    if not ocr_record and document.original_file:
        try:
            result = OCREngineService.process_document(document.original_file.path)
        except Exception:
            try:
                document.original_file.seek(0)
                result = OCREngineService.process_document(document.original_file)
            except Exception as e:
                result = {"fields": {}, "raw_text": "", "overall_confidence": 0.0, "quality_metrics": {}, "deskew_angle": 0.0, "engine_used": "Error"}

        fields = result.get("fields", {})
        ocr_record, _ = OCRAnalysis.objects.update_or_create(
            document=document,
            defaults={
                "raw_text": result.get("raw_text", ""),
                "overall_confidence": result.get("overall_confidence", 0.0),
                "engine_used": result.get("engine_used", "None"),
                "quality_metrics": result.get("quality_metrics", {}),
                "deskew_angle": result.get("deskew_angle", 0.0),
            }
        )

        ExtractedField.objects.filter(document=document).delete()
        for field_name, field_data in fields.items():
            val = field_data.get("value", "") if isinstance(field_data, dict) else str(field_data)
            conf = field_data.get("confidence", 0.0) if isinstance(field_data, dict) else 0.85
            ExtractedField.objects.create(
                document=document,
                field_name=field_name,
                field_value=str(val),
                confidence=float(conf)
            )

    extracted_fields = document.extracted_fields.all()
    fields_dict = {f.field_name: f for f in extracted_fields}

    context = {
        'document': document,
        'ocr_record': ocr_record,
        'extracted_fields': extracted_fields,
        'fields_dict': fields_dict,
    }
    return render(request, 'ocr_engine/results.html', context)
