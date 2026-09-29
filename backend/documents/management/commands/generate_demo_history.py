"""
VerifyX AI - Generate Demo History Command
==========================================
Generates clearly labelled synthetic DEMO/TEST screening records for
demonstration, testing, pagination, and date-range filter inspection.

IMPORTANT:
- Marks each generated record with is_demo=True.
- Does not modify or use ML training data or real user records.
- Does not touch or alter trained models.
"""

import io
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw

from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.utils import timezone

from documents.models import Document, DocumentQualityAnalysis
from identity_verification.models import VerificationResult


def create_demo_specimen_image(doc_type: str, demo_id: str, status: str, risk: int) -> bytes:
    """Generate a lightweight synthetic specimen JPEG image for demo storage."""
    img = Image.new('RGB', (640, 400), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    # Frame border
    draw.rounded_rectangle([15, 15, 625, 385], radius=12, outline=(148, 163, 184), width=3)
    # Header box
    draw.rectangle([18, 18, 622, 70], fill=(224, 242, 254))
    draw.text((30, 28), f"VERIFYX AI - DEMO / TEST SPECIMEN ({demo_id})", fill=(3, 105, 161))

    # Content
    draw.text((35, 90), f"DOCUMENT TYPE: {doc_type}", fill=(15, 23, 42))
    draw.text((35, 130), f"STATUS: {status} | RISK SCORE: {risk}/100", fill=(51, 65, 85))
    draw.text((35, 170), f"GENERATED AT: {timezone.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", fill=(100, 116, 139))
    draw.text((35, 220), "NOTICE: Synthetic testing record created strictly for UI & audit demonstration.", fill=(148, 163, 184))
    draw.text((35, 250), "DO NOT USE OR REPRESENT AS OFFICIAL GOVERNMENT CREDENTIALS.", fill=(225, 29, 72))

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


class Command(BaseCommand):
    help = "Generate synthetic demo/test screening records for history page demonstration"

    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=30,
            help='Number of demo screening records to create (default: 30)'
        )

    def handle(self, *args, **options):
        count = options['count']
        if count <= 0:
            self.stderr.write(self.style.ERROR("Count must be greater than 0."))
            return

        self.stdout.write(f"Generating {count} synthetic DEMO screening records...")

        # Find starting sequence number for demo IDs (e.g. DEMO-VX-0001)
        existing_demo_docs = Document.objects.filter(is_demo=True, verification_id__startswith='DEMO-VX-')
        max_seq = 0
        for d in existing_demo_docs:
            try:
                seq = int(d.verification_id.split('-')[-1])
                if seq > max_seq:
                    max_seq = seq
            except (ValueError, IndexError):
                pass

        # Preset distributions for realistic diverse demonstration
        doc_types = [
            'AADHAAR', 'AADHAAR', 'AADHAAR', 'AADHAAR',
            'PAN', 'PAN', 'PAN',
            'PASSPORT', 'DRIVING_LICENSE'
        ]

        # Outcomes: 50% Verified, 30% Review Required, 20% Mismatch
        outcomes = [
            ('COMPLETED', 'LOW_RISK', [4, 8, 12, 16, 18, 21, 24, 27]),
            ('COMPLETED', 'LOW_RISK', [5, 9, 14, 17, 19, 22, 25, 28]),
            ('COMPLETED', 'LOW_RISK', [6, 10, 15, 18, 20, 23, 26, 29]),
            ('MANUAL_REVIEW', 'SUSPICIOUS_MANUAL_REVIEW', [32, 36, 40, 44, 47, 51, 55, 58]),
            ('MANUAL_REVIEW', 'SUSPICIOUS_MANUAL_REVIEW', [34, 38, 42, 45, 48, 52, 56, 59]),
            ('FAILED', 'HIGH_RISK', [64, 68, 72, 76, 81, 85, 90, 94]),
        ]

        explanations = {
            'LOW_RISK': "Demo Specimen passed automated multi-signal screening with low risk. Document integrity intact, OCR text validated, biometric similarity passed.",
            'SUSPICIOUS_MANUAL_REVIEW': "Demo Specimen flagged for secondary officer review. Moderate text uncertainty detected or non-critical compression anomaly noted.",
            'HIGH_RISK': "Demo Specimen rejected due to critical anomaly. Discrepant identity fields or suspicious manipulation signatures detected during screening."
        }

        now = timezone.now()
        created_records = []

        for i in range(1, count + 1):
            seq_num = max_seq + i
            demo_id = f"DEMO-VX-{seq_num:04d}"

            # Pick document type and outcome
            doc_type = random.choice(doc_types)
            status, risk_cat, score_range = random.choice(outcomes)
            risk_score = random.choice(score_range)

            # Spread dates across the last 35 days with varied hours/mins/secs
            # Calculate days back so we have records throughout the whole month
            days_ago = (i * 35) // count
            hours_offset = random.randint(8, 20)
            mins_offset = random.randint(0, 59)
            secs_offset = random.randint(0, 59)
            record_date = (now - timedelta(days=days_ago)).replace(
                hour=hours_offset, minute=mins_offset, second=secs_offset, microsecond=0
            )

            # Generate sample JPEG bytes
            img_bytes = create_demo_specimen_image(doc_type, demo_id, status, risk_score)
            img_file = ContentFile(img_bytes, name=f"{demo_id.lower()}.jpg")

            # Create Document record
            doc = Document(
                verification_id=demo_id,
                document_type=doc_type,
                processing_status=status,
                is_demo=True,
                uploaded_by=None
            )
            doc.original_file.save(f"{demo_id.lower()}.jpg", img_file, save=False)
            doc.save()

            # Fix auto_now_add timestamps to the synthetic past date
            Document.objects.filter(id=doc.id).update(
                created_at=record_date,
                updated_at=record_date,
                upload_timestamp=record_date
            )

            # Create VerificationResult
            if status == 'COMPLETED':
                ocr_sc = random.uniform(85.0, 98.0)
                face_sc = random.uniform(88.0, 97.0) if doc_type != 'PAN' else 0.0
                tamper_sc = random.uniform(90.0, 99.0)
                consist_sc = random.uniform(90.0, 100.0)
            elif status == 'MANUAL_REVIEW':
                ocr_sc = random.uniform(55.0, 75.0)
                face_sc = random.uniform(65.0, 80.0) if doc_type != 'PAN' else 0.0
                tamper_sc = random.uniform(65.0, 80.0)
                consist_sc = random.uniform(70.0, 85.0)
            else:
                ocr_sc = random.uniform(40.0, 65.0)
                face_sc = random.uniform(20.0, 45.0) if doc_type != 'PAN' else 0.0
                tamper_sc = random.uniform(25.0, 50.0)
                consist_sc = random.uniform(30.0, 60.0)

            vr = VerificationResult(
                document=doc,
                overall_risk_score=float(risk_score),
                risk_category=risk_cat,
                ocr_score=round(ocr_sc, 1),
                face_score=round(face_sc, 1),
                tamper_score=round(tamper_sc, 1),
                consistency_score=round(consist_sc, 1),
                explanation=explanations[risk_cat if risk_cat in explanations else 'SUSPICIOUS_MANUAL_REVIEW']
            )
            vr.save()

            VerificationResult.objects.filter(id=vr.id).update(
                created_at=record_date,
                updated_at=record_date
            )

            # Create Quality Analysis
            qa = DocumentQualityAnalysis(
                document=doc,
                classified_type=doc_type,
                classification_confidence=0.95,
                quality_score=92 if status == 'COMPLETED' else (75 if status == 'MANUAL_REVIEW' else 50),
                is_poor_quality=(status == 'FAILED'),
                metrics={
                    'blur_score': 150.0,
                    'brightness_score': 140.0,
                    'contrast_score': 60.0
                },
                issues=[] if status != 'FAILED' else ["Elevated image degradation and high ELA residual variance detected."],
                checklist=["✓ Resolution acceptable", "✓ Text readable"]
            )
            qa.save()

            created_records.append({
                'id': demo_id,
                'type': doc_type,
                'status': status,
                'risk': risk_score,
                'date': record_date.strftime('%d-%m-%Y %H:%M:%S')
            })

        self.stdout.write(self.style.SUCCESS(
            f"\n[OK] Successfully created {len(created_records)} DEMO screening records (is_demo=True)."
        ))
        self.stdout.write(f"{'Verification ID':<16} | {'Doc Type':<15} | {'Status':<15} | {'Risk':<8} | {'Screening Date'}")
        self.stdout.write("-" * 80)
        for r in created_records[:10]:
            self.stdout.write(f"{r['id']:<16} | {r['type']:<15} | {r['status']:<15} | {r['risk']:<8} | {r['date']}")
        if len(created_records) > 10:
            self.stdout.write(f"... and {len(created_records) - 10} more records.")

        total_demo = Document.objects.filter(is_demo=True).count()
        total_real = Document.objects.filter(is_demo=False).count()
        self.stdout.write(self.style.SUCCESS(
            f"\nDatabase Summary: {total_demo} Demo Records, {total_real} Real Records. Total: {total_demo + total_real}"
        ))
