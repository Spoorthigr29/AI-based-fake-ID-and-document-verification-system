/**
 * VERIFYX AI — Backend API Client & Forensic Pipeline Connector
 * =============================================================
 * Connects the Netlify Frontend with the Django AI Verification Engine.
 * Supports configurable API host, CORS communication, real-time stage callbacks,
 * and robust demo scenarios.
 */

const VerifyXAPI = (function () {
    const DEFAULT_BACKEND = 'http://127.0.0.1:8000';
    const STORAGE_KEY = 'verifyx_backend_url';

    function getBackendUrl() {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored && stored.trim()) {
            return stored.trim().replace(/\/+$/, '');
        }
        // In local development or Netlify proxy
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return window.location.port === '8000' ? '' : DEFAULT_BACKEND;
        }
        return DEFAULT_BACKEND;
    }

    function setBackendUrl(url) {
        if (!url || !url.trim()) {
            localStorage.removeItem(STORAGE_KEY);
        } else {
            localStorage.setItem(STORAGE_KEY, url.trim().replace(/\/+$/, ''));
        }
    }

    async function checkBackendHealth() {
        const base = getBackendUrl();
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3500);
            const res = await fetch(`${base}/api/documents/upload/`, {
                method: 'OPTIONS',
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return res.ok || res.status === 405 || res.status === 400 || res.status === 200;
        } catch (e) {
            try {
                // Secondary check for root endpoint
                const controller2 = new AbortController();
                const timeoutId2 = setTimeout(() => controller2.abort(), 3000);
                const res2 = await fetch(`${base}/`, {
                    method: 'HEAD',
                    mode: 'no-cors',
                    signal: controller2.signal
                });
                clearTimeout(timeoutId2);
                return true;
            } catch (err2) {
                return false;
            }
        }
    }

    async function uploadDocument(docFile, selfieFile, docType) {
        const base = getBackendUrl();
        const formData = new FormData();
        formData.append('original_file', docFile);
        formData.append('document_type', docType || 'sample_aadhaar');
        if (selfieFile) {
            formData.append('applicant_selfie', selfieFile);
        }

        const res = await fetch(`${base}/api/documents/upload/`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.error || errData.message || `Upload failed with status ${res.status}`);
        }

        return await res.json();
    }

    async function startVerificationPipeline(verificationId, onStageProgress) {
        const base = getBackendUrl();
        const res = await fetch(`${base}/api/documents/start/${verificationId}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.error || errData.message || `Pipeline execution failed with status ${res.status}`);
        }

        return await res.json();
    }

    async function fetchVerificationTelemetry(verificationId) {
        const base = getBackendUrl();
        const res = await fetch(`${base}/api/documents/debug/${verificationId}/`);
        if (!res.ok) {
            const resAlt = await fetch(`${base}/api/verification/${verificationId}/debug/`);
            if (resAlt.ok) {
                return await resAlt.json();
            }
            throw new Error(`Telemetry for ${verificationId} could not be retrieved.`);
        }
        return await res.json();
    }

    // High-fidelity instant demo evaluation scenarios
    const DEMO_CASES = {
        'normal': {
            verification_id: 'VX-DEMO-00177',
            document_type_selected: 'sample_aadhaar',
            document_classifier_result: { detected_type: 'Aadhaar Card', confidence: 0.98, is_type_mismatch: false },
            image_quality: { quality_category: 'EXCELLENT', quality_score: 95, is_poor_quality: false, issues: [], metrics: { blur_laplacian: 512.4, brightness: 188.2, contrast: 64.1, glare_percentage: 0.8 } },
            ocr_analysis: {
                engine: 'EasyOCR Multi-Engine',
                overall_confidence: 0.94,
                extracted_fields: [
                    { field: 'full_name', value: 'SPOORTHI G R', confidence: 0.96, validation_status: 'VALID' },
                    { field: 'dob', value: '14/05/2002', confidence: 0.95, validation_status: 'VALID' },
                    { field: 'gender', value: 'FEMALE', confidence: 0.98, validation_status: 'VALID' },
                    { field: 'aadhaar_number', value: 'XXXX XXXX 8942', confidence: 0.97, validation_status: 'VALID' }
                ]
            },
            qr_detection: { qr_detected: true, qr_decoded: true, format_type: 'UIDAI Secure QR', signature_status: 'Cryptographic Digest Validated', consistency_status: 'CONSISTENT', consistency_score: 1.0 },
            template_analysis: { template_status: 'AUTHENTIC_LAYOUT', confidence: 0.96 },
            tamper_detection: { tamper_score: 0.04, tamper_status: 'LOW', region_scores: { photo_box: 0.02, name_box: 0.03, dob_box: 0.01, number_box: 0.04, background_texture: 0.05 }, forensic_signals: { signals: ['Natural sensor noise distribution', 'Zero clone/brush artifact anomalies'] } },
            face_biometrics: { document_face_detected: true, selfie_face_detected: true, liveness_status: 'PASSED', similarity_score: 0.94, calibrated_threshold: 0.70, match_status: 'MATCHED' },
            identity_consistency: { consistency_score: 100, status: 'CONSISTENT', field_matches: { name_match: true, dob_match: true, document_number_match: true, gender_match: true } },
            risk_engine: { risk_score: 12, risk_category: 'LOW_RISK', explanation: 'All multi-level biometric, cryptographic QR, optical layout, and cross-consistency checks verified with high confidence. Document classified as authentic.', factor_contributions: [{ factor: 'Biometric Facial Similarity', contribution: 'Positive (94% Match)' }, { factor: 'Cryptographic QR Consistency', contribution: 'Positive (100% Match)' }, { factor: 'Forensic ELA Tamper Analysis', contribution: 'Clean (4% Noise Variance)' }] }
        },
        'tampered': {
            verification_id: 'VX-DEMO-00178',
            document_type_selected: 'sample_pan',
            document_classifier_result: { detected_type: 'PAN Card', confidence: 0.95, is_type_mismatch: false },
            image_quality: { quality_category: 'ACCEPTABLE', quality_score: 82, is_poor_quality: false, issues: [], metrics: { blur_laplacian: 340.1, brightness: 160.5, contrast: 52.3 } },
            ocr_analysis: {
                engine: 'EasyOCR Multi-Engine',
                overall_confidence: 0.89,
                extracted_fields: [
                    { field: 'pan_number', value: 'ABCDE1234F', confidence: 0.91, validation_status: 'VALID' },
                    { field: 'holder_name', value: 'RAHUL SHARMA', confidence: 0.72, validation_status: 'SUSPICIOUS_FONT' },
                    { field: 'dob', value: '22/09/1990', confidence: 0.88, validation_status: 'VALID' }
                ]
            },
            qr_detection: { qr_detected: false, qr_decoded: false, format_type: 'None', signature_status: 'No QR Code on specimen', consistency_status: 'NOT_APPLICABLE' },
            template_analysis: { template_status: 'SUSPICIOUS_SPACING', confidence: 0.68 },
            tamper_detection: { tamper_score: 0.78, tamper_status: 'HIGH', region_scores: { name_box: 0.84, dob_box: 0.69, number_box: 0.22, photo_box: 0.15, background_texture: 0.75 }, forensic_signals: { signals: ['High error level analysis variance on name strip', 'Font stroke edge discontinuity detected', 'Resampling grid misalignment'] } },
            face_biometrics: { document_face_detected: false, selfie_face_detected: true, liveness_status: 'PASSED', similarity_score: 0.0, calibrated_threshold: 0.70, match_status: 'NOT_APPLICABLE' },
            identity_consistency: { consistency_score: 65, status: 'INCONSISTENT', field_matches: { name_match: false, dob_match: true, document_number_match: true, gender_match: true } },
            risk_engine: { risk_score: 82, risk_category: 'HIGH_RISK', explanation: 'Forensic Error Level Analysis (ELA) detected high-confidence digital tampering and font splice anomalies in the Name text strip. Rejection recommended.', factor_contributions: [{ factor: 'Forensic Tamper Anomaly', contribution: 'Severe (+55 Risk)' }, { factor: 'Typography / Spacing Discontinuity', contribution: 'High (+20 Risk)' }] }
        },
        'identity_mismatch': {
            verification_id: 'VX-DEMO-00179',
            document_type_selected: 'sample_driving_license',
            document_classifier_result: { detected_type: 'Driving License', confidence: 0.96, is_type_mismatch: false },
            image_quality: { quality_category: 'GOOD', quality_score: 88, is_poor_quality: false, issues: [] },
            ocr_analysis: {
                engine: 'EasyOCR Multi-Engine',
                overall_confidence: 0.91,
                extracted_fields: [
                    { field: 'license_number', value: 'KA01 20180019284', confidence: 0.94, validation_status: 'VALID' },
                    { field: 'holder_name', value: 'PRIYA SUNDARAM', confidence: 0.92, validation_status: 'VALID' },
                    { field: 'dob', value: '10/02/1995', confidence: 0.91, validation_status: 'VALID' }
                ]
            },
            qr_detection: { qr_detected: true, qr_decoded: true, format_type: 'Parivahan QR', signature_status: 'Decoded', consistency_status: 'MISMATCH', mismatched_fields: ['holder_name', 'dob'] },
            template_analysis: { template_status: 'AUTHENTIC_LAYOUT', confidence: 0.93 },
            tamper_detection: { tamper_score: 0.28, tamper_status: 'MEDIUM', region_scores: { name_box: 0.35, dob_box: 0.32, number_box: 0.12, photo_box: 0.18, background_texture: 0.20 }, forensic_signals: { signals: ['Visual text conflicts with encoded metadata payload'] } },
            face_biometrics: { document_face_detected: true, selfie_face_detected: true, liveness_status: 'PASSED', similarity_score: 0.88, calibrated_threshold: 0.70, match_status: 'MATCHED' },
            identity_consistency: { consistency_score: 42, status: 'INCONSISTENT', field_matches: { name_match: false, dob_match: false, document_number_match: true, gender_match: true } },
            risk_engine: { risk_score: 68, risk_category: 'MANUAL_REVIEW', explanation: 'Visual OCR text conflicts directly with encrypted QR metadata payload. Document forwarded for institutional officer review.', factor_contributions: [{ factor: 'Metadata / QR Discrepancy', contribution: 'Significant (+45 Risk)' }] }
        },
        'poor_quality': {
            verification_id: 'VX-DEMO-00180',
            document_type_selected: 'sample_passport',
            document_classifier_result: { detected_type: 'Passport', confidence: 0.74, is_type_mismatch: false },
            image_quality: { quality_category: 'POOR', quality_score: 38, is_poor_quality: true, issues: ['Severe motion blur detected (Laplacian: 42.1)', 'Specular glare covers MRZ strip', 'Insufficient illumination (< 40 Lux)'], metrics: { blur_laplacian: 42.1, brightness: 62.4, contrast: 24.1, glare_percentage: 18.5 } },
            ocr_analysis: { engine: 'EasyOCR Multi-Engine', overall_confidence: 0.42, extracted_fields: [] },
            qr_detection: { qr_detected: false, qr_decoded: false, format_type: 'None' },
            template_analysis: { template_status: 'UNREADABLE', confidence: 0.35 },
            tamper_detection: { tamper_score: 0.15, tamper_status: 'LOW', region_scores: {} },
            face_biometrics: { document_face_detected: false, selfie_face_detected: true, liveness_status: 'PASSED', similarity_score: 0.0, match_status: 'NOT_APPLICABLE' },
            identity_consistency: { consistency_score: 50, status: 'INCONSISTENT', field_matches: {} },
            risk_engine: { risk_score: 55, risk_category: 'MANUAL_REVIEW', explanation: 'Image quality is too low or blurry to extract critical security features. Rescan requested from applicant.', factor_contributions: [{ factor: 'Severe Optical Blur & Glare', contribution: 'Critical (Rescan Required)' }] }
        },
        'face_mismatch': {
            verification_id: 'VX-DEMO-00181',
            document_type_selected: 'sample_passport',
            document_classifier_result: { detected_type: 'Passport Data Page', confidence: 0.97, is_type_mismatch: false },
            image_quality: { quality_category: 'EXCELLENT', quality_score: 92, is_poor_quality: false, issues: [] },
            ocr_analysis: {
                engine: 'EasyOCR Multi-Engine',
                overall_confidence: 0.96,
                extracted_fields: [
                    { field: 'passport_number', value: 'Z9841029', confidence: 0.98, validation_status: 'VALID' },
                    { field: 'holder_name', value: 'VIKRAM MALHOTRA', confidence: 0.97, validation_status: 'VALID' },
                    { field: 'dob', value: '04/11/1986', confidence: 0.95, validation_status: 'VALID' }
                ]
            },
            qr_detection: { qr_detected: false, qr_decoded: false, format_type: 'ICAO MRZ', signature_status: 'MRZ Checksum Valid' },
            template_analysis: { template_status: 'AUTHENTIC_LAYOUT', confidence: 0.95 },
            tamper_detection: { tamper_score: 0.08, tamper_status: 'LOW', region_scores: { photo_box: 0.06, background_texture: 0.05 } },
            face_biometrics: { document_face_detected: true, selfie_face_detected: true, liveness_status: 'PASSED', similarity_score: 0.31, calibrated_threshold: 0.70, match_status: 'FAILED_MISMATCH' },
            identity_consistency: { consistency_score: 95, status: 'CONSISTENT', field_matches: { name_match: true, dob_match: true, document_number_match: true, gender_match: true } },
            risk_engine: { risk_score: 88, risk_category: 'HIGH_RISK', explanation: 'Live applicant facial embedding failed biometrics comparison against the identity document photograph (Cosine similarity 0.31 < 0.70 threshold). Potential synthetic identity fraud.', factor_contributions: [{ factor: 'Biometric Face Discrepancy', contribution: 'Critical (+75 Risk)' }] }
        }
    };

    return {
        getBackendUrl,
        setBackendUrl,
        checkBackendHealth,
        uploadDocument,
        startVerificationPipeline,
        fetchVerificationTelemetry,
        DEMO_CASES
    };
})();

// Export globally
window.VerifyXAPI = VerifyXAPI;
