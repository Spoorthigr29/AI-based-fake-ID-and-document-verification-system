# VerifyX AI — Frontend Architecture & UI Assets

This directory contains the user-facing presentation layer for **VerifyX AI (AI-Based Fake Identity & Document Screening System)**.

---

## 📁 Directory Structure

```
frontend/
├── templates/                     # Modular Django HTML5 templates
│   ├── base.html                  # Global base layout (Dark theme, Glassmorphism, Bootstrap 5.3 + FontAwesome)
│   ├── accounts/                  # User login & authentication views
│   ├── dashboard/                 # Central forensic intelligence dashboard & analytics
│   ├── documents/                 # Document ingestion, camera capture, progress animation & verification report
│   ├── face_verification/         # Biometric facial comparison results & interactive side-by-side viewer
│   ├── ocr_engine/                # Multi-engine OCR text & field extraction inspector
│   ├── reports/                   # Executive PDF/Print summary generation
│   └── tamper_detection/          # Forensic ELA & digital manipulation analysis visualizer
├── static/                        # Static client assets
│   ├── css/                       # Custom design system stylesheets
│   ├── js/                        # Client-side scripts (Dynamic capabilities, camera capture, Charts)
│   └── images/                    # UI branding, logos, and synthetic sample assets
└── README.md                      # Frontend documentation
```

---

## 🎨 Design System & Visual Aesthetics

- **Glassmorphism & Cyber-Forensic Theme**: Dark navy background (`#0b0f19` / `#0f172a`) with translucent frosted-glass cards (`rgba(30, 41, 59, 0.7)`), cyan/emerald accent highlights, and subtle border glows.
- **Dynamic Capabilities Switching**: Client-side JavaScript updates UI fields, notices, and camera controls in real-time when switching between document templates (e.g. automatically hiding webcam capture for non-photo documents like PAN cards).
- **Interactive Camera Viewfinder**: WebRTC `getUserMedia()` integration enabling direct live webcam capture with canvas-based frame extraction, preview, and retake workflows.
- **Forensic Visualization**: Real-time Chart.js radar charts and probability gauges for multi-modal risk signals.

---

## 🚀 Key Templates & Views

| Template | Route | Description |
| :--- | :--- | :--- |
| `documents/upload.html` | `/documents/upload/` | Document intake console with dynamic capabilities switching and live webcam selfie capture. |
| `documents/processing.html` | `/documents/processing/<id>/` | 12-stage animated forensic screening pipeline progress screen. |
| `documents/detail.html` | `/documents/<id>/` | Comprehensive Verification Screening Audit Report with radar charts, risk breakdown, and forensic telemetry. |
| `dashboard/dashboard.html` | `/dashboard/` | High-level operations console with screening metrics and recent cases. |
| `dashboard/analytics.html` | `/analytics/` | Statistical analytics on document types, risk distributions, and anomaly rates. |
