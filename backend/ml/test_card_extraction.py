import os
import sys
sys.path.insert(0, os.path.abspath('backend'))
import cv2
import numpy as np
from PIL import Image

doc_path = r'backend/media/documents/2026/09/27/c77a3e27e14d4eeebebd6871709f2808.jpg'
img = cv2.imread(doc_path)
h, w = img.shape[:2]

# Auto-detect document card quadrilateral / contour
def extract_document_card(image):
    h, w = image.shape[:2]
    # Resize for fast contour search
    scale = 800.0 / max(h, w)
    small = cv2.resize(image, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edges & morphological close
    edges = cv2.Canny(blur, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_box = None
    max_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > (small.shape[0] * small.shape[1] * 0.10): # At least 10% of image
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) >= 4 and area > max_area:
                max_area = area
                x, y, cw, ch = cv2.boundingRect(c)
                # Map back to original
                card_box = (int(x / scale), int(y / scale), int(cw / scale), int(ch / scale))
                
    if card_box is not None:
        x, y, cw, ch = card_box
        card_crop = image[max(0, y):min(h, y+ch), max(0, x):min(w, x+cw)]
        print(f"Extracted card crop: {card_crop.shape}")
        return card_crop
    print("No prominent outer card border found, using original image")
    return image

card = extract_document_card(img)

# Now test Authenticity Predictor on original vs card crop
from ml.predict_document import DocumentAuthenticityPredictor
predictor = DocumentAuthenticityPredictor()
print("\n--- On Original Full Photo ---")
print(predictor.predict(img))

print("\n--- On Extracted Card Crop ---")
print(predictor.predict(card))
