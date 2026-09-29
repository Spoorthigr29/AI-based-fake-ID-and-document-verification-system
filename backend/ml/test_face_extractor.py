import cv2
import numpy as np
from PIL import Image

doc_path = r'backend/media/documents/2026/09/27/c77a3e27e14d4eeebebd6871709f2808.jpg'
selfie_path = r'backend/media/selfies/2026/09/27/f74719ef8caa47358c3b82fef91d75fb.jpg'

doc_img = cv2.imread(doc_path)
selfie_img = cv2.imread(selfie_path)

print("Doc shape:", doc_img.shape if doc_img is not None else "None")
print("Selfie shape:", selfie_img.shape if selfie_img is not None else "None")

# Function to detect face using Skin color + Eye/Facial structure in document
def find_face_candidates(img, is_doc=False):
    h, w = img.shape[:2]
    # Downscale for processing if huge
    max_dim = 1200
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        work_img = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        work_img = img.copy()
        
    wh, ww = work_img.shape[:2]
    hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(work_img, cv2.COLOR_BGR2YCrCb)
    
    # Skin masks
    mask_hsv = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([35, 255, 255]))
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 130, 70]), np.array([255, 185, 135]))
    skin = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = ch / max(1, cw)
        # Face aspect ratio usually 1.0 - 1.8
        if 0.8 <= aspect <= 2.0 and cw >= 30 and ch >= 35:
            # Map back to original image coordinates
            ox = int(x / scale)
            oy = int(y / scale)
            ocw = int(cw / scale)
            och = int(ch / scale)
            faces.append((ox, oy, ocw, och))
            
    print(f"Found {len(faces)} face candidates for is_doc={is_doc}: {faces[:5]}")
    return faces

find_face_candidates(doc_img, is_doc=True)
find_face_candidates(selfie_img, is_doc=False)
