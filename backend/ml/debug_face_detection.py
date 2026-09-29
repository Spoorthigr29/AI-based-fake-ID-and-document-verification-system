import cv2
import numpy as np
from PIL import Image

path = r'backend/media/documents/2026/09/27/c77a3e27e14d4eeebebd6871709f2808.jpg'
img = cv2.imread(path)
print("Image shape:", img.shape if img is not None else "None")

# Try Haar cascade on various downscaled resolutions and with various parameters
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
norm = clahe.apply(gray)

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
cascade_alt = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
cascade_profile = cv2.data.haarcascades + 'haarcascade_profileface.xml'

face_cascade = cv2.CascadeClassifier(cascade_path)
face_alt = cv2.CascadeClassifier(cascade_alt)

for scale in [1.0, 0.5, 0.33, 0.25]:
    resized = cv2.resize(norm, (0, 0), fx=scale, fy=scale)
    rects = face_cascade.detectMultiScale(resized, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    rects2 = face_alt.detectMultiScale(resized, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    print(f"Scale {scale} (size {resized.shape}): default={len(rects)}, alt={len(rects2)}")
    if len(rects) > 0:
        print("  Detected default boxes:", rects)
    if len(rects2) > 0:
        print("  Detected alt boxes:", rects2)
