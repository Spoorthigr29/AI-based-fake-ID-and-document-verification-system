import cv2
import torch
import torchvision

print("cv2.dnn available:", hasattr(cv2, 'dnn'))
if hasattr(cv2, 'dnn'):
    print("cv2.dnn functions:", [f for f in dir(cv2.dnn) if 'Net' in f or 'read' in f])

print("torchvision models:", dir(torchvision.models))
