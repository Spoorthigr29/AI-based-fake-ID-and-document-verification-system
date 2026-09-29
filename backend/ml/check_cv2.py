import cv2
print("cv2 version:", cv2.__version__)
print("has CascadeClassifier:", hasattr(cv2, 'CascadeClassifier'))
if hasattr(cv2, 'objdetect'):
    print("cv2.objdetect:", dir(cv2.objdetect))

# Let's check MTCNN / InsightFace / dlib / Haar / DNN
try:
    import insightface
    print("insightface available!")
except Exception as e:
    print("insightface not available:", e)

try:
    import face_recognition
    print("face_recognition available!")
except Exception as e:
    print("face_recognition not available:", e)

try:
    from PIL import Image
    print("PIL available!")
except Exception as e:
    print("PIL not available:", e)
