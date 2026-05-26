import cv2
from fer import FER

# Load model
detector = FER(mtcnn=True)

def detect_emotions(frame):
    results = detector.detect_emotions(frame)
    return results