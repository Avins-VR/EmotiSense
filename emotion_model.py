from fer import FER

detector = FER()

def detect_emotions(frame):
    return detector.detect_emotions(frame)
