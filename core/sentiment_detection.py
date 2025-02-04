import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

def load_CNN():

    """Load the trained CNN model for emotion detection."""

    model = keras.models.load_model("../models/CNN.keras")
    return model

def init_MediaPipe():
    
    """Initialize the MediaPipe Face Detection model."""

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    return face_detection

def preprocess_face(face_img, target_size=(48, 48)):
    
    """Preprocess the face image for emotion detection."""

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=-1) 
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def overlay_emoji(frame, emoji, x, y, w, h):
    
    """Overlay the emoji on the face in the frame."""

    if emoji is None or emoji.size == 0 or w <=0 or h <=0:
        return frame
    
    try:
        emoji = cv2.resize(emoji, (w, h))
        alpha = emoji[:, :, 3] / 255.0
        for c in range(3):
            frame_roi = frame[y:y+h, x:x+w, c]
            if frame_roi.shape != alpha.shape:
                alpha = cv2.resize(alpha, (frame_roi.shape[1], frame_roi.shape[0]))
            frame[y:y+h, x:x+w, c] = alpha * emoji[:, :, c] + (1 - alpha) * frame_roi
    except Exception as e:
        print(f"Error overlaying emoji: {e}")
    return frame

def video_face_detection(face_detection, model, EMOJI_MAP):
    
    """Perform real-time emotion detection on webcam feed."""

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                face_w = max(10, int(bbox.width * w)) 
                face_h = max(10, int(bbox.height * h))


                face_w = min(face_w, w - x)
                face_h = min(face_h, h - y)

                if face_w <= 0 or face_h <= 0:
                    continue


                face_roi = frame[y:y+face_h, x:x+face_w]

   
                if face_roi.size == 0 or face_roi.shape[0] != face_h or face_roi.shape[1] != face_w:
                    continue
                
                processed_face = preprocess_face(face_roi)


                predictions = model.predict(processed_face)
                emotion_id = np.argmax(predictions)
                emotion_label, emoji = EMOJI_MAP[emotion_id]

                frame = overlay_emoji(frame, emoji, x, y, face_w, face_h)

        cv2.imshow('Emoji Emotion Overlay', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main_sentiment():
    EMOJI_MAP = {
        0: ("colere", cv2.imread("../images/emojis/angry.png", cv2.IMREAD_UNCHANGED)),
        1: ("degout", cv2.imread("../images/emojis/disgust.png", cv2.IMREAD_UNCHANGED)),
        2: ("peur", cv2.imread("../images/emojis/fear.png", cv2.IMREAD_UNCHANGED)),
        3: ("heureux", cv2.imread("../images/emojis/happy.png", cv2.IMREAD_UNCHANGED)),
        4: ("neutre", cv2.imread("../images/emojis/neutral.png", cv2.IMREAD_UNCHANGED)),
        5: ("triste", cv2.imread("../images/emojis/sad.png", cv2.IMREAD_UNCHANGED)),
        6: ("surprise", cv2.imread("../images/emojis/surprised.png", cv2.IMREAD_UNCHANGED))
    }
    for emotion_id, (label, emoji) in EMOJI_MAP.items():
        if emoji is None:
            raise FileNotFoundError(f"Emoji {label} (ID: {emotion_id}) not found.")
        
    model = load_CNN()
    face_detection = init_MediaPipe()
    video_face_detection(face_detection, model, EMOJI_MAP)


if __name__ == "__main__":
    main_sentiment()

