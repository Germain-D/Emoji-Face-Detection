import cv2
import mediapipe as mp

def f_load_models():

    """Load the MediaPipe Face Detection model."""

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    return face_detection

def f_detect_faces(emoji, face_detection):

    """Detect faces in real-time webcam feed and overlay an emoji on each face."""

    cap = cv2.VideoCapture(0)
    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    width = min(width, w - x)
                    height = min(height, h - y)

                    if width > 0 and height > 0:  
                        emoji_resized = cv2.resize(emoji, (width, height))
                        
                        alpha = emoji_resized[:, :, 3] / 255.0
                        for c in range(3):
                            try:
                                frame[y:y+height, x:x+width, c] = (
                                    alpha * emoji_resized[:, :, c] + 
                                    (1 - alpha) * frame[y:y+height, x:x+width, c]
                                )
                            except ValueError:

                                emoji_resized = cv2.resize(emoji_resized, (frame[y:y+height, x:x+width, c].shape[1], frame[y:y+height, x:x+width, c].shape[0]))
                                alpha = emoji_resized[:, :, 3] / 255.0
                                frame[y:y+height, x:x+width, c] = alpha * emoji_resized[:, :, c] + (1 - alpha) * frame[y:y+height, x:x+width, c]

            cv2.imshow('Emoji Face Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def f_main():
    face_detection = f_load_models()
    emoji = cv2.imread("../images/emoji.png", cv2.IMREAD_UNCHANGED)
    f_detect_faces(emoji, face_detection)

if __name__ == "__main__":
    f_main()