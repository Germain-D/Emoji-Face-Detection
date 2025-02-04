import cv2
import mediapipe as mp
import os

def simple_f_load_models():
    
    """Load the MediaPipe Face Detection model and the drawing utility."""

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    return mp_face_detection, mp_drawing

def simple_f_detect_faces(emoji, mp_face_detection):

    """Detect faces in a photo and overlay an emoji on each face."""


    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        image_path = os.path.join(os.path.dirname(__file__), "../images/photo.png")
        image = cv2.imread(image_path)
    
        if image is None:
            print(f"Error : Image not found at location {image_path}")
            return 
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
                    width, height = int(bboxC.width * w), int(bboxC.height * h)


                    emoji_resized = cv2.resize(emoji, (width, height))
                    for c in range(0, 3):
                        image[y:y+height, x:x+width, c] = \
                            emoji_resized[:, :, c] * (emoji_resized[:, :, 3] / 255.0) + \
                            image[y:y+height, x:x+width, c] * (1.0 - emoji_resized[:, :, 3] / 255.0)
            cv2.imshow("Face with Emoji", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def simple_f_main():
    mp_face_detection, mp_drawing = simple_f_load_models()
    
    emoji_path = os.path.join(os.path.dirname(__file__), "../images/emoji.png")
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    
    if emoji is None:
        print(f"Error: Emoji not found at location {emoji_path}")
        return
    
    simple_f_detect_faces(emoji, mp_face_detection)


if __name__ == "__main__":
    simple_f_main()