import questionary
from core.simple_face_detection import simple_f_main
from core.webcam_face_detection import f_main
from core.sentiment_detection import main_sentiment
from core.train import train_main

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Deactivate OneDNN for Windows
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Deactivate TensorFlow logs

def main_menu():

    choice = questionary.rawselect(
        "What would you like to do?",
        choices=[
            "1- Train model",
            "2- Detect faces in a photo",
            "3- Detect faces in real-time webcam",
            "4- Detect users' facial expressions in real-time webcam",
            "5- Exit"
        ],
    ).ask()

    if "1" in choice:
        print("Training model...")
        train_main()
    elif "2" in choice:
        print("Detecting faces in a photo...")
        simple_f_main()
    elif "3" in choice:
        print("Detecting faces in real-time webcam...")
        f_main()
    elif "4" in choice:
        print("Detecting users' facial expressions in real-time webcam...")
        main_sentiment()
    else:
        print("\n👋 Goodbye!")
       

if __name__ == "__main__":
    main_menu()