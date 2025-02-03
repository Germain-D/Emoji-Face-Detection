# Emoji Face Detection and Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-red)

An interactive system that **detects facial expressions in real-time**, overlays dynamic emojis based on your emotions (e.g., smiling, laughing), and analyzes sentiment in static images. Built with OpenCV, MediaPipe, and Keras/TensorFlow.

## 🎯 Project Objective

**Primary Goal**:  
Detect users' facial expressions in real-time using a webcam and **dynamically overlay emotion-specific emojis** (🎭) onto detected faces. This turns your physical expressions into playful digital feedback!  

**Examples**:
- 😊 **Smile** → Displays a smiling emoji.
- 😂 **Laughter** → Shows a laughing emoji.
- 🥱 **Yawn** → Triggers a sleepy emoji.
- 😐 **Neutral** → Default neutral face.

**Key Objectives**:
1. **Real-Time Expression Detection**: Classify facial expressions (smile, laugh, neutral) using machine learning.
2. **Dynamic Emoji Switching**: Instantly update emojis based on detected emotions.
3. **Edge-Case Handling**: Smooth emoji placement even near screen borders.
4. **Sentiment Analysis**: Predict emotions from static images as an auxiliary feature.

---

## Features

- **Real-Time Face Detection**: Identify faces in live webcam feeds using MediaPipe.
- **Dynamic Emoji Overlay**: Replace detected faces with custom emojis (with transparency support).
- **Sentiment Analysis**: Predict emotions (e.g., positive/negative) from static images using a pre-trained CNN model.
- **Bounding Box Adjustments**: Handle edge cases where faces are near the screen borders.

## Installation

### Prerequisites
- Python 3.9+
- Webcam (for real-time detection)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Emoji-Face-Detection.git
   cd Emoji-Face-Detection
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Real-Time Face Detection with Emoji Overlay
Run the script and specify an emoji image (PNG with transparency):
```bash
python detect_emoji.py --model ./models/happy.png
```
- Press `q` to quit the live feed.
- Adjust emoji size dynamically based on face detection bounding boxes.

### Sentiment Analysis on Images
Use the pre-trained CNN model to analyze sentiment in a static image:
```python
python sentiment_predict.py --image ./images/photo.jpg
```
**Output Example**:
```
Sentiment: Positive | Confidence: 93.45%
```

## Model Training
The CNN model for sentiment analysis is pre-trained and stored in `./models/`. To retrain it:
1. Place your dataset in `./data/` (structured into train/validation folders).
2. Run the training script:
   ```bash
   python train_model.py --epochs 20 --batch_size 32
   ```
3. Trained models will be saved to `./models/`.

## Examples
| Real-Time Emoji Overlay | Sentiment Prediction | 
|-------------------------|----------------------|
| ![Demo](demo.gif)       | ![Result](result.png)|

## Troubleshooting
- **Shape Mismatch Errors**: Ensure the input image dimensions match `model.input_shape` (check with `print(model.input_shape)`).
- **Webcam Not Detected**: Verify camera permissions or use a static image for testing.
- **Dependency Conflicts**: Use the exact versions in `requirements.txt`.

## Contributing
Pull requests are welcome! For major changes, open an issue first to discuss your ideas.

## License
[MIT](LICENSE)

---

**References**:
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [Keras Model Saving Guide](https://keras.io/guides/saving_model/)
``` 
