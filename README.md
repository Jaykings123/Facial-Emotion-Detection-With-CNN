# Facial Emotion Detection

This project implements a real-time facial emotion detection system using a Convolutional Neural Network (CNN) trained on the FER2013 dataset. The system can detect 7 different emotions from facial expressions in real-time using a webcam.

## Features

- Real-time facial emotion detection using webcam
- Support for 7 emotion categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- High accuracy model trained on FER2013 dataset
- Face detection using YuNet ONNX model

## Project Structure

```
├── README.md
├── requirements.txt
├── webcam.py                 # Real-time emotion detection using webcam
├── facial_emotion_detection.py  # Main implementation and model training
├── facial_emotion_model.keras   # Trained model file
├── face_detection_yunet_2023mar.onnx  # Face detection model
├── deploy.prototxt           # Model deployment configuration
├── project_report.md         # Detailed project documentation
└── images/                   # Sample images and visualizations
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd facial-emotion-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the FER2013 dataset:
   - The dataset is not included in the repository due to its size
   - Download it from [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
   - Place the dataset in the project root directory

4. Run the real-time emotion detection:
   ```bash
   python webcam.py
   ```

## Model Architecture

The CNN model consists of:
- Multiple convolutional layers with batch normalization
- Max pooling layers
- Dropout for regularization
- Dense layers for classification

## Performance

The model achieves high accuracy in emotion detection and can process webcam feed in real-time. For detailed performance metrics and visualizations, please refer to `project_report.md`.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 