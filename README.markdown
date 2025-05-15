# Arabic Sign Language Detection

This project implements a real-time Arabic Sign Language (ArSL) detection system using MediaPipe for hand detection and a Keras-based Convolutional Neural Network (CNN) for classifying hand gestures into Arabic alphabet signs.

## Overview

The system captures video from a webcam, detects hands using MediaPipe's hand tracking, crops the hand region, and classifies the gesture using a pre-trained Keras model. It supports 31 Arabic sign language classes and displays predictions with confidence scores in real-time.

## Features

- **Real-time Hand Detection**: Uses MediaPipe to detect and track hands with high accuracy.
- **Gesture Classification**: Classifies hand gestures into one of 31 Arabic alphabet signs using a CNN.
- **Smoothing Mechanism**: Implements prediction smoothing to reduce flickering and improve stability.
- **FPS Display**: Shows frames per second for performance monitoring.
- **User Interface**: Displays bounding boxes around detected hands and prediction results on the video feed.

## Requirements

To run this project, ensure you have the following:

- Python 3.8 or higher
- A webcam for real-time video capture
- The trained Keras model file (`arabic_sign_lang.keras`)
- Dependencies listed in `requirements.txt`

## Installation

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Trained Model**:
   - Ensure the `arabic_sign_lang.keras` file is placed in the project directory.
   - If training your own model, refer to the provided Jupyter notebook (`Notebook.ipynb`) for training instructions.

## Usage

1. **Run the Detection Script**:
   ```bash
   python main2.py
   ```

2. **Interact with the Application**:
   - The webcam will start, and the system will detect hands in the video feed.
   - A blue bounding box will appear around detected hands, with the predicted sign and confidence score displayed.
   - If no hand is detected, "No hand detected" will be shown.
   - Press `q` to quit the application.

3. **Training the Model** (Optional):
   - Open `Notebook.ipynb` in a Jupyter environment (e.g., Jupyter Notebook or Kaggle).
   - Follow the notebook to preprocess the dataset, train the CNN, and save the model as `arabic_sign_lang.keras`.

## Project Structure

- `main2.py`: Main script for real-time sign language detection.
- `Notebook.ipynb`: Jupyter notebook for dataset preprocessing, model training, and evaluation.
- `arabic_sign_lang.keras`: Pre-trained Keras model for sign classification.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation (this file).

## Dataset

The project uses the RGB Arabic Alphabets Sign Language Dataset, available on Kaggle. It contains images of hand gestures for 31 Arabic alphabet signs. The dataset is processed in `Notebook.ipynb` for training the CNN model.

## Configuration

Key configurations in `main2.py` include:

- `MODEL_PATH`: Path to the Keras model file (`arabic_sign_lang.keras`).
- `INPUT_SIZE`: Image size for model input (200x200 pixels).
- `HAND_DETECTION_CONFIDENCE`: Minimum confidence for hand detection (0.7).
- `HAND_TRACKING_CONFIDENCE`: Minimum confidence for hand tracking (0.5).
- `PREDICTION_THRESHOLD`: Minimum confidence for accepting a prediction (0.6).
- `PADDING_FACTOR`: Padding around the detected hand for cropping (0.5).

## Dependencies

See `requirements.txt` for the full list of required Python packages. Key libraries include:

- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- TensorFlow (`tensorflow`)
- NumPy (`numpy`)

## Notes

- Ensure your webcam is accessible and properly configured.
- The model assumes the hand is the primary focus in the cropped region. Background noise or multiple hands may affect accuracy.
- For better performance, train the model with a diverse dataset and fine-tune hyperparameters in `Notebook.ipynb`.
- The system is optimized for real-time performance but may require a GPU for faster processing.

## Troubleshooting

- **Webcam