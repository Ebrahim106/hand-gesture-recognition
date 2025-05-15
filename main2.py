# ============================ #
# Arabic Sign Language Detection with MediaPipe & Keras
# ============================ #

# ---------- Imports ----------
import cv2
import numpy as np
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------- Configuration ----------
MODEL_PATH = "arabic_sign_lang.keras"             # Path to the trained Keras model
INPUT_SIZE = (200, 200)                           # Input size expected by the CNN model
HAND_DETECTION_CONFIDENCE = 0.7                   # Minimum confidence for detecting hands
HAND_TRACKING_CONFIDENCE = 0.5                    # Minimum confidence for tracking hands
PADDING_FACTOR = 0.5                              # Padding around the detected hand for cropping
PREDICTION_THRESHOLD = 0.6                        # Minimum confidence for accepting prediction

# Arabic sign language classes (must match model output)
class_names = [
    'Ain', 'Al', 'Alef', 'Beh', 'Dad', 'Dal', 'Feh', 'Ghain', 'Hah', 'Heh',
    'Jeem', 'Kaf', 'Khah', 'Laa', 'Lam', 'Meem', 'Noon', 'Qaf', 'Reh', 'Sad',
    'Seen', 'Sheen', 'Tah', 'Teh', 'Teh_Marbuta', 'Thal', 'Theh', 'Waw', 'Yeh', 'Zah', 'Zain'
]

# ---------- Preprocessing ----------
def preprocess_image(image):
    """Preprocess image for model input."""
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ---------- Main Detection Function ----------
def run_real_time_detection():
    # Load Keras model
    try:
        model = load_model(MODEL_PATH)
        print(" Keras model loaded successfully.")
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE
    )
    print(" MediaPipe Hands initialized.")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Webcam not accessible.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(" Starting real-time detection. Press 'q' to quit.")

    # ---------- Tracking Variables ----------
    frame_count = 0
    start_time = time.time()
    fps = 0
    last_prediction = "No hand detected"
    prediction_confidence = 0.0
    prediction_history = []
    max_history = 5

    # ---------- Main Loop ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Hand Detection
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                # Bounding box
                h, w, _ = frame.shape
                x_min, x_max, y_min, y_max = w, 0, h, 0
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min, x_max = min(x_min, x), max(x_max, x)
                    y_min, y_max = min(y_min, y), max(y_max, y)

                # Square crop with padding
                square_size = max(x_max - x_min, y_max - y_min)
                center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                padded_size = int(square_size * (1 + PADDING_FACTOR))
                x1 = max(0, center_x - padded_size // 2)
                y1 = max(0, center_y - padded_size // 2)
                x2 = min(w, center_x + padded_size // 2)
                y2 = min(h, center_y + padded_size // 2)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                hand_img = frame[y1:y2, x1:x2]

                if hand_img.size != 0:
                    processed_img = preprocess_image(hand_img)
                    predictions = model.predict(processed_img, verbose=0)[0]
                    class_idx = np.argmax(predictions)
                    confidence = predictions[class_idx]

                    if confidence >= PREDICTION_THRESHOLD:
                        predicted_class = class_names[class_idx] if class_idx < len(class_names) else f"Unknown ({class_idx})"
                        prediction_history.append((predicted_class, confidence))
                        if len(prediction_history) > max_history:
                            prediction_history.pop(0)

                        # Smoothing
                        pred_counts = {}
                        for pred, conf in prediction_history:
                            if pred in pred_counts:
                                pred_counts[pred] = (pred_counts[pred][0] + 1, pred_counts[pred][1] + conf)
                            else:
                                pred_counts[pred] = (1, conf)

                        max_count, max_conf = 0, 0
                        for pred, (count, conf) in pred_counts.items():
                            avg_conf = conf / count
                            if count > max_count or (count == max_count and avg_conf > max_conf):
                                max_count, max_conf = count, avg_conf
                                last_prediction, prediction_confidence = pred, avg_conf

                # Display prediction
                cv2.putText(display_frame, f"Sign: {last_prediction} ({prediction_confidence:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            cv2.putText(display_frame, "No hand detected", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            prediction_history = []

        # FPS calculation
        frame_count += 1
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            frame_count = 0
            start_time = current_time

        # Display
        cv2.imshow("Arabic Sign Language Detection", display_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------- Cleanup ----------
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

# ---------- Run Script ----------
if __name__ == "__main__":
    run_real_time_detection()
