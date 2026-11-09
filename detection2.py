import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import tensorflow as tf
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import argparse
warnings.filterwarnings('ignore')

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    
    if angle < 0:
        angle += 360
    
    return angle

def angles_finder(landmarks):
    angles = []
    
    all_visibilities = []
    for pair in angle_pair_indices:
        # Calculate average visibility for this angle triplet
        avg_visibility = (landmarks[pair[0].value].visibility + 
                        landmarks[pair[1].value].visibility + 
                        landmarks[pair[2].value].visibility) / 3
        all_visibilities.append(avg_visibility)
        
        angle = calculateAngle(
            (landmarks[pair[0].value].x, landmarks[pair[0].value].y, landmarks[pair[0].value].z),
            (landmarks[pair[1].value].x, landmarks[pair[1].value].y, landmarks[pair[1].value].z),
            (landmarks[pair[2].value].x, landmarks[pair[2].value].y, landmarks[pair[2].value].z)
        )
        angles.append(angle)
    
    # Only return None if the overall average visibility is too low
    if np.mean(all_visibilities) < 0.6:
        return None, None
        
    return angles, all_visibilities

def predict_yoga_pose(angles, model, ideal_angles):
    input_data = np.array(angles).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    predicted_class_index = np.argmax(prediction)
    predicted_class = le.inverse_transform([predicted_class_index])[0]
    confidence = prediction[0][predicted_class_index]
    
    angle_differences = input_data - ideal_angles.loc[predicted_class].values
    
    return predicted_class, confidence, angle_differences

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model, scaler, and ideal angles
loaded_model = tf.keras.models.load_model('yoga_pose_model.keras')
scaler = joblib.load('yoga_pose_scaler.joblib')
le = joblib.load('yoga_pose_label_encoder.joblib')
ideal_angles = pd.read_csv('ideal_angles.csv', index_col='Label')

# Load ideal images
ideal_image_folder = 'ideal_images'
ideal_images = {pose_name: cv2.imread(os.path.join(ideal_image_folder, f"{pose_name}.jpg"))
                for pose_name in ideal_angles.index}

# Parse arguments
parser = argparse.ArgumentParser(description="Yoga Pose Estimation & Feedback")
parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or path to video/image file")
parser.add_argument("--width", type=int, default=1280, help="Window display width (default: 1280)")
parser.add_argument("--height", type=int, default=720, help="Window display height (default: 720)")
args = parser.parse_args()

# Modified window creation with split screen layout
screen_width, screen_height = args.width, args.height
video_width = screen_width // 2
video_height = screen_height
cv2.namedWindow('Yoga Pose Prediction', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Yoga Pose Prediction', screen_width, screen_height)

# Set the interval for processing frames (e.g., every 0.5 second)
process_interval = .5 # in seconds
last_process_time = 0.0

last_predicted_pose = ""
last_confidence = 0.0
last_angle_differences = None
feedback = []

angle_pair_indices = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE)
]

angle_names = {
    0: "left_elbow_angle",
    1: "right_elbow_angle",
    2: "left_shoulder_angle",
    3: "right_shoulder_angle",
    4: "left_knee_angle",
    5: "right_knee_angle",
    6: "hand_angle",
    7: "left_hip_angle",
    8: "right_hip_angle",
    9: "neck_angle_uk",
    10: "left_wrist_angle_bk",
    11: "right_wrist_angle_bk"
}

max_angle_variance = 18  # Maximum allowed variance from ideal angle
min_classification_confidence = 0.65  # Minimum required classification confidence

def process_frame(image):
    global last_process_time, last_predicted_pose, last_confidence, last_angle_differences, feedback

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(image_rgb)

    # Create a larger canvas to hold both the video and information
    display_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Resize the input image to take up half the screen
    image_resized = cv2.resize(image, (video_width, video_height))

    # Draw the pose landmarks on the resized image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if it's time to process a frame
        current_time = time.time()
        if current_time - last_process_time >= process_interval:
            # Extract angles from landmarks
            angles, visibilities = angles_finder(results.pose_landmarks.landmark)

            # Only predict if all required landmarks are visible enough
            if angles is not None:
                # Predict yoga pose and get confidence
                predicted_pose, confidence, angle_differences = predict_yoga_pose(angles, loaded_model, ideal_angles)

                # Update the last predicted pose, confidence, and angle differences
                last_predicted_pose = predicted_pose
                last_confidence = confidence
                last_angle_differences = angle_differences

                # Check angles against the maximum allowed variance
                feedback = []
                if confidence >= min_classification_confidence:
                    for i, angle_diff in enumerate(angle_differences[0]):
                        if abs(angle_diff) > max_angle_variance:
                            angle_name = angle_names.get(i, f"Angle {i}")
                            if angle_name in ["left_wrist_angle_bk", "right_wrist_angle_bk"] and last_predicted_pose != "BaddhaKonasana":
                                continue  # Skip these angles unless the pose is BaddhaKonasana
                            if angle_name == "neck_angle_uk" and last_predicted_pose != "UtkataKonasana":
                                continue  # Skip this angle unless the pose is UtkataKonasana
                            
                            if angle_diff < 0:
                                feedback.append(f"Decrease angle of {angle_name} by {abs(angle_diff):.2f} degrees")
                            else:
                                feedback.append(f"Increase angle of {angle_name} by {abs(angle_diff):.2f} degrees")
                else:
                    last_predicted_pose = "No pose detected"
                    last_confidence = 0.0
                    last_angle_differences = None
                    feedback = []
            else:
                last_predicted_pose = "Body not fully visible"
                last_confidence = 0.0
                last_angle_differences = None
                feedback = []

            last_process_time = current_time

    # Place the video feed on the left half of the screen
    display_frame[:, :video_width] = image_resized

    # Display the last predicted pose, confidence, and problematic angles on the right side
    if last_predicted_pose:
        # Pose and Confidence Text
        text = f"Pose: {last_predicted_pose}"
        cv2.putText(display_frame, text, (video_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        confidence_text = f"Confidence: {last_confidence:.2f}"
        cv2.putText(display_frame, confidence_text, (video_width + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display Feedback
        if last_angle_differences is not None and last_confidence >= min_classification_confidence:
            for idx, line in enumerate(feedback):
                cv2.putText(display_frame, line, (video_width + 10, 90 + idx * 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif last_confidence < min_classification_confidence:
            cv2.putText(display_frame, "No pose detected", (video_width + 10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the ideal image for the detected pose
        if last_predicted_pose in ideal_images and ideal_images[last_predicted_pose] is not None:
            ideal_image = ideal_images[last_predicted_pose]
            ideal_image_resized = cv2.resize(ideal_image, (video_width, video_height // 2))
            
            # Place ideal image in the bottom right quadrant
            display_frame[video_height//2:, video_width:] = ideal_image_resized

    return display_frame

is_image = os.path.isfile(args.source) and args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

if is_image:
    image = cv2.imread(args.source)
    if image is None:
        print(f"Error: Unable to read image '{args.source}'")
    else:
        last_process_time = 0.0
        display_frame = process_frame(image)
        cv2.imshow('Yoga Pose Prediction', display_frame)
        print("Prediction complete. Displaying window (Press any key or 'q' to close).")
        cv2.waitKey(0)
else:
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera or video source '{args.source}'.")
    else:
        print("Starting video/webcam stream (Press 'q' to exit)...")
        consecutive_empty = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                consecutive_empty += 1
                if consecutive_empty > 30:
                    print("Stream ended or no frames received.")
                    break
                continue
            consecutive_empty = 0

            display_frame = process_frame(image)
            cv2.imshow('Yoga Pose Prediction', display_frame)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()

cv2.destroyAllWindows()