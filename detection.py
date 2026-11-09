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
    
    # Compare angles to ideal angles
    adjust_angle = np.vectorize(lambda x: x if x <= 180 else 180 - x)
    input_data = adjust_angle(input_data)
    angle_differences = input_data - ideal_angles.loc[predicted_class].values
    angle_differences = adjust_angle(angle_differences)
    
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

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Yoga Pose Prediction', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Yoga Pose Prediction', 1080, 720)
# Set the interval for processing frames (e.g., every 1 second)
process_interval = .5 # in seconds
last_process_time = time.time()

last_predicted_pose = ""
last_confidence = 0.0
last_angle_differences = None

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

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(image_rgb)

    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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
                last_angle_differences = sorted(last_angle_differences, key=abs, reverse=True)
                print(last_angle_differences[0],last_predicted_pose)
            else:
                last_predicted_pose = "Body not fully visible"
                last_confidence = 0.0
                last_angle_differences = None

            last_process_time = current_time

    # Display the last predicted pose, confidence, and problematic angles
    if last_predicted_pose:
        text = f"Pose: {last_predicted_pose} | Confidence: {last_confidence:.2f}"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if last_angle_differences is not None:
            # Get the indices of the top 3 problematic angles by absolute magnitude
            problematic_indices = np.argsort(np.abs(last_angle_differences[0]))[::-1] # Get indices of the top 3 angles
            text_lines = ["Problematic Angles:"]  # Use a list to store each line

            for i in problematic_indices:
                angle_name = angle_names.get(i, f"Angle {i}")

                # Show specific angles only in their respective poses
                if (angle_name in ["left_wrist_angle_bk", "right_wrist_angle_bk"] and last_predicted_pose != "BaddhaKonasana"):
                    continue  # Skip these angles unless the pose is BaddhaKonasana
                if (angle_name == "neck_angle_uk" and last_predicted_pose != "UtkataKonasana"):
                    continue  # Skip this angle unless the pose is UtkataKonasana
                
                # Append the angle name and its difference to the list
                text_lines.append(f"{angle_name} ({last_angle_differences[0][i]:.2f})")
            
            # Render each line of text on the image
            for idx, line in enumerate(text_lines):
                cv2.putText(image, line, (10, 60 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Increment the Y position for each line


    # Display the image
    cv2.imshow('Yoga Pose Prediction', image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()