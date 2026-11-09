import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def angles_finder(landmarks, angle_pair_indices):
    angles = []
    all_visibilities = []
    for pair in angle_pair_indices:
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
    
    if np.mean(all_visibilities) < 0.6:
        return None, None
        
    return angles, all_visibilities

def predict_yoga_pose(angles, model, scaler, le, ideal_angles):
    input_data = np.array(angles).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)
    
    predicted_class_index = np.argmax(prediction)
    predicted_class = le.inverse_transform([predicted_class_index])[0]
    confidence = prediction[0][predicted_class_index]
    
    angle_differences = input_data - ideal_angles.loc[predicted_class].values
    return predicted_class, confidence, angle_differences

def run_tests():
    print("=" * 60)
    print("YOGA POSE ESTIMATION - AUTOMATED VERIFICATION SUITE")
    print("=" * 60)
    
    # 1. Verify model and artifacts
    print("\n[Step 1] Loading models and preprocessors...")
    model = tf.keras.models.load_model('yoga_pose_model.keras')
    scaler = joblib.load('yoga_pose_scaler.joblib')
    le = joblib.load('yoga_pose_label_encoder.joblib')
    ideal_angles = pd.read_csv('ideal_angles.csv', index_col='Label')
    print("-> Model loaded successfully.")
    print("-> Scaler and Label Encoder loaded successfully.")
    print(f"-> Classes: {list(le.classes_)}")
    print(f"-> Ideal angles table shape: {ideal_angles.shape}")

    # 2. Setup MediaPipe
    print("\n[Step 2] Initializing MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
    
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

    # 3. Test on sample images from dataset
    test_cases = [
        ("ArdhaChandrasana", "dataset2/ArdhaChandrasana/Images/v1.jpeg"),
        ("BaddhaKonasana", "dataset2/BaddhaKonasana/Images/BK_59.jpg"),
        ("Downward_dog", "dataset2/Downward_dog/Images/downward_dog5.jpg"),
        ("Natarajasana", "dataset2/Natarajasana/Images/images (2).jpeg"),
        ("Triangle", "dataset2/Triangle/Images/triangle1.jpg"),
        ("UtkataKonasana", "dataset2/UtkataKonasana/Images/UK_25.jpeg"),
        ("Veerabhadrasana", "dataset2/Veerabhadrasana/Images/images (10).jpeg"),
        ("Vrukshasana", "dataset2/Vrukshasana/Images/images (1).jpeg")
    ]

    print("\n[Step 3] Testing pose classification and angle feedback on sample images:")
    passed_count = 0
    total_count = len(test_cases)
    
    for expected_pose, img_path in test_cases:
        if not os.path.exists(img_path):
            print(f"❌ File not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if not results.pose_landmarks:
            print(f"❌ {expected_pose}: No landmarks detected in {img_path}")
            continue
            
        angles, visibilities = angles_finder(results.pose_landmarks.landmark, angle_pair_indices)
        if angles is None:
            print(f"⚠️ {expected_pose}: Visibility too low (< 0.6)")
            continue
            
        predicted_pose, confidence, angle_differences = predict_yoga_pose(
            angles, model, scaler, le, ideal_angles
        )
        
        is_correct = (predicted_pose == expected_pose)
        if is_correct:
            passed_count += 1
            status = "✅ PASS"
        else:
            status = f"⚠️ MISMATCH (Expected {expected_pose})"
            
        print(f"\n{status} -> {img_path}")
        print(f"   Predicted Pose: {predicted_pose} (Confidence: {confidence * 100:.1f}%)")
        
        # Test feedback calculation
        max_angle_variance = 18
        feedbacks = []
        for i, angle_diff in enumerate(angle_differences[0]):
            if abs(angle_diff) > max_angle_variance:
                a_name = angle_names.get(i, f"Angle {i}")
                if a_name in ["left_wrist_angle_bk", "right_wrist_angle_bk"] and predicted_pose != "BaddhaKonasana":
                    continue
                if a_name == "neck_angle_uk" and predicted_pose != "UtkataKonasana":
                    continue
                action = "Decrease" if angle_diff < 0 else "Increase"
                feedbacks.append(f"{action} {a_name} by {abs(angle_diff):.1f}°")
        
        if feedbacks:
            print(f"   Feedback items ({len(feedbacks)}): {', '.join(feedbacks[:3])}{'...' if len(feedbacks) > 3 else ''}")
        else:
            print("   Pose alignment: Perfect (within variance limits)")

    pose.close()
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed_count}/{total_count} images correctly predicted.")
    print("=" * 60)
    assert passed_count >= 6, f"Expected at least 6/8 passes, got {passed_count}"
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY! 🎉")

if __name__ == '__main__':
    run_tests()

