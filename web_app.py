import os
import cv2
import base64
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Load Pre-trained Artifacts
print("Loading Yoga Pose AI Models & Preprocessors...")
model = tf.keras.models.load_model('yoga_pose_model.keras')
scaler = joblib.load('yoga_pose_scaler.joblib')
le = joblib.load('yoga_pose_label_encoder.joblib')
ideal_angles = pd.read_csv('ideal_angles.csv', index_col='Label')

# 2. Setup MediaPipe
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

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

angle_names = [
    "Sol Dirsek", "Sağ Dirsek", "Sol Omuz", "Sağ Omuz",
    "Sol Diz", "Sağ Diz", "Kol/El Hizası", "Sol Kalça",
    "Sağ Kalça", "Boyun", "Sol Bilek", "Sağ Bilek"
]

angle_keys = [
    "left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle",
    "left_knee_angle", "right_knee_angle", "hand_angle", "left_hip_angle",
    "right_hip_angle", "neck_angle_uk", "left_wrist_angle_bk", "right_wrist_angle_bk"
]

sample_map = {
    "ArdhaChandrasana": "dataset2/ArdhaChandrasana/Images/v1.jpeg",
    "BaddhaKonasana": "dataset2/BaddhaKonasana/Images/BK_59.jpg",
    "Downward_dog": "dataset2/Downward_dog/Images/downward_dog5.jpg",
    "Natarajasana": "dataset2/Natarajasana/Images/images (2).jpeg",
    "Triangle": "dataset2/Triangle/Images/triangle1.jpg",
    "UtkataKonasana": "dataset2/UtkataKonasana/Images/UK_25.jpeg",
    "Veerabhadrasana": "dataset2/Veerabhadrasana/Images/images (10).jpeg",
    "Vrukshasana": "dataset2/Vrukshasana/Images/images (1).jpeg"
}

def calculate_angle(l1, l2, l3):
    x1, y1, _ = l1
    x2, y2, _ = l2
    x3, y3, _ = l3
    ang = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    if ang < 0:
        ang += 360
    return ang

def process_image_frame(image, max_variance=18.0, min_conf=0.65):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    
    if not results.pose_landmarks:
        return {"no_pose": True, "message": "Görüntüde vücut duruşu algılanamadı."}
        
    landmarks = results.pose_landmarks.landmark
    
    angles = []
    visibilities = []
    for pair in angle_pair_indices:
        avg_vis = (landmarks[pair[0].value].visibility + 
                   landmarks[pair[1].value].visibility + 
                   landmarks[pair[2].value].visibility) / 3.0
        visibilities.append(avg_vis)
        
        a = calculate_angle(
            (landmarks[pair[0].value].x, landmarks[pair[0].value].y, landmarks[pair[0].value].z),
            (landmarks[pair[1].value].x, landmarks[pair[1].value].y, landmarks[pair[1].value].z),
            (landmarks[pair[2].value].x, landmarks[pair[2].value].y, landmarks[pair[2].value].z)
        )
        angles.append(a)
        
    avg_visibility = float(np.mean(visibilities))
    if avg_visibility < 0.55:
        return {"no_pose": True, "message": f"Vücut kadraja tam sığmıyor (Görünürlük: %{avg_visibility*100:.0f})."}

    # ML Inference
    input_data = np.array(angles).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled, verbose=0)
    pred_idx = np.argmax(pred)
    predicted_pose = le.inverse_transform([pred_idx])[0]
    confidence = float(pred[0][pred_idx])

    # Annotate Frame
    annotated = image.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
    )

    # Convert annotated image to base64
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64_img = base64.b64encode(buf).decode('utf-8')

    # Posture Feedback
    ideal_row = ideal_angles.loc[predicted_pose].values
    diffs = input_data[0] - ideal_row

    feedback = []
    chart_labels = []
    chart_curr = []
    chart_ideal = []

    for i in range(len(diffs)):
        k_name = angle_keys[i]
        d_name = angle_names[i]
        curr_val = round(float(input_data[0][i]), 1)
        ideal_val = round(float(ideal_row[i]), 1)
        diff_val = round(float(diffs[i]), 1)

        # Pose specific angle filter
        if k_name in ["left_wrist_angle_bk", "right_wrist_angle_bk"] and predicted_pose != "BaddhaKonasana":
            continue
        if k_name == "neck_angle_uk" and predicted_pose != "UtkataKonasana":
            continue

        chart_labels.append(d_name)
        chart_curr.append(curr_val)
        chart_ideal.append(ideal_val)

        if abs(diff_val) > max_variance and confidence >= min_conf:
            if diff_val < 0:
                action = "increase"
                msg = f"{abs(diff_val):.1f}° daha açın"
            else:
                action = "decrease"
                msg = f"{abs(diff_val):.1f}° daraltın"

            feedback.append({
                "joint": d_name,
                "action": action,
                "diff": diff_val,
                "message": msg
            })

    return {
        "no_pose": False,
        "pose": predicted_pose,
        "confidence": confidence,
        "ideal_image_url": f"/api/ideal_image/{predicted_pose}",
        "feedback": feedback,
        "chart_data": {
            "labels": chart_labels[:8],
            "current": chart_curr[:8],
            "ideal": chart_ideal[:8]
        },
        "annotated_image": b64_img
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ideal_image/<pose_name>')
def get_ideal_image(pose_name):
    folder = 'ideal_images'
    filename = f"{pose_name}.jpg"
    return send_from_directory(folder, filename)

@app.route('/api/sample/<pose_name>')
def get_sample(pose_name):
    if pose_name not in sample_map:
        return jsonify({"error": "Geçersiz poz adı"}), 400
    
    path = sample_map[pose_name]
    if not os.path.exists(path):
        return jsonify({"error": "Görsel dosyası bulunamadı"}), 404
        
    img = cv2.imread(path)
    result = process_image_frame(img)
    return jsonify(result)

@app.route('/api/predict_upload', methods=['POST'])
def predict_upload():
    if 'image' not in request.files:
        return jsonify({"error": "Görsel yüklenmedi"}), 400
        
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Geçersiz görsel formatı"}), 400
        
    result = process_image_frame(img)
    return jsonify(result)

@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Görsel verisi eksik"}), 400
        
    try:
        raw_b64 = data['image']
        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Görsel çözülemedi"}), 400
            
        result = process_image_frame(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f"\n🚀 YogaAI Web Sunucusu Başlatılıyor: http://127.0.0.1:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)

