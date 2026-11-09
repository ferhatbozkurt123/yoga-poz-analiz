import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from PIL import Image

st.set_page_config(
    page_title="Yoga Pose Estimation & Feedback",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .feedback-box-good {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .feedback-box-warn {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('yoga_pose_model.keras')
    scaler = joblib.load('yoga_pose_scaler.joblib')
    le = joblib.load('yoga_pose_label_encoder.joblib')
    ideal_angles = pd.read_csv('ideal_angles.csv', index_col='Label')
    
    ideal_image_folder = 'ideal_images'
    ideal_images = {}
    for pose_name in ideal_angles.index:
        img_p = os.path.join(ideal_image_folder, f"{pose_name}.jpg")
        if os.path.exists(img_p):
            ideal_images[pose_name] = cv2.imread(img_p)
            
    return model, scaler, le, ideal_angles, ideal_images

model, scaler, le, ideal_angles, ideal_images = load_models()

mp_pose = mp.solutions.pose
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

angle_names = {
    0: "Sol Dirsek Açısı (Left Elbow)",
    1: "Sağ Dirsek Açısı (Right Elbow)",
    2: "Sol Omuz Açısı (Left Shoulder)",
    3: "Sağ Omuz Açısı (Right Shoulder)",
    4: "Sol Diz Açısı (Left Knee)",
    5: "Sağ Diz Açısı (Right Knee)",
    6: "El/Kol Hizası Açısı (Hand Angle)",
    7: "Sol Kalça Açısı (Left Hip)",
    8: "Sağ Kalça Açısı (Right Hip)",
    9: "Boyun Açısı (Neck Angle)",
    10: "Sol Bilek Açısı (Left Wrist)",
    11: "Sağ Bilek Açısı (Right Wrist)"
}

angle_keys = [
    "left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle",
    "left_knee_angle", "right_knee_angle", "hand_angle", "left_hip_angle",
    "right_hip_angle", "neck_angle_uk", "left_wrist_angle_bk", "right_wrist_angle_bk"
]

def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def extract_angles(landmarks):
    angles = []
    all_visibilities = []
    for pair in angle_pair_indices:
        avg_vis = (landmarks[pair[0].value].visibility + 
                   landmarks[pair[1].value].visibility + 
                   landmarks[pair[2].value].visibility) / 3
        all_visibilities.append(avg_vis)
        
        angle = calculate_angle(
            (landmarks[pair[0].value].x, landmarks[pair[0].value].y, landmarks[pair[0].value].z),
            (landmarks[pair[1].value].x, landmarks[pair[1].value].y, landmarks[pair[1].value].z),
            (landmarks[pair[2].value].x, landmarks[pair[2].value].y, landmarks[pair[2].value].z)
        )
        angles.append(angle)
        
    return angles, np.mean(all_visibilities)

# Sidebar
st.sidebar.title("🧘 Ayarlar & Test Modu")

mode = st.sidebar.radio(
    "Girdi Yöntemi Seçin:",
    ["📁 Hazır Örnek Pozlar", "📤 Kendi Fotoğrafını Yükle", "📸 Web Kamerasından Çek"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Hassasiyet Ayarları")
min_conf = st.sidebar.slider("Min. Model Güven Eşiği (%)", 50, 95, 65, 5) / 100.0
max_variance = st.sidebar.slider("Maks. İzin Verilen Açı Sapması (°)", 5, 30, 18, 1)

sample_dict = {
    "ArdhaChandrasana (Yarım Ay)": "dataset2/ArdhaChandrasana/Images/v1.jpeg",
    "BaddhaKonasana (Kelebek)": "dataset2/BaddhaKonasana/Images/BK_59.jpg",
    "Downward Dog (Aşağı Bakan Köpek)": "dataset2/Downward_dog/Images/downward_dog5.jpg",
    "Natarajasana (Dansçı)": "dataset2/Natarajasana/Images/images (2).jpeg",
    "Triangle (Üçgen)": "dataset2/Triangle/Images/triangle1.jpg",
    "UtkataKonasana (Tanrıça)": "dataset2/UtkataKonasana/Images/UK_25.jpeg",
    "Veerabhadrasana (Savaşçı)": "dataset2/Veerabhadrasana/Images/images (10).jpeg",
    "Vrukshasana (Ağaç)": "dataset2/Vrukshasana/Images/images (1).jpeg"
}

input_image = None

if mode == "📁 Hazır Örnek Pozlar":
    chosen_pose = st.sidebar.selectbox("Test Edilecek Yoga Pozunu Seçin:", list(sample_dict.keys()))
    sample_path = sample_dict[chosen_pose]
    if os.path.exists(sample_path):
        input_image = cv2.imread(sample_path)
    else:
        st.error(f"Görsel bulunamadı: {sample_path}")

elif mode == "📤 Kendi Fotoğrafını Yükle":
    uploaded_file = st.sidebar.file_uploader("Bir yoga pozu fotoğrafı yükleyin", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)

elif mode == "📸 Web Kamerasından Çek":
    camera_photo = st.camera_input("Kameraya doğru bir yoga pozu verin ve fotoğraf çekin")
    if camera_photo is not None:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)

# Main UI
st.markdown('<div class="main-title">🧘 AI Yoga Poz Analizi & Postür Geri Bildirim Sistemi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">MediaPipe İskelet Takibi ve Derin Öğrenme ile Anlık Yoga Pozu Tespiti ve Açı Düzeltme Rehberi</div>', unsafe_allow_html=True)

if input_image is not None:
    # Pose Detection
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2) as pose:
        img_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        annotated_image = input_image.copy()
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=3)
            )
            
            angles, avg_vis = extract_angles(results.pose_landmarks.landmark)
            
            if avg_vis < 0.6:
                st.warning(f"⚠️ Vücut yeterince görünür değil (Görünürlük: %{avg_vis*100:.1f}). Lütfen tüm vücudun kadraja girmesini sağlayın.")
            else:
                # Prediction
                input_data = np.array(angles).reshape(1, -1)
                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled, verbose=0)
                pred_idx = np.argmax(pred)
                pred_pose = le.inverse_transform([pred_idx])[0]
                confidence = float(pred[0][pred_idx])
                
                # Visuals Side by Side
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("🎯 Analiz Edilen Poz (İskelet Takibi)")
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                with col_right:
                    st.subheader("🧘 İdeal / Referans Duruş")
                    if pred_pose in ideal_images and ideal_images[pred_pose] is not None:
                        st.image(cv2.cvtColor(ideal_images[pred_pose], cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.info("Bu poz için referans görsel bulunamadı.")
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(label="Algılanan Poz", value=pred_pose)
                with m2:
                    conf_pct = int(confidence * 100)
                    st.metric(label="Model Güven Skoru", value=f"%{conf_pct}")
                with m3:
                    if confidence >= min_conf:
                        st.metric(label="Durum", value="✅ Tespit Edildi")
                    else:
                        st.metric(label="Durum", value="⚠️ Güven Eşiği Altında")
                
                st.progress(confidence if confidence <= 1.0 else 1.0)
                
                if confidence >= min_conf:
                    # Angle Differences and Feedback
                    ideal_row = ideal_angles.loc[pred_pose].values
                    diffs = input_data[0] - ideal_row
                    
                    st.markdown("---")
                    st.subheader("💡 Postür Düzeltme & Açı Geri Bildirimleri")
                    
                    feedback_items = []
                    table_rows = []
                    
                    for i, diff in enumerate(diffs):
                        key_name = angle_keys[i]
                        display_name = angle_names[i]
                        curr_angle = input_data[0][i]
                        ideal_angle = ideal_row[i]
                        abs_diff = abs(diff)
                        
                        # Filtering specific pose angles
                        if key_name in ["left_wrist_angle_bk", "right_wrist_angle_bk"] and pred_pose != "BaddhaKonasana":
                            continue
                        if key_name == "neck_angle_uk" and pred_pose != "UtkataKonasana":
                            continue
                            
                        status = "✅ İdeal" if abs_diff <= max_variance else "⚠️ Düzeltilmeli"
                        
                        table_rows.append({
                            "Eklem / Bölge": display_name,
                            "Mevcut Açı (°)": f"{curr_angle:.1f}°",
                            "İdeal Açı (°)": f"{ideal_angle:.1f}°",
                            "Fark (°)": f"{diff:+.1f}°",
                            "Durum": status
                        })
                        
                        if abs_diff > max_variance:
                            if diff < 0:
                                direction = f"🔺 **{display_name}** açısını **{abs_diff:.1f}° artırın** (Genişletin)"
                            else:
                                direction = f"🔻 **{display_name}** açısını **{abs_diff:.1f}° daraltın** (Kapatın)"
                            feedback_items.append(direction)
                    
                    fb_col1, fb_col2 = st.columns([1, 1])
                    
                    with fb_col1:
                        if not feedback_items:
                            st.success("🎉 **Tebrikler!** Duruşunuz referans modelle harika bir uyum içinde. Tüm eklem açılarınız ideal sınırlar içinde.")
                        else:
                            st.warning(f"**{len(feedback_items)} Bölgede Düzeltme Önerisi:**")
                            for item in feedback_items:
                                st.markdown(f"- {item}")
                                
                    with fb_col2:
                        # Angle Comparison Chart
                        df_table = pd.DataFrame(table_rows)
                        st.dataframe(df_table, use_container_width=True, hide_index=True)
                        
                    # Visual Angle Bar Chart
                    st.markdown("---")
                    st.subheader("📊 Eklem Açıları Karşılaştırması (Mevcut vs İdeal)")
                    chart_data = pd.DataFrame({
                        "Eklem": [r["Eklem / Bölge"].split(" (")[0] for r in table_rows],
                        "Mevcut Açı": [float(r["Mevcut Açı (°)"].replace("°", "")) for r in table_rows],
                        "İdeal Açı": [float(r["İdeal Açı (°)"].replace("°", "")) for r in table_rows]
                    }).set_index("Eklem")
                    st.bar_chart(chart_data)
                    
                else:
                    st.error("Görüntüdeki poz desteklenen 8 yoga pozundan birine yeterli güvenilirlikle benzetilemedi. Lütfen daha net bir duruş deneyin.")
        else:
            st.error("❌ Fotoğrafta bir insan vücudu veya duruş iskeleti algılanamadı. Lütfen kadrajda net ve tam boy görünen bir görsel seçin.")
else:
    st.info("👈 Başlamak için sol menüden hazır bir örnek poz seçin veya kendi fotoğrafınızı yükleyin.")

