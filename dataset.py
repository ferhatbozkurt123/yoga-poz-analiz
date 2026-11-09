import math
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    
    height, width, _ = image.shape
    landmarks = []
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=output_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
    
    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    if angle < 0:
        angle += 360
    
    return angle

def angles_finder(landmarks):
    angles = []
    angle_pairs = [
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
    
    for pair in angle_pairs:
        angle = calculateAngle(landmarks[pair[0].value], landmarks[pair[1].value], landmarks[pair[2].value])
        angles.append(angle)
    
    return angles

df = pd.DataFrame(columns=['Label', 'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 
                           'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle', 
                        'hand_angle', 'left_hip_angle', 'right_hip_angle', 'neck_angle_uk', 
                        'left_wrist_angle_bk', 'right_wrist_angle_bk'])

# dataset = pd.read_csv('dataset1/train.csv')
# for _, row in dataset.iterrows():
#     print(row['image_id'])
#     img = cv2.imread("dataset1/images/train_images/" + row['image_id'])
#     label = row['class_6']
#     output_image, landmarks = detectPose(img, pose, display=False)
#     if landmarks:
#         angles = angles_finder(landmarks)
#         new_row = {'Id' : row['image_id'], 'Label': label, **dict(zip(df.columns[1:], angles))}
#         df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
# print(df.head())
# df.to_csv("results.csv", index=False)

train_path = "datset2"
for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)
    if os.path.isdir(folder_path):
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            print(f"Processing: {folder} -> {img_name}")
            img = cv2.imread(img_path)
            label = folder
            output_image, landmarks = detectPose(img, pose, display=False)
            if landmarks:
                angles = angles_finder(landmarks)
                id = label + "_" + img_name
                new_row = {'Label': label, **dict(zip(df.columns[1:], angles)), 'Id': id}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                print(f"No landmarks detected for {img_name}")
df.to_csv("results2.csv", index=False)

def process_image(path):
    img = cv2.imread(path)
    output_image, landmarks = detectPose(img, pose, display=False)
    if landmarks:
        angles = angles_finder(landmarks)
        print(angles)

# image_path = "datset2/ArdhaChandrasana/Images/4.png"
# process_image(image_path)