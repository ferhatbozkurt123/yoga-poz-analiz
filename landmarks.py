import os
import cv2
import mediapipe as mp

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

image_paths = ["dataset2/ArdhaChandrasana/Images/v1.jpeg", "dataset2/BaddhaKonasana/Images/BK_59.jpg", "dataset2/Downward_dog/Images/downward_dog5.jpg", "dataset2/Natarajasana/Images/images (2).jpeg", "dataset2/Triangle/Images/triangle1.jpg", "dataset2/UtkataKonasana/Images/UK_25.jpeg", "dataset2/Veerabhadrasana/Images/images (10).jpeg", "dataset2/Vrukshasana/Images/images (1).jpeg"]

for image_path in image_paths:
    sample_img = cv2.imread(image_path)

    # Perform pose detection after converting the image into RGB format.
    results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

    # Create a copy of the sample image to draw landmarks on.
    img_copy = sample_img.copy()

    # Check if any landmarks are found.
    if results.pose_landmarks:
        # Draw Pose landmarks on the sample image.
        mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        
        # Create the 'ideal_images' folder if it doesn't exist
        ideal_images_folder = 'ideal_images'
        if not os.path.exists(ideal_images_folder):
            os.makedirs(ideal_images_folder)
        
        path_parts = image_path.replace('\\', '/').split('/')
        pose_name = next(part for part in path_parts if 'dataset' not in part.lower() and 'images' not in part.lower())
        
        # Save the image with landmarks to the 'ideal_images' folder
        output_image_path = os.path.join(ideal_images_folder, f"{pose_name}.jpg")
        cv2.imwrite(output_image_path, img_copy)
        
        # Display the output image with the landmarks drawn, also convert BGR to RGB for display. 
        cv2.imshow('Output', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Release mediapipe pose instance
pose.close()