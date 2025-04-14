import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model

# ---------------------------
# Configuration & Constants
# ---------------------------
sign_names = ['Hello', 'Hungry', 'I_am', 'Love', 'Money', 'Mother',
              'Okay', 'Please', 'Sorry', 'Stop', 'Water', 'Where', 'Why', 'Yes']

IMG_SIZE = 224      

# ---------------------------
# 1. Setup MobileNet for 64D Feature Extraction
# ---------------------------
base_model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# ---------------------------
# 2. Load the Pre-trained Fused Model (expects 106D input)
# ---------------------------
fused_model_path = "/content/drive/MyDrive/Models/sign_language_fused_model.h5"
fused_model = load_model(fused_model_path)

# ---------------------------
# 3. Setup MediaPipe Hands for Landmark Detection
# ---------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# 4. Hand Outline Preprocessing (for the 64D CNN branch)
# ---------------------------
def preprocess_hand_outline(img_bgr):
    if img_bgr is None:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    processed_3ch = cv2.merge([closed, closed, closed])
    return processed_3ch

def get_image_feature(img_bgr):
    if img_bgr is None:
        return None
    processed = preprocess_hand_outline(img_bgr)
    if processed is None:
        return None
    resized = cv2.resize(processed, (IMG_SIZE, IMG_SIZE))
    resized = resized.astype('float32')
    resized = preprocess_input(resized)
    img_batch = np.expand_dims(resized, axis=0)
    feat = feature_extractor.predict(img_batch, verbose=0)
    return feat.flatten()


LOWER_GREEN = (35, 80, 80)
UPPER_GREEN = (85, 255, 255)
MIN_DIST = 2
THRESH_REL = 0.1

def extract_42d_landmarks_sensitivity(img_bgr,
                                      lower_green=LOWER_GREEN,
                                      upper_green=UPPER_GREEN,
                                      min_dist=MIN_DIST,
                                      thresh_rel=THRESH_REL):
    if img_bgr is None:
        return None
    h, w, _ = img_bgr.shape
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    local_max_coords = peak_local_max(dist_norm, min_distance=min_dist, threshold_rel=thresh_rel, exclude_border=False)
    markers = np.zeros_like(dist_transform, dtype=np.int32)
    for i, (r, c) in enumerate(local_max_coords, start=1):
        markers[r, c] = i
    labels = watershed(-dist_norm, markers, mask=mask)
    landmark_points = []
    for label_val in np.unique(labels):
        if label_val == 0:
            continue
        label_mask = (labels == label_val).astype(np.uint8)
        M = cv2.moments(label_mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            landmark_points.append((cx, cy))
    if len(landmark_points) < 21:
        landmark_points.extend([(0, 0)] * (21 - len(landmark_points)))
    elif len(landmark_points) > 21:
        landmark_points.sort(key=lambda p: (p[1], p[0]))
        landmark_points = landmark_points[:21]
    landmark_points.sort(key=lambda p: (p[1], p[0]))
    landmarks_42 = np.array(landmark_points, dtype=np.float32).flatten()
    for i in range(0, len(landmarks_42), 2):
        landmarks_42[i] /= w
        landmarks_42[i+1] /= h
    return landmarks_42

# ---------------------------
# 6. Test Pipeline on Pre-Captured Image
# ---------------------------
def test_pipeline(image_path):
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print("Error: Image not found.")
        return

    preprocess_input = raw_img.copy()
    cnn_feature = get_image_feature(preprocess_input)
    preprocessed_outline = preprocess_hand_outline(preprocess_input)

    hands_static = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    results = hands_static.process(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))

    annotated_img = raw_img.copy()
    hands_static.close()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

    landmark_feature = extract_42d_landmarks_sensitivity(annotated_img)
    print(landmark_feature)
    fused_feature = np.concatenate([cnn_feature, landmark_feature], axis=0).reshape(1, -1)
    #print(fused_feature)
    preds = fused_model.predict(fused_feature)
    #print(preds)

    class_idx = int(np.argmax(preds))
    predicted_label = sign_names[class_idx]
    print("Predicted class:", predicted_label)
    final_img = raw_img.copy()
    cv2.putText(final_img, f"Prediction: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    plt.title("Raw Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(preprocessed_outline, cv2.COLOR_BGR2RGB))
    plt.title("Preprocessed Outline")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title("HPE Annotated")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title("Prediction: " + predicted_label)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Run the test pipeline on multiple images
# paths = [ "/content/drive/MyDrive/Dataset with No HPE/Why/3.jpg","/content/drive/MyDrive/Dataset Test/Hello/0.jpg", "/content/drive/MyDrive/Dataset Test/Hello/25.jpg", "/content/drive/MyDrive/Dataset Test/Hello/50.jpg",
#          "/content/drive/MyDrive/Dataset Test/Hungry/25.jpg", "/content/drive/MyDrive/Dataset Test/Hungry/2.jpg", "/content/drive/MyDrive/Dataset Test/Hungry/50.jpg",
#          "/content/drive/MyDrive/Dataset Test/Why/25.jpg", "/content/drive/MyDrive/Dataset Test/Why/50.jpg", "/content/drive/MyDrive/Dataset Test/Where/25.jpg",
#          "/content/drive/MyDrive/Dataset Test/Yes/30.jpg", "/content/drive/MyDrive/Dataset Test/Please/40.jpg", "/content/drive/MyDrive/Dataset Test/Mother/33.jpg",
#          "/content/drive/MyDrive/Dataset Test/Stop/1.jpg", "/content/drive/MyDrive/Dataset Test/Stop/25.jpg"]

paths = ["/content/drive/MyDrive/Dataset with No HPE/Yes/100.jpg" ]

for path in paths:
    test_pipeline(path)
