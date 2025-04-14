# ============================
# Step 0: Imports & Setup
# ============================
import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

RGB_DIR = "/content/drive/MyDrive/Dataset with No HPE"

HPE_DIR = "/content/drive/MyDrive/Dataset_Shuffled"

classes = sorted([d for d in os.listdir(RGB_DIR) if os.path.isdir(os.path.join(RGB_DIR, d))])
print("Found classes:", classes)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=False,
    brightness_range=[0.3, 1.5]
)

def apply_same_transform(img1, img2, datagen):
    """
    Apply the same random transformation to both img1 and img2.
    Returns (aug_img1, aug_img2) in float, but we convert to [0..255] uint8 afterward.
    """
    if img1 is None or img2 is None:
        return None, None

    transform_params = datagen.get_random_transform(img1.shape)

    aug_img1 = datagen.apply_transform(img1, transform_params)
    aug_img2 = datagen.apply_transform(img2, transform_params)

    aug_img1 = np.clip(aug_img1, 0, 255).astype(np.uint8)
    aug_img2 = np.clip(aug_img2, 0, 255).astype(np.uint8)

    return aug_img1, aug_img2

# ---------------------------
# Step 1: Preprocessing for Raw RGB Images
# ---------------------------
def preprocess_hand_outline(img_bgr):
    """
    1) Convert to grayscale.
    2) Apply median blur to reduce noise.
    3) Apply Otsu's threshold for a binary mask.
    4) Morphological opening & closing to remove small noise and fill holes.
    5) Return the cleaned (binary) image expanded to 3 channels for MobileNet.
    """
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

# ---------------------------
# Step 2: MobileNet Feature Extractor (64D)
# ---------------------------
IMG_SIZE = 224 
base_model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

def get_image_feature(img_bgr):
    """
    Preprocess the hand outline, resize to 224x224,
    run through MobileNet, and get a 64D feature vector.
    """
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

# ---------------------------
# Step 3: Landmark Extraction (42D) for HPE Images
# ---------------------------
LOWER_GREEN = (35, 80, 80)
UPPER_GREEN = (85, 255, 255)
MIN_DIST = 2
THRESH_REL = 0.1

def extract_42d_landmarks_sensitivity(img_bgr,
                                      lower_green=LOWER_GREEN,
                                      upper_green=UPPER_GREEN,
                                      min_dist=MIN_DIST,
                                      thresh_rel=THRESH_REL):
    """
    1) Convert to HSV, mask green.
    2) Distance transform + peak_local_max => watershed => separate overlapping dots.
    3) Compute centroids, pad/truncate to 21 points => flatten to 42D => normalize to [0,1].
    Returns None if extraction fails or image is None.
    """
    if img_bgr is None:
        return None

    h, w, _ = img_bgr.shape
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    local_max_coords = peak_local_max(dist_norm,
                                      min_distance=min_dist,
                                      threshold_rel=thresh_rel,
                                      exclude_border=False)

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

    num_points = len(landmark_points)
    if num_points < 21:
        landmark_points.extend([(0, 0)] * (21 - num_points))
    elif num_points > 21:
        landmark_points.sort(key=lambda p: (p[1], p[0]))
        landmark_points = landmark_points[:21]

    landmark_points.sort(key=lambda p: (p[1], p[0]))

    landmarks_42 = np.array(landmark_points, dtype=np.float32).flatten()
    for i in range(0, len(landmarks_42), 2):
        x_idx, y_idx = i, i+1
        landmarks_42[x_idx] /= w
        landmarks_42[y_idx] /= h

    return landmarks_42

# ---------------------------
# Step 4: Generate 64D + 42D Features with Identical Augmentations
# ---------------------------
rgb_features = {}
landmark_features = {}

print("\nExtracting features with identical data augmentation:")
for i, cls in enumerate(classes):
    print(f"  Processing class {i+1}/{len(classes)}: {cls}")

    rgb_class_dir = os.path.join(RGB_DIR, cls)
    hpe_class_dir = os.path.join(HPE_DIR, cls)

    if not os.path.isdir(hpe_class_dir):
        continue

    rgb_files = os.listdir(rgb_class_dir)
    hpe_files = os.listdir(hpe_class_dir)

    for fname in tqdm(rgb_files, desc=f"Class '{cls}'", leave=False):
        rgb_path = os.path.join(rgb_class_dir, fname)
        hpe_path = os.path.join(hpe_class_dir, fname)

        if not os.path.exists(hpe_path):
            continue

        img_bgr_rgb = cv2.imread(rgb_path)
        img_bgr_hpe = cv2.imread(hpe_path)

        if img_bgr_rgb is None or img_bgr_hpe is None:
            continue

        # 1) Apply the SAME random transform to both
        aug_rgb, aug_hpe = apply_same_transform(img_bgr_rgb, img_bgr_hpe, datagen)
        if aug_rgb is None or aug_hpe is None:
            continue

        # 2) Now extract 64D from the augmented RGB
        feat_64d = get_image_feature(aug_rgb)
        if feat_64d is None:
            continue

        feat_42d = extract_42d_landmarks_sensitivity(aug_hpe)
        if feat_42d is None or np.all(feat_42d == 0):
            continue

        key = f"{cls}/{fname}"
        rgb_features[key] = feat_64d
        landmark_features[key] = feat_42d

print(f"Total augmented RGB features extracted: {len(rgb_features)}")
print(f"Total augmented HPE landmarks extracted: {len(landmark_features)}")

# ---------------------------
# Step 5: Concatenate (64D + 42D = 106D) with Mismatch Check
# ---------------------------
data_pairs = []
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

for key, feat_64d in rgb_features.items():
    if key in landmark_features:
        feat_42d = landmark_features[key]
        combined_106 = np.concatenate([feat_64d, feat_42d], axis=0)
        cls_name = key.split('/')[0]
        label = class_to_idx[cls_name]
        data_pairs.append((combined_106, label))

random.shuffle(data_pairs)
features, labels = zip(*data_pairs)
features = np.array(features, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print(f"\nFinal fused dataset size: {len(features)} (106D vectors)")

# ---------------------------
# Step 7: Model Training (on 106D)
# ---------------------------
num_classes = len(classes)
model = Sequential([
    Dense(128, activation='relu', input_shape=(106,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nTraining the model on 106D fused features...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    verbose=1
)


# # ---------------------------
# # Step 8: Save the Model
# # ---------------------------
# final_model_path = "/content/drive/MyDrive/Models/sign_language_fused_aug_model.h5"
# model.save(final_model_path)
# print(f"✅ Final Model saved at {final_model_path}")

# ---------------------------
# Step 9: Accuracy Checks
# ---------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes))
