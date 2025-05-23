import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# === Set your image path here ===
img_path = os.path.expanduser('./ckplus/ck/CK+48/happy/S010_006_00000014.png')


# === Load the trained SVM model ===
model = joblib.load('svm_model.joblib')

# === Preprocessing functions ===
def colortogray(im):
    image = cv2.imread(im)
    if image is None:
        raise FileNotFoundError(f"Image not found: {im}")
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imgray

def resizeImage(image, size):
    return cv2.resize(image, (size, size))

def extract_features(image_path):
    im = colortogray(image_path)
    im = resizeImage(im, 64)
    features = hog(im, orientations=7, pixels_per_cell=(8, 8),
                   cells_per_block=(4, 4), block_norm='L2-Hys', transform_sqrt=False)
    return features.reshape(1, -1)  # Reshape to (1, N) for prediction

# === Run prediction ===
try:
    features = extract_features(img_path)
    prediction = model.predict(features)
    print(f"[RESULT] The predicted facial expression is: {prediction[0]}")
except Exception as e:
    print(f"[ERROR] {e}")

