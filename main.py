# pattern_model.py
# -*- coding: utf-8 -*-

import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from imutils import paths
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from patoolib import extract_archive


def unzip_data():
    zipfile = 'ckplus.zip'
    extract_to = 'ckplus'
    if not os.path.exists(extract_to):
        os.mkdir(extract_to)
        extract_archive(zipfile, outdir=extract_to)


def colortogray(im):
    image = cv2.imread(im)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imgray


def resizeImage(image, size):
    return cv2.resize(image, (size, size))


def feat_lab(imagePaths):
    features = []
    labels = []

    for imagePath in imagePaths:
        im = colortogray(imagePath)
        im = resizeImage(im, 64)
        fd1 = hog(im, orientations=7, pixels_per_cell=(8, 8),
                  cells_per_block=(4, 4), block_norm='L2-Hys', transform_sqrt=False)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        features.append(fd1)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = classification_report(y_train, pred)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n{confusion_matrix(y_train, pred)}\n")
    else:
        pred = clf.predict(X_test)
        clf_report = classification_report(y_test, pred)
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, pred)}\n")


def main():
    unzip_data()

    imagePaths = list(paths.list_images('ckplus/CK+48'))

    # Display first 10 images
    fig = plt.figure(figsize=(20, 20))
    for i in range(10):
        fig.add_subplot(1, 10, i + 1)
        plt.imshow(np.array(cv2.imread(imagePaths[i])), cmap='gray')
        label = imagePaths[i].split(os.path.sep)[-2]
        plt.title(label)
    plt.show()

    features, labels = feat_lab(imagePaths)

    print("[INFO] The number of features ...", features.shape[1])
    print("[INFO] The number of samples ...", labels.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    print("[INFO] The number of images used in training ...", X_train.shape[0])
    print("[INFO] The number of images used in testing ...", X_test.shape[0])

    svm_clf = SVC(kernel='rbf', gamma='scale', C=10)
    svm_clf.fit(X_train, y_train)

    print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)

    
    # Save the model
    joblib.dump(svm_clf, 'svm_model.joblib')

    # Load it later
    svm_clf_loaded = joblib.load('svm_model.joblib')



if __name__ == '__main__':
    main()

