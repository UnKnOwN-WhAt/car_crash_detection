import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Base folder path
data_folder = 'data'

# Function to extract features using a simple image resizing and flattening approach
def extract_features(image_path, size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size)
    return img.flatten()

# Function to prepare data
def prepare_data(folder):
    X, y = [], []
    labels = {'Accident': 1, 'Non Accident': 0}
    for label, class_num in labels.items():
        folder_path = os.path.join(folder, label)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(class_num)
    return np.array(X), np.array(y)

# Prepare training data
train_path = os.path.join(data_folder, 'train')
print("Preparing training data...")
X_train, y_train = prepare_data(train_path)

# Prepare validation (test) data
test_path = os.path.join(data_folder, 'test')
print("Preparing validation data...")
X_test, y_test = prepare_data(test_path)

# Train SVM model
print("Training SVM model...")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(svm_model, 'svm_accident_detection_model.pkl')
print("Model saved as 'svm_accident_detection_model.pkl'")

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
