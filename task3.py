import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset path
dataset_path = r"kagglecatsanddogs_5340\PetImages"

data = []
labels = []

# Load images
for category in ["Cat", "Dog"]:
    path = os.path.join(dataset_path, category)
    label = 0 if category == "Cat" else 1

    for img_name in os.listdir(path)[:500]:  # Use 500 images per class
        try:
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))

            data.append(img.flatten())
            labels.append(label)

        except:
            pass

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Total Images Loaded:", len(data))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

# Train SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display sample predictions
plt.figure(figsize=(12, 4))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.title("Cat" if y_pred[i] == 0 else "Dog")
    plt.axis('off')

plt.tight_layout()
plt.show()