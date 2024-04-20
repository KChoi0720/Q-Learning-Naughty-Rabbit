# Implement a Simple Feedforward Neural Network to classify images using the CIFAR- 10 dataset
import tensorflow as tf
import ssl
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ssl._create_default_https_context = ssl._create_unverified_context

# Loading the data into train and test sets

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 1. Data Preprocessing 

cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

# 2. Model Architecture 

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# 3. Training 
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])


history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))



# 4. Evaluation 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

probability = model.predict(X_test)
# Convert predicted probabilities to class labels
predictions = np.argmax(probability, axis=1)
# Ensure that y_test is a 1D array of integers
y_test = y_test.ravel()

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
F1_score = f1_score(y_test, predictions, average='weighted')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1_score: {F1_score * 100:.2f}%")

# An:
# Accuracy: 46.72%
# Precision: 52.87%
# Recall: 46.72%
# F1_score: 46.19%

# 5. Accuracy-Loss Curve 
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 2.25)
plt.show()


# B Implement a Convolutional Neural Network (CNN) to classify images in the CIFAR-10 dataset 

# 8. Data Preprocessing 

cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

# 9. Model Architecture 
model = keras.models.Sequential([
keras.layers.Conv2D(filters=64, kernel_size=7, strides=(1,1), activation="relu", padding="same", input_shape=[32, 32, 3]),
keras.layers.MaxPooling2D((2,2)),
keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"),
keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Flatten(),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(64, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(10, activation="softmax")
])

# 10. Training 

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adamax",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=15,
                    validation_data=(X_valid, y_valid),verbose=1)


# 11. Evaluation 

probability = model.predict(X_test)
# Convert predicted probabilities to class labels
predictions = np.argmax(probability, axis=1)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Accuracy: 76.00%


# 12. Confusion Matrix 

from sklearn.metrics import confusion_matrix
import seaborn as sns


confusion = confusion_matrix(y_test, predictions)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# Print the confusion Matrix
print("Confusion Matrix:")
print(confusion)


# 13. Accuracy-Loss Curve

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 2.25)
plt.show()


# 14. Learning Rate 

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score

cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

model = keras.models.Sequential([
keras.layers.Conv2D(filters=64, kernel_size=7, strides=(1,1), activation="relu", padding="same", input_shape=[32, 32, 3]),
keras.layers.MaxPooling2D((2,2)),
keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"),
keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Flatten(),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(64, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(10, activation="softmax")
])

optimizer = keras.optimizers.Adamax(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer,
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=15,
                    validation_data=(X_valid, y_valid),verbose=1)
probability = model.predict(X_test)
predictions = np.argmax(probability, axis=1)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy_learning_rate_0.01: {accuracy * 100:.2f}%")

# Accuracy with learning_rate_0.01 is 10.00%


# 15. Number of Epochs 

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score

cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

model = keras.models.Sequential([
keras.layers.Conv2D(filters=64, kernel_size=7, strides=(1,1), activation="relu", padding="same", input_shape=[32, 32, 3]),
keras.layers.MaxPooling2D((2,2)),
keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"),
keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
keras.layers.MaxPooling2D(2),
keras.layers.Flatten(),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(64, activation="relu"),
keras.layers.Dropout(0.5),
keras.layers.Dense(10, activation="softmax")
])


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adamax",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid),verbose=1)
probability = model.predict(X_test)
# Convert predicted probabilities to class labels
predictions = np.argmax(probability, axis=1)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy_epochs_10: {accuracy * 100:.2f}%")


