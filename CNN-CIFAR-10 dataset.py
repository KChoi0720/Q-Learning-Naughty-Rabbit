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


#
#
#
#

from sklearn.tree import DecisionTreeClassifier
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

# Reshape the input data to two dimensions (samples, features)
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_faltten = X_test.reshape(X_test.shape[0], -1)

# Decision Trees model
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_train_flatten, y_train)
predictions = tree_clf.predict(X_test_faltten)
accuracy_decision_trees = accuracy_score(y_test, predictions)
print(f"accuracy_decision_trees: {accuracy_decision_trees * 100:.2f}%")

# The accuracy of random forests model is 33.02%
# The accuracy of decision trees model is 19.46%.


# Implement Data Augmentation 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

# Define the data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,       # Random rotation in the range of [-20, 20] degrees
    width_shift_range=0.1,   # Random horizontal shift by up to 10% of the image width
    height_shift_range=0.1,  # Random vertical shift by up to 10% of the image height
    horizontal_flip=True,    # Randomly flip images horizontally
    zoom_range=0.1,          # Random zoom by up to 10%
    fill_mode='nearest'      # Fill points outside the input boundaries with the nearest available value
)

# Create data generators for training and validation data
train_datagen = datagen.flow(X_train, y_train, batch_size=32, seed=42)
valid_datagen = datagen.flow(X_valid, y_valid, batch_size=32, seed=42)

# Build model here (include model.compile, model.fit, etc.)
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
model.fit(train_datagen, epochs=15, validation_data=valid_datagen, verbose=1)

# Evaluate the model on the test set
probability = model.predict(X_test)
predictions = np.argmax(probability, axis=1)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy with Data Augmentation: {accuracy * 100:.2f}%")

#  Implement k-fold cross-validation on the CNN model you've built to assess its robustness. 

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


# a. Prepare the data for k-fold cross-validation.
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

# One-hot encode the target labels
encoder = OneHotEncoder(sparse=False)
y_train_onehot = encoder.fit_transform(y_train)
y_valid_onehot = encoder.transform(y_valid)


#    b. Implement k-fold cross-validation (choose k as 5) for the CNN model.Record the performance metrics for each fold.
    # define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvscores = []
loss_scores = []
for train, test in kfold.split(X_train, y_train):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[32, 32, 3]),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    # Fit the model
    history = model.fit(X_train[train], y_train_onehot[train], epochs=15, batch_size=10, verbose=0, validation_data=(X_valid, y_valid_onehot))
    # evaluate the model
    scores = model.evaluate(X_train[test], y_train_onehot[test], verbose=0)
    loss = history.history['val_loss'][-1]  # Loss at the last epoch
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # Record performance metrics for this fold
    cvscores.append(scores[1] * 100)
    loss_scores.append(loss)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


#    c. Compute the average and standard deviation of the performance metrics across all folds. 
average_accuracy = np.mean(cvscores)
std_dev_accuracy = np.std(cvscores)

average_loss = np.mean(loss_scores)
std_dev_loss = np.std(loss_scores)

print(f"Average Accuracy Across Folds: {average_accuracy:.4f}")
print(f"Standard Deviation of Accuracy Across Folds: {std_dev_accuracy:.4f}")
print(f"Average Loss Across Folds: {average_loss:.4f}")
print(f"Standard Deviation of Loss Across Folds: {std_dev_loss:.4f}")


