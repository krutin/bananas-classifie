import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Define the number of classes in your dataset
num_classes = 8  # Replace 8 with the actual number of classes

# Load the preprocessed images and labels from files
preprocessed_images = np.load('preprocessed_images.npy')
preprocessed_labels = np.load('preprocessed_labels.npy')

# Additional preprocessing steps if needed
# Example: Normalize pixel values to the [0, 1] range
preprocessed_images = preprocessed_images.astype("float32") / 255.0

# Example: Convert labels to one-hot encoding (for classification tasks)
label_encoder = LabelEncoder()
label_encoder.fit(preprocessed_labels)
preprocessed_labels_encoded = label_encoder.transform(preprocessed_labels)
preprocessed_labels_one_hot = to_categorical(preprocessed_labels_encoded)

# Split the data into training (75%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(
    preprocessed_images, preprocessed_labels_one_hot, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.6, random_state=42)

# Define a simple CNN model (you can modify this as needed)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # num_classes should be set based on your dataset
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_val, y_val))

# Save the trained model to a file
model.save('banana_ripeness_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

