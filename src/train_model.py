import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the dataset
train_data = pd.read_csv("/home/arx/Projects/SignTrans/dataset/sign_mnist_train.csv")
test_data = pd.read_csv("/home/arx/Projects/SignTrans/dataset/sign_mnist_test.csv")

# Extract labels and images
y_train = train_data['label']
X_train = train_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255.0
y_test = test_data['label']
X_test = test_data.drop(columns=['label']).values.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=26)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=26)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
# Save the model with a .h5 extension (or .keras if you prefer)
model.save("/home/arx/Projects/SignTrans/models/my_model.h5")
print("Model saved successfully!")
