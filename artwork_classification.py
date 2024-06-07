import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Constants
img_height, img_width = 128, 128  # Example dimensions, adjust as needed

# Load metadata from artists.csv
metadata_df = pd.read_csv("artists.csv")

# Create a mapping from artist names to labels
artist_to_label = {artist: idx for idx, artist in enumerate(metadata_df['name'].unique())}
num_classes = len(artist_to_label)

# Load and preprocess images
images = []
labels = []

for index, row in metadata_df.iterrows():
    # Assuming image filenames are in the format <id>.jpg
    image_path = f"images/{row['id']}.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path).resize((img_width, img_height))
        image = np.array(image) / 255.0  # Normalize pixel values
        images.append(image)
        labels.append(artist_to_label[row['name']])  # Use artist name to generate label

images = np.array(images)
labels = np.array(labels)

# Split dataset into training, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Define CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
