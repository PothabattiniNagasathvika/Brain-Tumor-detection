import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = 224

data = []
labels = []

dataset_path = "brain_tumor_dataset"

# Load dataset
for category in ["yes", "no"]:
    path = os.path.join(dataset_path, category)
    label = 1 if category == "yes" else 0

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image)
        labels.append(label)

# Convert to array
data = np.array(data) / 255.0
labels = np.array(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Load EfficientNet
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train, y_train,
    epochs=15,
    validation_data=(X_test, y_test)
)

# ✅ SAFE SAVE (NO ERROR VERSION)
model.save_weights("brain_tumor_weights.h5")

print("✅ Training Completed Successfully")