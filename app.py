import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = 224

# Model structure
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Load weights
model.load_weights("brain_tumor_weights.h5")

file_path = ""

# Upload image
def upload_image():
    global file_path
    file_path = filedialog.askopenfilename()

    img = Image.open(file_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)

    panel.config(image=img)
    panel.image = img

    label_result.config(text="Image Loaded", fg="blue")

# Detect tumor
def detect():
    if file_path == "":
        label_result.config(text="Upload image first", fg="orange")
        return

    img = cv2.imread(file_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    pred = model.predict(img)
    confidence = pred[0][0]

    if confidence > 0.7:
        result = f"Tumor Detected ❗ ({confidence*100:.2f}%)"
        label_result.config(text=result, fg="red")
    else:
        result = f"No Tumor ✅ ({(1-confidence)*100:.2f}%)"
        label_result.config(text=result, fg="green")

# GUI
root = tk.Tk()
root.title("🧠 Brain Tumor Detection AI")
root.geometry("400x500")

title = tk.Label(root, text="Brain Tumor Detection", font=("Arial", 18, "bold"))
title.pack(pady=10)

panel = tk.Label(root)
panel.pack(pady=10)

btn1 = tk.Button(root, text="Upload MRI Image", command=upload_image, width=20)
btn1.pack(pady=5)

btn2 = tk.Button(root, text="Detect Tumor", command=detect, width=20)
btn2.pack(pady=5)

label_result = tk.Label(root, text="Result will appear here", font=("Arial", 14))
label_result.pack(pady=20)

root.mainloop()