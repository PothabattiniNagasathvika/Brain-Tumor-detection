import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

IMG_SIZE = 224

# 🌐 Page setup
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# 🎨 PREMIUM STYLE
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main-card {
    background-color: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,255,255,0.2);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00ffff;
}
.subtitle {
    text-align: center;
    color: #cccccc;
    margin-bottom: 25px;
}
.stButton>button {
    background: linear-gradient(90deg, #00ffff, #00cccc);
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00cccc, #009999);
}
</style>
""", unsafe_allow_html=True)

# 🧠 HEADER
st.markdown('<div class="title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered MRI Analysis</div>', unsafe_allow_html=True)

# 🔹 Load model
@st.cache_resource
def load_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights("brain_tumor_weights.h5")

    return model

model = load_model()

# 📦 CARD UI
st.markdown('<div class="main-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Detect Tumor"):
        with st.spinner("Analyzing..."):
            pred = model.predict(img)
            confidence = pred[0][0]

            st.markdown("---")

            if confidence > 0.7:
                st.error(f"❗ Tumor Detected\n\nConfidence: {confidence*100:.2f}%")
            else:
                st.success(f"✅ No Tumor Detected\n\nConfidence: {(1-confidence)*100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)