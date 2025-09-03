# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import tempfile
import time

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Food Classification", page_icon="🍔", layout="wide")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("food_model.h5")  # ganti dengan path model kamu
    return model

model = load_model()

# Kelas Food101
@st.cache_resource
def load_classes():
    with open("food101_classes.txt") as f:  # file txt berisi daftar kelas Food101
        classes = [line.strip() for line in f.readlines()]
    return classes

class_names = load_classes()

# ------------------- PREDICT FUNCTION -------------------
def predict(image: Image.Image):
    img = image.resize((224, 224))  # sesuaikan dengan input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    return class_names[idx], float(np.max(preds[0]))

# ------------------- UI -------------------
st.title("🍴 Food Classification (Food101)")
st.write("Upload gambar atau gunakan kamera untuk mengenali jenis makanan.")

tab1, tab2 = st.tabs(["📸 Live Camera", "🖼️ Upload Gambar"])

with tab1:
    st.subheader("Ambil Foto dari Kamera")
    picture = st.camera_input("Ambil gambar makanan kamu")
    if picture is not None:
        img = Image.open(picture).convert("RGB")
        label, conf = predict(img)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img, caption="Gambar dari kamera", use_column_width=True)
        with col2:
            st.success(f"🍽️ Prediksi: **{label}**\n\n📊 Confidence: {conf:.2f}")

with tab2:
    st.subheader("Upload Gambar Makanan")
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        label, conf = predict(img)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img, caption="Gambar yang diupload", use_column_width=True)
        with col2:
            st.success(f"🍽️ Prediksi: **{label}**\n\n📊 Confidence: {conf:.2f}")
