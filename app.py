# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Food Classification", page_icon="🍔", layout="wide")

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("pretrain_food.keras")  # path model kamu
        return model
    except Exception as e:
        st.error("❌ Gagal load model, cek path dan versi TensorFlow.")
        st.stop()

model = load_model()

# ------------------- LOAD CLASSES -------------------
@st.cache_resource
def load_classes():
    with open("food101_classes.txt") as f:
        return [line.strip() for line in f.readlines()]

class_names = load_classes()

# ------------------- PREDICT FUNCTION -------------------
def predict(image: Image.Image):
    img = image.resize((224, 224))  # sesuaikan dengan input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    return class_names[idx], float(np.max(preds[0]))

# ------------------- SIDEBAR -------------------
st.sidebar.title("ℹ️ Tentang Dataset Food-101")
st.sidebar.markdown("""
**Food-101 Dataset**  
- 101 kategori makanan berbeda  
- 101,000 gambar (~1000 per kelas)  
- Gambar diambil dari web, berbagai kondisi pencahayaan dan sudut  
- Digunakan untuk training model klasifikasi makanan  

📌 Referensi: [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
""")

# ------------------- MAIN UI -------------------
st.title("🍴 Food Classification (Food101)")
st.write("Upload gambar atau gunakan kamera untuk mengenali jenis makanan.")

tab1, tab2 = st.tabs(["📸 Kamera", "🖼️ Upload Gambar"])

# -------- CAMERA TAB --------
with tab1:
    st.subheader("Ambil Foto dari Kamera")
    picture = st.camera_input("Ambil gambar makanan kamu")
    if picture:
        img = Image.open(picture).convert("RGB")
        label, conf = predict(img)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img, caption="Gambar dari kamera", use_column_width=True)
        with col2:
            st.markdown("### 🍽️ Prediksi")
            st.success(f"**{label}**")
            st.info(f"Confidence: {conf:.2f}")

# -------- UPLOAD TAB --------
with tab2:
    st.subheader("Upload Gambar Makanan")
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        label, conf = predict(img)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img, caption="Gambar yang diupload", use_column_width=True)
        with col2:
            st.markdown("### 🍽️ Prediksi")
            st.success(f"**{label}**")
            st.info(f"Confidence: {conf:.2f}")

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("Made with ❤️ by Akmal | Dataset: Food-101")
