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
        model = tf.keras.models.load_model("pretrain_food.keras")
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
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    return class_names[idx], float(np.max(preds[0]))

# ------------------- HEADER -------------------
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3075/3075977.png", width=80)  # logo contoh
with col2:
    st.markdown("<h1 style='margin-bottom:0;'>🍴 Food Classification (Food-101)</h1>", unsafe_allow_html=True)
    st.write("Klasifikasi gambar makanan dengan model deep learning (Food-101 Dataset).")

# ------------------- MAIN TABS -------------------
tab1, tab2, tab3 = st.tabs(["📸 Kamera", "🖼️ Upload Gambar", "📊 Tentang Dataset"])

# -------- CAMERA TAB --------
with tab1:
    st.subheader("Ambil Foto dari Kamera")
    picture = st.camera_input("Ambil gambar makanan kamu")
    if picture:
        img = Image.open(picture).convert("RGB")
        label, conf = predict(img)

        st.markdown("### 🔍 Hasil Prediksi")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img, caption="📸 Gambar dari kamera", use_column_width=True)
        with col2:
            st.success(f"🍽️ Jenis Makanan: **{label}**")
            st.info(f"📊 Confidence: {conf:.2f}")

# -------- UPLOAD TAB --------
with tab2:
    st.subheader("Upload Gambar Makanan")
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        label, conf = predict(img)

        st.markdown("### 🔍 Hasil Prediksi")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img, caption="🖼️ Gambar yang diupload", use_column_width=True)
        with col2:
            st.success(f"🍽️ Jenis Makanan: **{label}**")
            st.info(f"📊 Confidence: {conf:.2f}")

# -------- DATASET TAB --------
with tab3:
    st.subheader("📊 Tentang Dataset Food-101")
    st.image("https://production-media.paperswithcode.com/datasets/Food-101-0000001068-bdbb54d5_mxgVjph.jpg", use_column_width=True)
    st.markdown("""
    **Food-101 Dataset** adalah dataset populer untuk tugas klasifikasi makanan.  
    - **Jumlah kelas**: 101 kategori makanan berbeda  
    - **Jumlah gambar**: 101,000 (±1000 per kelas)  
    - **Sumber**: Gambar diambil dari web dengan kondisi beragam  
    - **Tujuan**: Benchmark untuk klasifikasi visual di domain makanan  
    
    📌 Referensi resmi: [ETH Zurich Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
    """)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("<center>Made with ❤️ by Akmal | Dataset: Food-101</center>", unsafe_allow_html=True)
