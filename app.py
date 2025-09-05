import os, time
from typing import List
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# ---------------------------- CONFIG / STYLE ----------------------------
st.set_page_config(
    page_title="Food-101 Classifier",
    page_icon="🍜",
    layout="centered",
)

st.markdown("""
<style>
/* Background gradient */
body {
    background: linear-gradient(160deg, #fff7ed, #fee2e2);
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
    padding: 30px 20px; border-radius: 20px;
    color: white; text-align: center; margin-bottom: 25px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    transition: transform 0.4s ease;
}
.hero:hover {transform: scale(1.03);}
.hero h1 {margin: 0; font-size: 2.5rem;}
.hero p {margin-top: 5px; opacity: 0.9; font-size:1.2rem;}

/* Card */
.card {
    border-radius: 20px; padding: 22px;
    background: rgba(255,255,255,0.15);
    margin-bottom: 18px; border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    backdrop-filter: blur(8px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {transform: translateY(-7px); box-shadow: 0 10px 30px rgba(0,0,0,0.3);}

/* Probability bar */
.prob-bar {height:14px; background:#e5e7eb; border-radius:999px; overflow:hidden; margin-bottom:8px;}
.prob-fill {height:100%; border-radius:999px; transition: width 0.6s ease;}

/* Hide default footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------- HERO ----------------------------
st.markdown("""
<div class="hero">
    <h1>🍔🍣 Food-101 Classifier</h1>
    <p>Upload atau ambil foto makananmu — model akan menebak jenisnya!</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------- Load Model ----------------------------
@st.cache_resource
def load_model(path="pretrain_food.keras"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path, compile=False)

model = None
try:
    with st.spinner("Loading Food-101 model..."):
        model = load_model()
    st.success("✅ Model loaded")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ---------------------------- Load Classes ----------------------------
@st.cache_resource
def load_classes(path="food101_classes.txt") -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f.readlines()]

class_names = load_classes()

# ---------------------------- Utils ----------------------------
def preprocess_pil(img: Image.Image, size: int = 224):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32)/255.0
    return arr[None, ...]

def infer_image(pil_img: Image.Image, topk: int = 5):
    if model is None: return
    x = preprocess_pil(pil_img, 224)
    t0 = time.time()
    logits = model.predict(x)
    dt = (time.time()-t0)*1000

    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    idxs = probs.argsort()[::-1][:topk]

    col1, col2 = st.columns([0.55,0.45])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(pil_img, caption="📸 Input Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 Prediction\n**{class_names[idxs[0]]}** · {probs[idxs[0]]*100:.2f}%")
        st.caption(f"Latency: {dt:.1f} ms")
        for i in idxs:
            st.markdown(f"🍽️ **{class_names[i]}**: {probs[i]*100:.2f}%")
            st.markdown(f"""
                <div class="prob-bar">
                    <div class="prob-fill" style="width:{probs[i]*100:.2f}%;
                        background: linear-gradient(90deg,#f59e0b,#ef4444);"></div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------- Tabs ----------------------------
tab1, tab2, tab3 = st.tabs(["📸 Camera", "🖼️ Upload", "📚 About Dataset"])

with tab1:
    st.subheader("📸 Capture from Webcam")
    cam = st.camera_input("Ambil foto makanan")
    if cam: infer_image(Image.open(cam))

with tab2:
    st.subheader("🖼️ Upload an Image")
    file = st.file_uploader("Pilih JPG/PNG", type=["jpg","jpeg","png"])
    if file: infer_image(Image.open(file))

with tab3:
    st.subheader("📚 About the Food-101 Dataset")
    st.image("https://production-media.paperswithcode.com/datasets/Food-101-0000001068-bdbb54d5_mxgVjph.jpg", use_container_width=True)
    st.markdown("""
    **Food-101** adalah dataset populer untuk klasifikasi makanan:  

    - 🍱 **Jumlah kelas:** 101 kategori makanan  
    - 📷 **Jumlah gambar:** 101,000 (±1000 per kelas)  
    - 🌍 **Sumber:** Gambar diambil dari web dengan kondisi beragam  
    - 🎯 **Tujuan:** Benchmark untuk klasifikasi visual di domain makanan  

    Dataset ini pertama kali diperkenalkan oleh **ETH Zurich** dan sering dipakai untuk 
    menguji CNN maupun arsitektur modern (ResNet, ViT, EfficientNet).  
    """)
