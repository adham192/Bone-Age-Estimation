import io
import gdown
import os
import streamlit as st
from PIL import Image
import json
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image_from_bytes
import urllib.request

st.set_page_config(
    page_title="Bone Age Estimation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Unified Background */
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Container */
.wrapper {
    max-width: 1150px;
    margin: auto;
    padding-top: 50px;
}

/* Title */
.title {
    font-size: 48px;
    font-weight: 700;
    color: #60a5fa;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #cbd5e1;
    margin-top: 10px;
    max-width: 700px;
}

/* Cards */
.card {
    background-color: #1e293b;
    padding: 30px;
    border-radius: 18px;
    margin-top: 40px;
    border: 1px solid #334155;
}

/* Step cards */
.step {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #1f2937;
}

/* Input labels */
label {
    color: #cbd5e1 !important;
}

/* File uploader + selectbox background fix */
section[data-testid="stFileUploader"] {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}

div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    border: 1px solid #334155 !important;
}

/* Full-Width Centered Professional Button */
div.stButton > button {
    display: flex !important;
    justify-content: center !important; /* Horizontally centers the text */
    align-items: center !important;     /* Vertically centers the text */
    margin: 50px auto !important;
    width: 100% !important;             /* Force it to fill the card width */
    height: 100px !important;           /* Prominent height */
    
    /* Font Styling */
    font-size: 36px !important;         /* Massive, readable font */
    font-weight: 700 !important;
    white-space: nowrap !important;     /* Keep 'Estimate Bone Age' on one line */
    letter-spacing: 2px !important;
    
    /* Clinical Theme */
    color: #ffffff !important;
    background: linear-gradient(90deg, #3b82f6, #1e40af) !important;
    border-radius: 18px !important;
    border: none !important;
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.4);
    transition: all 0.3s ease;
}

/* Hover effect to show interactivity */
div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 45px rgba(59, 130, 246, 0.6);
    background: linear-gradient(90deg, #60a5fa, #2563eb) !important;
}

div.stButton > button:active {
    transform: scale(0.99);
}

/* Hover Effect */
div.stButton > button:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 60px rgba(59, 130, 246, 0.6);
    background: linear-gradient(90deg, #60a5fa, #2563eb) !important;
}

div.stButton > button:active {
    transform: scale(0.97); /* "Click" feedback */
}

/* Result */
.result-box {
    margin-top: 25px;
    padding: 25px;
    background-color: #0b1220;
    border-left: 5px solid #3b82f6;
    border-radius: 12px;
    font-size: 20px;
}

/* Interactive bone */
.bone {
    animation: float 5s ease-in-out infinite;
}

.bone:hover {
    transform: scale(1.1);
    transition: 0.3s ease;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-15px); }
    100% { transform: translateY(0px); }
}

</style>
""", unsafe_allow_html=True)

# Layout

st.markdown("<div class='wrapper'>", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("<div class='title'>Bone Age Estimation System</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='subtitle'>
    An AI-powered clinical decision support system developed to estimate 
    pediatric skeletal maturity from hand radiographs. Designed with 
    professional medical presentation standards and usability in mind.
    </div>
    """, unsafe_allow_html=True)

# The dynamic bone in the UI
with col2:
    st.markdown("""
    <div class="bone-container">
        <svg xmlns="http://www.w3.org/2000/svg" width="180" height="180" viewBox="0 0 14 14">
            <title>bone</title>
            <g fill="none">
                <path fill="#3b82f6" d="M11.5 2.5a2 2 0 1 0-4 0a2 2 0 0 0 .59 1.41L3.91 8.09A2 2 0 0 0 2.5 7.5a2 2 0 1 0 0 4a2 2 0 0 0 4 0a2 2 0 0 0-.59-1.41l4.18-4.18a2 2 0 0 0 1.41.59a2 2 0 0 0 0-4"/>
                <path stroke="#60a5fa" stroke-linecap="round" stroke-linejoin="round" d="M11.5 2.5a2 2 0 1 0-4 0a2 2 0 0 0 .59 1.41L3.91 8.09A2 2 0 0 0 2.5 7.5a2 2 0 1 0 0 4a2 2 0 0 0 4 0a2 2 0 0 0-.59-1.41l4.18-4.18a2 2 0 0 0 1.41.59a2 2 0 0 0 0-4"/>
            </g>
        </svg>
    </div>
    <style>
    .bone-container {
        display: flex;
        justify-content: center;
        animation: float 5s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }
    </style>
    """, unsafe_allow_html=True)
# How the system works

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### How the System Works")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='step'><b>1. Upload X-ray</b><br>Provide pediatric hand radiograph.</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='step'><b>2. Select Gender</b><br>Specify patient gender.</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='step'><b>3. Generate Estimation</b><br>System calculates skeletal maturity.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Input

st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Hand X-ray (PNG / JPG)", type=["png","jpg","jpeg"])
gender = st.selectbox("Patient Gender", ["Female","Male","Other"])

if uploaded:
    image_bytes = uploaded.read()          
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, width=450)

st.markdown("<div class='big-button'>", unsafe_allow_html=True)
estimate = st.button("Estimate Bone Age")
st.markdown("</div>", unsafe_allow_html=True)

model_path = "Application/bone_age_InceptionV3_final_8_8_MAE.keras"
url = "https://huggingface.co/Adham192/bone-age-estimation/resolve/main/bone_age_InceptionV3_final_8_8_MAE.keras"
json_path = "Application/normalisation_stats.json"
@st.cache_resource
def load_model_and_stats():
    if not os.path.exists(model_path):
        with st.spinner("Loading model for the first time, please wait..."):
            urllib.request.urlretrieve(url, model_path)
            st.success("Download complete!")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    with open(json_path) as f:
        stats = json.load(f)
    
    return model, stats["BONEAGE_MEAN"], stats["BONEAGE_STD"]

model, BONEAGE_MEAN, BONEAGE_STD = load_model_and_stats()


def estimate_age(image_bytes, gender):
    img_array    = preprocess_image_from_bytes(image_bytes)
    gender_input = np.array([[1.0 if gender == "Male" else 0.0]], dtype=np.float32)

    pred_norm   = model.predict([img_array, gender_input], verbose=0)[0][0]
    pred_months = pred_norm * BONEAGE_STD + BONEAGE_MEAN
    pred_months = max(0, round(float(pred_months)))

    MAE = 4
    low  = max(0, round(pred_months - MAE))
    high = round(pred_months + MAE)

    return pred_months // 12, pred_months % 12, low // 12, low % 12, high // 12, high % 12

if estimate:
    if not uploaded:
        st.error("Please upload an X-ray image before proceeding.")
    else:
        y, m, low_y, low_m, high_y, high_m = estimate_age(image_bytes, gender)
        st.markdown(f"""
        <div class='result-box'>
        <b>Assessment Result</b><br><br>
        The estimated bone age range for the {gender.lower()} child is between 
        <b>{low_y} years and {low_m} months</b> 
        to
        <b>{high_y} years and {high_m} months</b><br><br>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
