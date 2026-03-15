# app.py
import streamlit as st
import joblib
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import os
import time

# ------------------------------
# PAGE CONFIGURATION (must be first)
# ------------------------------
st.set_page_config(
    page_title="Arabic Threat Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# CUSTOM CSS for premium look
# ------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Style the main container */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.2rem;
        color: #fff;
        opacity: 0.9;
    }

    /* Card style for results */
    .result-card {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.5);
        transition: transform 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-5px);
    }

    .safe-result {
        border-left: 8px solid #2ecc71;
    }

    .threat-result {
        border-left: 8px solid #e74c3c;
    }

    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.2);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }

    /* Spinner style */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.9);
        font-size: 1rem;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: border 0.3s;
    }

    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.2);
    }

    /* File uploader */
    .stFileUploader > div {
        border-radius: 15px;
        border: 2px dashed rgba(255,255,255,0.5);
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        color: white;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
    }

    .metric-card h3 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px);
    }

    .sidebar-content {
        color: white;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Tesseract path configuration (for local and cloud)
# ------------------------------
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:  # Linux/Mac (Streamlit Cloud)
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# ------------------------------
# SIDEBAR with model info
# ------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: white; font-weight: 700;">🔍 Model Info</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>LinearSVM</h3>
            <p>Algorithm</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>TF-IDF</h3>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="color: white;">
        <p style="font-weight: 600;">📊 Dataset: L-HSAB</p>
        <p>⚙️ n-gram range: (2,5)</p>
        <p>🎯 Classes: 2 (Offensive / Non-Offensive)</p>
        <p>📅 Trained: 2025-03-15</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🔄 Reload Model"):
        st.cache_data.clear()
        st.rerun()

# ------------------------------
# MAIN HEADER
# ------------------------------
st.markdown("""
<div class="main-header">
    <h1>🔍 Arabic Threat Detection</h1>
    <p>Detect abusive and hate speech in Arabic text or images using AI</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD MODEL (cached)
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/lhsab_svm_model.pkl")
    vectorizer = joblib.load("vectorizers/lhsab_tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

label_mapping = {0: "Non-Offensive", 1: "Offensive/Hate"}

# ------------------------------
# TEXT CLEANING FUNCTION
# ------------------------------
def clean_arabic_text(text):
    """Clean Arabic text (same as training)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------
# OCR FUNCTION
# ------------------------------
def extract_text_from_image(image):
    """Extract Arabic text from image using Tesseract."""
    try:
        # Convert PIL to OpenCV format
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR with Arabic language
        text = pytesseract.image_to_string(
            thresh,
            lang='ara',
            config='--oem 3 --psm 6'
        )
        return text.strip()
    except Exception as e:
        return f"OCR Error: {e}"

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_text(text):
    """Run model prediction on cleaned text."""
    cleaned = clean_arabic_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    score = model.decision_function(vec)[0]
    confidence = min(abs(score) * 2, 1.0)  # scale to ~0-1
    return pred, confidence, cleaned, score

# ------------------------------
# TABS: Text Input & Image Upload
# ------------------------------
tab1, tab2 = st.tabs(["📝 Text Input", "🖼️ Image Upload"])

# ----- Tab 1: Text Input -----
with tab1:
    st.markdown("### ✍️ Enter Arabic text below")
    user_text = st.text_area(
        "",
        placeholder="اكتب النص العربي هنا...",
        height=150,
        key="text_input"
    )
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        analyze_btn = st.button("🚀 Analyze Text", use_container_width=True)
    
    if analyze_btn and user_text:
        with st.spinner("🔍 Analyzing..."):
            time.sleep(1)  # slight delay for effect
            pred, conf, cleaned, score = predict_text(user_text)
        
        # Display result in a nice card
        result_class = "safe-result" if pred == 0 else "threat-result"
        result_text = label_mapping[pred]
        
        st.markdown(f"""
        <div class="result-card {result_class}">
            <h2 style="margin-top:0;">📊 Analysis Result</h2>
            <p style="font-size:1.3rem;"><strong>Prediction:</strong> {result_text}</p>
            <p><strong>Confidence:</strong> {conf*100:.2f}%</p>
            <p><strong>Decision Score:</strong> {score:.4f}</p>
            <p><strong>Words:</strong> {len(cleaned.split())}</p>
            <p><strong>Characters:</strong> {len(cleaned)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if pred == 1:
            st.error("⚠️ This text appears to be offensive or hateful.")
        else:
            st.success("✅ This text appears to be non-offensive.")
    
    elif analyze_btn and not user_text:
        st.warning("⚠️ Please enter some text.")

# ----- Tab 2: Image Upload -----
with tab2:
    st.markdown("### 📤 Upload an image containing Arabic text")
    uploaded_file = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            ocr_btn = st.button("🔍 Extract & Analyze", use_container_width=True)
        
        if ocr_btn:
            with st.spinner("🔄 Running OCR..."):
                time.sleep(1.5)
                extracted = extract_text_from_image(image)
            
            if extracted and "OCR Error" not in extracted:
                st.markdown("### 📄 Extracted Text")
                st.info(extracted)
                
                with st.spinner("🔍 Analyzing text..."):
                    pred, conf, cleaned, score = predict_text(extracted)
                
                result_class = "safe-result" if pred == 0 else "threat-result"
                result_text = label_mapping[pred]
                
                st.markdown(f"""
                <div class="result-card {result_class}">
                    <h2 style="margin-top:0;">📊 Analysis Result</h2>
                    <p style="font-size:1.3rem;"><strong>Prediction:</strong> {result_text}</p>
                    <p><strong>Confidence:</strong> {conf*100:.2f}%</p>
                    <p><strong>Decision Score:</strong> {score:.4f}</p>
                    <p><strong>Words:</strong> {len(cleaned.split())}</p>
                    <p><strong>Characters:</strong> {len(cleaned)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if pred == 1:
                    st.error("⚠️ The extracted text appears to be offensive or hateful.")
                else:
                    st.success("✅ The extracted text appears to be non-offensive.")
            else:
                st.error("❌ Could not extract Arabic text from the image. Please ensure the image contains clear Arabic script.")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255,255,255,0.05); border-radius: 20px;">
            <p style="color: rgba(255,255,255,0.7); font-size: 1.2rem;">📸 Drag and drop an image or click to browse</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("""
<div class="footer">
    <p>🚀 Built with Streamlit & LinearSVM | Dataset: L-HSAB | OCR: Tesseract</p>
    <p>© 2025 Arabic Threat Detection System</p>
</div>
""", unsafe_allow_html=True)