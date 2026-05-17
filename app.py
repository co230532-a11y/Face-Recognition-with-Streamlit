import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
from pathlib import Path


# Custom modules
from src.preprocessing.preprocessing_techniques import PreprocessingPipeline
from src.core.evaluation import AccuracyCalculator, EvaluationMetrics


# Initialize components
pipeline = PreprocessingPipeline()
evaluator = AccuracyCalculator()


# Load Detectors
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


# --- CONFIGURATION ---
DATABASE_DIR = Path(r"database/reference_images")
TRUTH_BASELINE = 65.0  
CONFIDENCE_GAP = 5.0  


# Updated Session State to store both visual and biometric data
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'Front'
if 'captured_visuals' not in st.session_state:
    st.session_state.captured_visuals = {} # Stores the Full image + Green Box
if 'captured_biometrics' not in st.session_state:
    st.session_state.captured_biometrics = {} # Stores the Enhanced Crop


def get_enhanced_face_crop(image):
    """Finds face, draws green box, and enhances crop."""
    display_frame = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Detect Face
    faces = frontal_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            flipped_gray = cv2.flip(gray, 1)
            faces_flipped = profile_cascade.detectMultiScale(flipped_gray, 1.2, 5)
            if len(faces_flipped) > 0:
                xf, yf, wf, hf = faces_flipped[0]
                faces = [[gray.shape[1] - xf - wf, yf, wf, hf]]


    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
       
        # GREEN BOX LOGIC
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


        # Enhancement Logic
        face_roi = image[max(0, y):y+h, max(0, x):x+w]
        face_resized = cv2.resize(face_roi, (160, 160), interpolation=cv2.INTER_CUBIC)
        enhanced_face = pipeline.preprocess_normal(face_resized)
       
        return enhanced_face, display_frame, True
   
    return pipeline.preprocess_normal(cv2.resize(image, (160, 160))), display_frame, False


def calculate_detailed_metrics(ref_img, enhanced_test_face):
    """Matches pre-enhanced face against database."""
    ref_face, _, _ = get_enhanced_face_crop(ref_img)
    ref_gray = cv2.cvtColor(ref_face, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(enhanced_test_face, cv2.COLOR_BGR2GRAY)
   
    ssim = float(EvaluationMetrics.structural_similarity(ref_gray, test_gray))
    ref_feat = EvaluationMetrics.feature_vector_extraction(ref_face)
    test_feat = EvaluationMetrics.feature_vector_extraction(enhanced_test_face)
   
    cos_sim = float(evaluator.cosine_similarity(ref_feat, test_feat))
    cos_pct = float((cos_sim + 1) / 2 * 100)
   
    combined = (ssim * 0.05) + (cos_pct * 0.95)
    return {"Accuracy (%)": round(combined, 2), "SSIM": round(ssim, 4), "Cosine": round(cos_pct, 4)}


# --- UI SECTION ---
st.set_page_config(page_title="Secure Surveillance Terminal", layout="wide")


st.sidebar.header("📁 Reference Database")
db_images = sorted(list(DATABASE_DIR.glob("*.*")))
for img_p in db_images:
    st.sidebar.image(str(img_p), caption=img_p.name, use_container_width=True)


st.title("📡 Surveillance Identity Terminal")


# --- ALWAYS SHOW CAPTURED GALLERY (The "Surveillance Log") ---
if st.session_state.captured_visuals:
    st.subheader("📸 Live Surveillance Log")
    log_cols = st.columns(3)
    for i, angle in enumerate(['Front', 'Left', 'Right']):
        if angle in st.session_state.captured_visuals:
            with log_cols[i]:
                st.image(st.session_state.captured_visuals[angle], caption=f"Detected: {angle}", channels="BGR", use_container_width=True)
                st.image(st.session_state.captured_biometrics[angle], caption=f"Enhanced: {angle}", use_container_width=True)


current = st.session_state.current_step


if current != 'Complete':
    st.divider()
    st.subheader(f"Current Phase: {current}")
    img_buffer = st.camera_input(f"Snap {current} View", key=f"cam_{current}")


    if img_buffer:
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        frame = cv2.flip(cv2.imdecode(file_bytes, 1), 1)
       
        enhanced_face, boxed_frame, found = get_enhanced_face_crop(frame)
       
        # Save both versions to session state
        st.session_state.captured_visuals[current] = boxed_frame
        st.session_state.captured_biometrics[current] = enhanced_face
       
        with st.status(f"Processing {current} angle..."):
            time.sleep(2)
            if current == 'Front': st.session_state.current_step = 'Left'
            elif current == 'Left': st.session_state.current_step = 'Right'
            else: st.session_state.current_step = 'Complete'
            st.rerun()
else:
    # --- FINAL ANALYSIS VIEW ---
    st.success("✅ Analysis Complete.")
    if st.button("Clear Log & Restart"):
        st.session_state.current_step = 'Front'
        st.session_state.captured_visuals = {}
        st.session_state.captured_biometrics = {}
        st.rerun()


    results = []
    for ref_path in db_images:
        ref_img = cv2.imread(str(ref_path))
        angle_scores = []
        for angle, enhanced_frame in st.session_state.captured_biometrics.items():
            m = calculate_detailed_metrics(ref_img, enhanced_frame)
            angle_scores.append(m)
       
        best_match = max(angle_scores, key=lambda x: x['Accuracy (%)'])
        best_match.update({"Name": ref_path.name, "Path": str(ref_path)})
        results.append(best_match)


    sorted_res = sorted(results, key=lambda x: x['Accuracy (%)'], reverse=True)


    st.subheader("📊 Detailed Leaderboard")
    st.table(pd.DataFrame(sorted_res).drop(columns=["Path"]))


    st.subheader("🥇 Match Ranking")
    cols = st.columns(len(sorted_res))
    for i, res in enumerate(sorted_res):
        with cols[i]:
            st.image(res['Path'], caption=f"Rank {i+1}", use_container_width=True)
            st.metric("Accuracy", f"{res['Accuracy (%)']}%")

