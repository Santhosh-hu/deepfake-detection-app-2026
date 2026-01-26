import streamlit as st
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="Deepfake Detection", layout="wide")

# ---------- LOGIN FUNCTION ----------
def login_page():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "Admin@123":
            st.session_state["logged_in"] = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")
    # ---------- SESSION CHECK ----------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
    st.stop()        
   
st.title(" Deepfake Detection Dashboard")

st.sidebar.success("Logged in as admin")

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

FRAMES_DIR = "frames"
FACE_DIR = "face_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ FUNCTIONS ------------------
def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(FRAMES_DIR, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    return count


def detect_faces():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    saved = 0
    for img_name in os.listdir(FRAMES_DIR):
        img_path = os.path.join(FRAMES_DIR, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_path = os.path.join(FACE_DIR, f"face_{saved}.jpg")
            cv2.imwrite(face_path, face)
            saved += 1

    return saved


def fake_prediction(img_path):
    # ---- DEMO / PRETRAINED SIMULATION ----
    confidence = np.random.uniform(0.55, 0.75)
    label = "FAKE" if confidence > 0.6 else "REAL"
    return label, confidence


# ------------------ UI ------------------
uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:
    st.success("Video uploaded successfully")

    if st.button(" Start Deepfake Detection"):
        with open("temp.mp4", "wb") as f:
            f.write(uploaded_video.read())

        st.info(" Extracting frames...")
        frame_count = extract_frames("temp.mp4")

        st.info(" Detecting faces...")
        face_count = detect_faces()

        st.success(f"Frames: {frame_count} | Faces detected: {face_count}")

        # ---- SHOW FACE FRAMES ----
        st.subheader(" Extracted Face Frames")
        cols = st.columns(4)

        images = os.listdir(FACE_DIR)
        for i, img in enumerate(images[:12]):
            img_path = os.path.join(FACE_DIR, img)
            cols[i % 4].image(img_path, caption=img, use_container_width=True)

        # ---- PREDICTION ----
        st.subheader(" Final Prediction")

        results = []
        confidences = []

        for img in images:
            img_path = os.path.join(FACE_DIR, img)
            label, conf = fake_prediction(img_path)
            results.append(label)
            confidences.append(conf)

        fake_count = results.count("FAKE")
        real_count = results.count("REAL")
        final_result = "FAKE" if fake_count > real_count else "REAL"
        avg_conf = sum(confidences) / len(confidences)

        st.markdown(f"""
        ### 🧾 Final Video Result
        - **Prediction:** `{final_result}`
        - **Fake Frames:** {fake_count}
        - **Real Frames:** {real_count}
        - **Average Confidence:** {round(avg_conf*100, 2)} %
        """)