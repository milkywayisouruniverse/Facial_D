import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the emotion model
model = load_model("emotion_model.hdf5", compile=False)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("😊 Facial Emotion Detector App")
st.markdown("Detect facial emotions from images or webcam.")

option = st.radio("Choose Input Source", ['Upload Image', 'Use Webcam'])
def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.astype("float") / 255.0
    face_img = np.expand_dims(face_img, axis=0)        # (1, 64, 64)
    face_img = np.expand_dims(face_img, axis=-1)       # (1, 64, 64, 1)
    preds = model.predict(face_img)[0]
    return EMOTIONS[np.argmax(preds)]


# Upload Image
if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded Image", channels="BGR")

        if st.button("Detect Emotion"):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                emotion = predict_emotion(roi)
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Result')

# Webcam Mode (Basic, not real-time)
elif option == 'Use Webcam':
    run = st.button("Start Webcam")

    if run:
        cam = cv2.VideoCapture(0)
        stframe = st.image([])
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                emotion = predict_emotion(roi)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
