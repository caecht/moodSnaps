import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from PIL import Image

# Set page config
st.set_page_config(page_title="Emotion Detector", layout="wide")

# Load model and cascade classifier
@st.cache_resource
def load_models():
    emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return emotion_model, face_cascade

emotion_model, face_cascade = load_models()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emotion_photo_map = {
    'Angry': 'angry.jpg',
    'Disgust': 'disgust.jpg',
    'Fear': 'fear.jpg',
    'Happy': 'happy.jpg',
    'Sad': 'sad.jpg',
    'Surprise': 'surprise.jpg',
    'Neutral': 'neutral.jpg'
}

st.title("🎭 Emotion Recognition Detector")
st.write("Detect emotions with live webcam feed!")

# No mode selection, just webcam

def detect_emotions(frame):
    """Detect emotions in a frame and return annotated frame with emotions"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    detected_emotions = []
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        
        emotion = emotion_model.predict(roi, verbose=0)[0]
        emotion_index = np.argmax(emotion)
        emotion_text = emotion_labels[emotion_index]
        confidence = emotion[emotion_index]
        
        detected_emotions.append((emotion_text, confidence))
        
        # Draw rectangle and text
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frame, f"{emotion_text} ({confidence:.2f})", (x, y-10), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, detected_emotions

st.subheader(" Live Webcam Feed")

run = st.checkbox("Start Webcam")
frame_placeholder = st.empty()
emotion_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    
    last_emotion = None
    emotion_start_time = None
    emotion_display = {}
    
    while run:
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to access webcam")
            break
        
        # Detect emotions
        result_frame, detected_emotions = detect_emotions(frame)
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        frame_placeholder.image(result_rgb, width=800)
        
        # Track and display emotion duration
        if detected_emotions:
            current_emotion = detected_emotions[0][0]
            
            if current_emotion != last_emotion:
                last_emotion = current_emotion
                emotion_start_time = time.time()
                emotion_display = {"emotion": current_emotion, "duration": 0}
            else:
                emotion_duration = time.time() - emotion_start_time
                emotion_display = {"emotion": current_emotion, "duration": emotion_duration}
                
                # Change background color if emotion stays for 1.2 seconds
                emotion_colors = {
                    'Angry': '#ff6b6b',      # Red
                    'Disgust': '#9c8e00',    # Yellow-brown
                    'Fear': '#6200ea',       # Purple
                    'Happy': '#ffd700',      # Gold
                    'Sad': '#4d7fcc',        # Blue
                    'Surprise': '#ff9500',   # Orange
                    'Neutral': '#95a5a6'     # Gray
                }
                
                if emotion_duration > 1.2:
                    color = emotion_colors.get(current_emotion, '#ffffff')
                    st.markdown(f"<style>body {{background-color: {color};}}</style>", unsafe_allow_html=True)
                
                # Display emotion info
                with emotion_placeholder.container():
                    st.metric("Current Emotion", current_emotion, f"{emotion_duration:.1f}s")
                    
                    # Show photo if 2 seconds threshold reached
                    if emotion_duration > 2:
                        photo_path = emotion_photo_map.get(current_emotion)
                        if photo_path:
                            try:
                                photo = Image.open(photo_path)
                                st.image(photo, caption=f"{current_emotion} Photo", width='stretch')
                            except Exception as e:
                                st.warning(f"Could not load {photo_path}: {e}")
        
        # Add a small delay
        time.sleep(0.05)
    
    cap.release()
    st.success("Webcam stopped!")

