import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]

            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype('float') / 255.0

            roi = np.expand_dims(roi, axis = 0)

            emotion = emotion_model.predict(roi)[0]
            emotion_index = np.argmax(emotion)
            emotion_text = emotion_labels[emotion_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            print(f"Detected: {emotion_text}")

        cv2.imshow('Emotion Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

except KeyboardInterrupt:
    print("Exiting...")
finally:
    cap.release()


