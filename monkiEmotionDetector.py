import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

frame_count = 0
last_frame = None
last_emotion = None
emotion_start_time = None
emotion_photo_map = {
    'Angry': 'angry.jpg',
    'Disgust': 'disgust.jpg',
    'Fear': 'fear.jpg',
    'Happy': 'happy.jpg',
    'Sad': 'sad.jpg',
    'Surprise': 'surprise.jpg',
    'Neutral': 'neutral.jpg'
}

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

            # Track emotion duration
            if emotion_text != last_emotion:
                last_emotion = emotion_text
                emotion_start_time = time.time()
                print(f"Emotion changed to: {emotion_text}")
            else:
                emotion_duration = time.time() - emotion_start_time
                if emotion_duration > 1.2:  # 1.2 seconds of same emotion
                    # Try to display the corresponding photo
                    photo_path = emotion_photo_map.get(emotion_text)
                    if photo_path:
                        try:
                            img = cv2.imread(photo_path)
                            if img is not None:
                                cv2.imshow(f'{emotion_text} Photo', img)
                                print(f"✓ Displayed {emotion_text} photo from {photo_path}!")
                                emotion_start_time = time.time()  # Reset timer
                            else:
                                print(f"✗ Photo file not found or couldn't be read: {photo_path}")
                        except Exception as e:
                            print(f"✗ Error displaying photo: {e}")
                    else:
                        print(f"✗ No photo mapping for emotion: {emotion_text}")

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


