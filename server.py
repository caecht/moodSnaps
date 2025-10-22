from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load model once at startup
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/detect', methods=['POST'])
def detect_emotion():
    try:
        # Get image data from request
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect emotions
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'emotion': None, 'confidence': 0})
        
        # Get first face
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        
        # Predict emotion
        emotion_probs = emotion_model.predict(roi, verbose=0)[0]
        emotion_index = np.argmax(emotion_probs)
        emotion_text = emotion_labels[emotion_index]
        confidence = float(emotion_probs[emotion_index])
        
        return jsonify({
            'emotion': emotion_text,
            'confidence': confidence
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
