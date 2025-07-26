from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import webbrowser
import threading
import time

app = Flask(__name__)
model = tf.keras.models.load_model("emotion_model.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

responses = {
    "angry": "Take a deep breath. Let's calm down with a 2-minute break.",
    "disgust": "Maybe a short walk will refresh you.",
    "fear": "You're doing great. Believe in yourself!",
    "happy": "That's great! Keep up the energy!",
    "neutral": "Let’s focus on your study goals.",
    "sad": "Take a short break, then come back stronger. 💪",
    "surprise": "Whoa! Something unexpected? Stay focused."
}

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(gray, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = np.reshape(face_normalized, (1, 48, 48, 1))
            prediction = model.predict(face_input)
            emotion = emotion_labels[np.argmax(prediction)]

            # Get smart response
            tip = responses[emotion]

            # Draw Emotion and Tip
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (139,0,0), 2)
            cv2.putText(frame, f'Tip: {tip}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



def open_browser():
    time.sleep(1)  # Wait for Flask to start
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug=False)
