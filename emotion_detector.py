import cv2
import numpy as np
import tensorflow as tf
import random

# Load model
model = tf.keras.models.load_model("emotion_model.keras")

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Smart responses
responses = {
    "angry": "Take a deep breath. Let's calm down with a 2-minute break.",
    "disgust": "Maybe a short walk will refresh you.",
    "fear": "You're doing great. Believe in yourself!",
    "happy": "That's great! Keep up the energy!",
    "neutral": "Let’s focus on your study goals.",
    "sad": random.choice([
        "Don't worry, better days are coming!",
        "You're stronger than you think!",
        "Take a short break, then come back stronger. 💪"
    ]),
    "surprise": "Whoa! Something unexpected? Stay focused."
}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))

    prediction = model.predict(face)
    emotion = emotion_labels[np.argmax(prediction)]

    # Get smart response
    response = responses[emotion]

    # Display on screen
    cv2.putText(frame, f"Emotion: {emotion}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Tip: {response}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Emotion-Aware Study Companion", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
