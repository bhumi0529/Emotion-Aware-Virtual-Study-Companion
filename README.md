# 🎓 Emotion-Aware Virtual Study Companion 💡

This project is an intelligent emotion detection system designed to act as a **virtual study buddy**. It uses computer vision and machine learning to detect a user's facial emotion in real-time and offers **motivational or productivity tips** based on the detected emotion.

---

## 🚀 Features

- 😃 Detects facial emotion in real-time using webcam
- 🧠 Trained CNN model on the FER-2013 emotion dataset
- 📦 Uses OpenCV for face detection
- 💬 Provides productivity/motivational tips for each emotion
- 🎓 Ideal for students to stay engaged and emotionally aware while studying

---

## 📁 Project Structure

Emotion-Aware-Virtual-Study-Companion/
│
├── emotion_detector.py # Main app (webcam + prediction + tips)
├── haarcascade_frontalface_default.xml # Face detection model (OpenCV)
├── emotion_model.keras # Trained emotion detection model
├── model/ # Folder storing trained models
├── dataset/ # Dataset folder (optional if using FER-2013)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib (for training plots)

## 🧠 How It Works

1. The user runs `emotion_detector.py`
2. The webcam opens, and a face detection box appears
3. The system detects emotion from facial expressions
4. Based on the predicted emotion, a related tip is shown (e.g., stay calm, take a break, stay motivated)

## 📸 Supported Emotions

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Emotion-Aware-Virtual-Study-Companion.git
cd Emotion-Aware-Virtual-Study-Companion
2. Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies
pip install -r requirements.txt
4. Run the emotion detector
python emotion_detector.py
🎓 For Training the Model (Optional)
If you wish to retrain the model:
You can modify the number of epochs and dataset path inside the Emotion_model.ipynb

📜 Credits
FER-2013 Dataset from Kaggle

Haar Cascade from OpenCV

Developed by [Bhumi] during AI/ML Training

📌 Future Improvements
Add a graphical UI using Tkinter or Streamlit

Log emotions over time for emotional awareness charts

Add speech-based interaction