# AI-Generated Voice Detection

## 📌 Project Overview
This project focuses on **detecting AI-generated (synthetic) voice audio and distinguishing it from human-originated voice samples** using machine learning / deep learning techniques.

With the rapid rise of voice cloning, deepfake audio, and AI speech synthesis, the ability to identify synthetic audio has become critical for:
- Digital forensics
- Media authenticity
- Cybersecurity
- Misinformation detection

This system is designed to take an audio file as input and classify it into predefined categories such as **AI-generated voice** or **human voice**, based on learned acoustic and speech features.

---

## 🎯 Objectives
- Build a reliable model to classify AI-generated vs human voice audio
- Train and evaluate the model on labeled audio datasets
- Expose the trained model through an API for real-world testing and integration
- Provide a simple pipeline for inference and evaluation

---

## 🧠 Technologies & Tools Used
- **Python**
- **PyTorch** (model loading and inference)
- **NumPy / Pandas** (data handling)
- **Librosa** (audio feature extraction)
- **scikit-learn** (metrics & evaluation)
- **FastAPI** (API deployment – optional/extendable)
- **Uvicorn** (API server)

---

## 📂 Project Structure
```
AI-Generated-voice-detection/
│
├── model/                 # Trained model (.pt files)
├── data/                  # Audio datasets (if included)
├── notebooks/             # Jupyter notebooks for training/testing
├── app.py                 # FastAPI application (if used)
├── predict.py             # Inference / prediction logic
├── requirements.txt       # Required Python dependencies
└── README.md              # Project documentation
```

> Note: Folder names may vary depending on experimentation and updates.

---

## 🚀 How It Works
1. **Audio Input** – User provides an audio file (WAV/MP3)
2. **Preprocessing** – Audio is resampled, normalized, and converted into features
3. **Model Inference** – Extracted features are passed to the trained model
4. **Prediction** – Output classifies the audio as AI-generated or human voice

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/TenzinRigzin/AI-Generated-voice-detection.git
cd AI-Generated-voice-detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run Model Inference (Example)
```bash
python predict.py --audio sample.wav
```

### Run API (If FastAPI is implemented)
```bash
uvicorn app:app --reload
```

Access API at:
```
http://127.0.0.1:8000/docs
```

---

## 📊 Model Evaluation
- Accuracy
- Precision & Recall
- Confusion Matrix

Evaluation metrics are used to assess real-world reliability and robustness of the model.

---

## 🔒 Limitations
- Performance depends heavily on dataset quality and diversity
- Generalization across different AI voice generators may vary
- Background noise can affect predictions

---

## 🔮 Future Enhancements
- Support for more audio formats
- Multi-class classification for different AI voice generators
- Improved robustness against noise
- Cloud deployment of the API
- Frontend interface for live testing

---

## 👥 Team Members
- **Tenzin Rigzin**
- **Mehanath S**
- **Shamith Gowda**

---

## 📜 License
This project is intended for academic and research purposes.

---

## 📬 Contact
For queries or collaboration, feel free to reach out via GitHub.
