from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

load_dotenv()
import torch
import torch.nn as nn
import librosa
import numpy as np
import shutil
import os

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# SAME PARAMETERS AS TRAINING
# -------------------------------
SR = 16000
DURATION = 22.0
MAX_LEN = int(SR * DURATION)

N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 256
INCLUDE_DELTAS = True

N_FEATS = N_MFCC * 3  # mfcc + delta + delta2

# -------------------------------
# Model architecture (SAME)
# -------------------------------
class SmallCNN(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------
# Feature extraction (IDENTICAL)
# -------------------------------
def load_file(path):
    y, _ = librosa.load(path, sr=SR, mono=True)

    if len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)), mode="constant")
    else:
        y = y[:MAX_LEN]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    if INCLUDE_DELTAS:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feat = np.concatenate([mfcc, delta, delta2], axis=0)
    else:
        feat = mfcc

    return feat.astype(np.float32)

# -------------------------------
# Load trained model
# -------------------------------
model = SmallCNN(N_FEATS).to(device)
model.load_state_dict(
    torch.load("../model/best_model.pt", map_location=device)
)
model.eval()

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="AI Voice Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API is running"}

# -------------------------------
# Security Dependency
# -------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    expected_api_key = os.getenv("API_KEY")
    
    if api_key is None or api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    temp_file = "temp_audio.wav"

    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    feat = load_file(temp_file)
    x = torch.tensor(feat).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    os.remove(temp_file)

    return {
        "prediction": "human" if pred == 1 else "nonhuman",
        "confidence": {
            "nonhuman": float(probs[0]),
            "human": float(probs[1])
        }
    }
