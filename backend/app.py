"""
BanglaSER Backend API
FastAPI server for Bangla Speech Emotion Recognition
"""

import os
import io
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from typing import Dict

# ==========================================
# MODEL ARCHITECTURE 
# ==========================================
import torch.nn as nn

class BanglaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BanglaLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        last_step = self.bn(last_step)
        return self.fc(last_step)


# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "SR": 16000,
    "DURATION": 4,
    "N_MFCC": 40,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MODEL_PATH": os.path.join(os.path.dirname(__file__), "..", "bangla_ser_best.pth")
}

EMOTION_LABELS = {
    0: {"en": "Angry", "bn": "রাগান্বিত", "color": "#ff4444"},
    1: {"en": "Happy", "bn": "খুশি", "color": "#44ff44"},
    2: {"en": "Sad", "bn": "দুঃখিত", "color": "#4444ff"},
    3: {"en": "Neutral", "bn": "স্বাভাবিক", "color": "#888888"},
    4: {"en": "Surprise", "bn": "অবাক", "color": "#ff44ff"}
}

# ==========================================
# INITIALIZE FASTAPI
# ==========================================
app = FastAPI(
    title="BanglaSER API",
    description="Bangla Speech Emotion Recognition API",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# LOAD MODEL
# ==========================================
model = None

def load_model():
    global model
    try:
        print(f"Loading model from: {CONFIG['MODEL_PATH']}")
        print(f"Device: {CONFIG['DEVICE']}")
        
        model = BanglaLSTM(CONFIG['N_MFCC'], 128, 5).to(CONFIG['DEVICE'])
        
        # Load weights
        if not os.path.exists(CONFIG['MODEL_PATH']):
            print(f"❌ Model file not found at: {CONFIG['MODEL_PATH']}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Looking for file at absolute path: {os.path.abspath(CONFIG['MODEL_PATH'])}")
            raise FileNotFoundError(f"Model file not found: {CONFIG['MODEL_PATH']}")
        
        state_dict = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE'], weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()


# ==========================================
# AUDIO PROCESSING
# ==========================================
def process_audio(waveform, sr):
    """Process audio to model input format"""
    
    # Force Mono
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    # Resample if needed
    if sr != CONFIG['SR']:
        resampler = T.Resample(sr, CONFIG['SR'])
        waveform = resampler(waveform)
    
    # Fix length (pad or cut to 4 seconds)
    target_len = CONFIG['SR'] * CONFIG['DURATION']
    current_len = waveform.shape[1]
    
    if current_len < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - current_len))
    else:
        waveform = waveform[:, :target_len]
    
    # Extract Mel Spectrogram
    mel_spec = T.MelSpectrogram(
        sample_rate=CONFIG['SR'],
        n_fft=1024,
        hop_length=512,
        n_mels=CONFIG['N_MFCC']
    )
    to_db = T.AmplitudeToDB()
    
    spec = mel_spec(waveform)
    spec = to_db(spec)
    
    # Reshape: (1, n_mels, time) -> (time, n_mels) -> add batch dim
    input_tensor = spec.squeeze(0).permute(1, 0).unsqueeze(0)
    
    return input_tensor


def predict_emotion(waveform, sr):
    """Predict emotion from audio waveform"""
    try:
        # Process audio
        input_tensor = process_audio(waveform, sr).to(CONFIG['DEVICE'])
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        # Format results
        results = {}
        for idx, prob in enumerate(probs):
            emotion = EMOTION_LABELS[idx]
            results[emotion['en']] = {
                "probability": float(prob),
                "percentage": round(float(prob) * 100, 2),
                "label_bn": emotion['bn'],
                "label_en": emotion['en'],
                "color": emotion['color']
            }
        
        # Get top emotion
        top_idx = probs.argmax().item()
        top_emotion = EMOTION_LABELS[top_idx]
        
        return {
            "success": True,
            "top_emotion": {
                "label_en": top_emotion['en'],
                "label_bn": top_emotion['bn'],
                "confidence": round(float(probs[top_idx]) * 100, 2),
                "color": top_emotion['color']
            },
            "all_emotions": results
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    return {
        "message": "BanglaSER API is running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": CONFIG['DEVICE']
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded audio file
    Supports: .wav, .mp3, .m4a, .ogg, .flac, .webm
    """
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        print(f"Processing file: {file.filename}, extension: {file_ext}, size: {len(audio_bytes)} bytes")
        
        waveform = None
        sr = None
        
        # For M4A files, save to temp file first (librosa needs file path for M4A)
        if file_ext in ['.m4a', '.mp4', '.aac']:
            import tempfile
            print(f"M4A file detected. Attempting conversion...")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            try:
                try:
                    from pydub import AudioSegment
                    print("Trying pydub conversion...")
                    
                    audio_segment = AudioSegment.from_file(temp_audio_path, format=file_ext[1:])
                    
                    # Convert to wav in memory
                    wav_buffer = io.BytesIO()
                    audio_segment.export(wav_buffer, format='wav')
                    wav_buffer.seek(0)
                    
                    # Load with soundfile
                    waveform_np, sr = sf.read(wav_buffer)
                    print(f"✅ M4A converted via pydub: shape={waveform_np.shape}, sr={sr}")
                    
                    # Convert to torch tensor
                    if waveform_np.ndim == 1:
                        waveform = torch.FloatTensor(waveform_np).unsqueeze(0)
                    else:
                        waveform = torch.FloatTensor(waveform_np.T)
                    
                except Exception as pydub_error:
                    print(f"Pydub failed: {pydub_error}")
                    print("Trying librosa with file path...")
                    
                    waveform_np, sr = librosa.load(temp_audio_path, sr=None, mono=False)
                    print(f"✅ M4A loaded via librosa: shape={waveform_np.shape}, sr={sr}")
                    
                    # Convert to torch tensor
                    if waveform_np.ndim == 1:
                        waveform = torch.FloatTensor(waveform_np).unsqueeze(0)
                    else:
                        waveform = torch.FloatTensor(waveform_np)
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
        
        if waveform is None:
            try:
                audio_buffer.seek(0)
                waveform_np, sr = librosa.load(audio_buffer, sr=None, mono=False)
                print(f"✅ Audio loaded with librosa: shape={waveform_np.shape}, sr={sr}")
                
                # Convert to torch tensor
                if waveform_np.ndim == 1:
                    waveform = torch.FloatTensor(waveform_np).unsqueeze(0)
                else:
                    waveform = torch.FloatTensor(waveform_np)
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                
            except Exception as librosa_error:
                print(f"Librosa error: {librosa_error}")
                
                # Fallback to soundfile for WAV/FLAC
                try:
                    audio_buffer.seek(0)
                    waveform_np, sr = sf.read(audio_buffer)
                    print(f"✅ Audio loaded with soundfile: shape={waveform_np.shape}, sr={sr}")
                    
                    # Convert to torch tensor
                    if waveform_np.ndim == 1:
                        waveform = torch.FloatTensor(waveform_np).unsqueeze(0)
                    else:
                        waveform = torch.FloatTensor(waveform_np.T)
                    
                except Exception as sf_error:
                    print(f"❌ All audio loaders failed.")
                    print(f"   File: {file.filename}")
                    print(f"   Extension: {file_ext}")
                    print(f"   Librosa error: {librosa_error}")
                    print(f"   Soundfile error: {sf_error}")
                    
                    error_msg = "Cannot read audio file. "
                    
                    if file_ext in ['.m4a', '.mp4', '.aac']:
                        error_msg += "M4A/AAC files require FFmpeg to be installed. Please either:\n"
                        error_msg += "1. Install FFmpeg (see INSTALL_FFMPEG.md)\n"
                        error_msg += "2. Convert your file to WAV format online at: https://cloudconvert.com/m4a-to-wav\n"
                        error_msg += "3. Upload a WAV, MP3, or OGG file instead"
                    else:
                        error_msg += "Please upload a valid audio file (WAV, MP3, OGG, FLAC, WEBM). Make sure the file is not corrupted."
                    
                    raise HTTPException(status_code=400, detail=error_msg)
        
        # Ensure waveform was loaded
        if waveform is None:
            raise HTTPException(status_code=400, detail="Failed to load audio file")
        
        print(f"Final tensor shape: {waveform.shape}")
        
        result = predict_emotion(waveform, sr)
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("Starting BanglaSER Backend Server")
    print("=" * 50)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
