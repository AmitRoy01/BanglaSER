# 🇧🇩 BanglaSER - Bangla Speech Emotion Recognition Web Application

A full-stack web application for detecting emotions from Bangla speech using deep learning (LSTM). Users can record live audio or upload audio files to get real-time emotion predictions.

![BanglaSER](https://img.shields.io/badge/BanglaSER-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![React](https://img.shields.io/badge/React-18.2-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal)

## 🎯 Features

- **Live Voice Recording**: Record audio directly from your microphone
- **File Upload**: Upload pre-recorded audio files (WAV, MP3, OGG, FLAC, WEBM)
- **Real-time Emotion Detection**: Get instant emotion predictions
- **5 Emotion Classes**: Angry, Happy, Sad, Neutral, Surprise
- **Professional UI**: Modern, responsive React interface
- **Bilingual Support**: English and Bangla emotion labels
- **Detailed Analysis**: Confidence scores for all emotions

## 📁 Project Structure

```
Bangla-Speech-Emotion-Recognition/
├── backend/
│   ├── app.py                 # FastAPI server
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example          # Environment variables template
│   └── .gitignore
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── EmotionDetector.js
│   │   │   └── EmotionDetector.css
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   ├── package.json
│   ├── .env
│   └── .gitignore
├── bangla_ser_best.pth        # Trained model file
└── README.md
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Backend Setup

1. **Navigate to backend directory**
```bash
cd backend
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify model file location**
Make sure `bangla_ser_best.pth` is in the project root directory (one level up from backend/).

### Frontend Setup

1. **Navigate to frontend directory**
```bash
cd frontend
```

2. **Install Node.js dependencies**
```bash
npm install
```

## ▶️ Running the Application

### Start Backend Server

```bash
cd backend
python app.py
```

The API will run on `http://localhost:8000`

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict emotion from audio

### Start Frontend Development Server

```bash
cd frontend
npm start
```

The React app will run on `http://localhost:3000`

## 🎨 Usage

1. **Open the application** in your browser at `http://localhost:3000`

2. **Choose input method:**
   - **Record**: Click "Start Recording" to record live audio (4 seconds recommended)
   - **Upload**: Click "Upload Audio" to select an audio file

3. **Analyze**: Click "Analyze Emotion" to get predictions

4. **View Results**: See the detected emotion with confidence scores

## 🎵 Supported Audio Formats

### Fully Supported (No Additional Setup)
- **WAV** (.wav) - Recommended
- **OGG** (.ogg)
- **FLAC** (.flac)

### Requires FFmpeg Installation
- **MP3** (.mp3)
- **M4A** (.m4a)
- **AAC** (.aac)
- **WEBM** (.webm)

### M4A File Handling

If you upload an M4A file and get an error:

**Option 1: Install FFmpeg** (Recommended for full support)
```bash
# Using winget (Windows 11)
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements

# Or download from: https://www.gyan.dev/ffmpeg/builds/
```
See [INSTALL_FFMPEG.md](INSTALL_FFMPEG.md) for detailed instructions.

**Option 2: Convert M4A to WAV**
- Use the included converter:
  ```bash
  python convert_audio.py your_file.m4a
  ```
- Or use online converters:
  - [CloudConvert](https://cloudconvert.com/m4a-to-wav)
  - [Convertio](https://convertio.co/m4a-wav/)

**Option 3: Use WAV/OGG files** instead of M4A

## 🧠 Model Information

- **Architecture**: Bidirectional LSTM with Batch Normalization
- **Input Features**: 40-dimensional Mel-spectrogram
- **Sample Rate**: 16,000 Hz
- **Audio Duration**: 4 seconds
- **Training Dataset**: BanglaSER Dataset

### Emotion Classes

| Index | English | Bangla | Color |
|-------|---------|--------|-------|
| 0 | Angry | রাগান্বিত | Red |
| 1 | Happy | খুশি | Green |
| 2 | Sad | দুঃখিত | Blue |
| 3 | Neutral | স্বাভাবিক | Gray |
| 4 | Surprise | অবাক | Purple |

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **PyTorch**: Deep learning framework
- **Torchaudio**: Audio processing
- **Uvicorn**: ASGI server

### Frontend
- **React**: UI library
- **Framer Motion**: Animations
- **Axios**: HTTP client
- **Lucide React**: Icons
- **React Hot Toast**: Notifications

## 📊 API Documentation

### POST /predict

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (audio file)

**Response:**
```json
{
  "success": true,
  "top_emotion": {
    "label_en": "Happy",
    "label_bn": "খুশি",
    "confidence": 87.5,
    "color": "#44ff44"
  },
  "all_emotions": {
    "Angry": {
      "probability": 0.05,
      "percentage": 5.0,
      "label_bn": "রাগান্বিত",
      "label_en": "Angry",
      "color": "#ff4444"
    },
    ...
  }
}
```

## 🔧 Configuration

### Backend Configuration

Edit `backend/.env`:
```env
MODEL_PATH=../bangla_ser_best.pth
HOST=0.0.0.0
PORT=8000
DEVICE=cpu  # or cuda for GPU
```

### Frontend Configuration

Edit `frontend/.env`:
```env
REACT_APP_API_URL=http://localhost:8000
```

## 🚀 Production Deployment

### Backend Deployment

1. **Install production dependencies**
2. **Set environment variables**
3. **Run with Gunicorn:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### Frontend Deployment

1. **Build production bundle:**
```bash
npm run build
```

2. **Serve with a static server** (Nginx, Apache, or serve package)

### Docker Deployment (Optional)

Create `Dockerfile` for containerized deployment.

## 📝 Troubleshooting

### Backend Issues

**Model not loading:**
- Check if `bangla_ser_best.pth` exists
- Verify file path in configuration
- Check PyTorch installation

**CORS errors:**
- Update `allow_origins` in FastAPI middleware
- Check frontend API URL configuration

### Frontend Issues

**API connection failed:**
- Verify backend server is running
- Check API URL in `.env` file
- Check browser console for errors

**Microphone not working:**
- Grant microphone permissions in browser
- Use HTTPS in production (required for microphone)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the MIT License.

## 👨‍💻 Authors

- Developed for Bangla Speech Emotion Recognition research
- Built with ❤️ for Bengali speakers

## 🙏 Acknowledgments

- BanglaSER Dataset creators
- PyTorch and FastAPI communities
- React and open-source contributors

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with 🇧🇩 for the Bangla-speaking community**
