# Quick Start Guide - BanglaSER Web Application

## 🚀 5-Minute Setup

### Step 1: Install Backend Dependencies

Open PowerShell in the backend folder:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies

Open a new PowerShell window in the frontend folder:

```powershell
cd frontend
npm install
```

### Step 3: Start the Application

**Terminal 1 - Backend:**
```powershell
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm start
```

### Step 4: Open Your Browser

Navigate to `http://localhost:3000`

## ✅ Verification Checklist

- [ ] Backend running on http://localhost:8000
- [ ] Frontend running on http://localhost:3000
- [ ] Model file `bangla_ser_best.pth` exists in project root
- [ ] No error messages in terminals

## 🎤 Testing

1. Click "Start Recording" and speak for 2-4 seconds
2. Click "Stop Recording"
3. Click "Analyze Emotion"
4. View your emotion prediction!

## ⚠️ Common Issues

**"Model file not found"**
- Ensure `bangla_ser_best.pth` is in the project root directory

**"Cannot connect to server"**
- Check if backend is running on port 8000
- Verify `REACT_APP_API_URL` in `frontend/.env`

**Microphone not working**
- Allow microphone permissions in browser
- Try uploading a file instead

## 🆘 Need Help?

Check the full README.md for detailed documentation.
