import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, Upload, Loader, Sparkles, Volume2, X } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';
import './EmotionDetector.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const EmotionDetector = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [audioURL, setAudioURL] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);
  const recordingTimerRef = useRef(null);

  // Convert audio blob to WAV format
  const convertToWav = async (audioBlob) => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Convert to WAV
    const wavBuffer = audioBufferToWav(audioBuffer);
    const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
    
    return wavBlob;
  };

  // AudioBuffer to WAV converter
  const audioBufferToWav = (buffer) => {
    const length = buffer.length * buffer.numberOfChannels * 2 + 44;
    const arrayBuffer = new ArrayBuffer(length);
    const view = new DataView(arrayBuffer);
    const channels = [];
    let offset = 0;
    let pos = 0;

    // Write WAV header
    const setUint16 = (data) => {
      view.setUint16(pos, data, true);
      pos += 2;
    };
    const setUint32 = (data) => {
      view.setUint32(pos, data, true);
      pos += 4;
    };

    // "RIFF" chunk descriptor
    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8); // file length - 8
    setUint32(0x45564157); // "WAVE"

    // "fmt " sub-chunk
    setUint32(0x20746d66); // "fmt "
    setUint32(16); // SubChunk1Size (16 for PCM)
    setUint16(1); // AudioFormat (1 for PCM)
    setUint16(buffer.numberOfChannels);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * buffer.numberOfChannels * 2); // byte rate
    setUint16(buffer.numberOfChannels * 2); // block align
    setUint16(16); // bits per sample

    // "data" sub-chunk
    setUint32(0x61746164); // "data"
    setUint32(length - pos - 4); // SubChunk2Size

    // Write interleaved data
    for (let i = 0; i < buffer.numberOfChannels; i++) {
      channels.push(buffer.getChannelData(i));
    }

    while (pos < length) {
      for (let i = 0; i < buffer.numberOfChannels; i++) {
        let sample = Math.max(-1, Math.min(1, channels[i][offset]));
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(pos, sample, true);
        pos += 2;
      }
      offset++;
    }

    return arrayBuffer;
  };

  // Start Recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        } 
      });
      
      // Use WebM format for recording (better browser support)
      const options = { mimeType: 'audio/webm' };
      mediaRecorderRef.current = new MediaRecorder(stream, options);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        // Create blob from recorded chunks
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        
        // Convert to WAV format
        try {
          const wavBlob = await convertToWav(audioBlob);
          const url = URL.createObjectURL(wavBlob);
          setAudioURL(url);
          setAudioFile(wavBlob);
          
          toast.success('✅ Recording saved! Click "Analyze Emotion" to detect emotion.');
          
        } catch (conversionError) {
          console.error('WAV conversion error:', conversionError);
          // Fallback: use original blob
          const url = URL.createObjectURL(audioBlob);
          setAudioURL(url);
          setAudioFile(audioBlob);
          toast.success('✅ Recording saved! Click "Analyze Emotion" to detect emotion.');
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
      toast.success('🎙️ Recording started');
    } catch (error) {
      toast.error('Microphone access denied');
      console.error('Error accessing microphone:', error);
    }
  };

  // Stop Recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
      
      // Stop timer
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
      
      toast.success('🎤 Recording stopped - Ready to analyze');
    }
  };

  // Handle File Upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast.error('File too large. Max 10MB');
        return;
      }
      
      const url = URL.createObjectURL(file);
      setAudioURL(url);
      setAudioFile(file);
      setResult(null);
      toast.success('File uploaded successfully');
    }
  };

  // Analyze Audio
  const analyzeAudio = async (audioBlob = audioFile) => {
    if (!audioBlob) {
      toast.error('Please record or upload audio first');
      return;
    }

    setIsProcessing(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.wav');

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      });

      if (response.data.success) {
        setResult(response.data);
        toast.success('🎉 Emotion detected!');
      } else {
        toast.error('Failed to analyze audio');
      }
    } catch (error) {
      console.error('Error analyzing audio:', error);
      let errorMessage = 'Error analyzing audio';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Request timeout. Try again.';
      } else if (error.response && error.response.data) {
        // Get the detailed error message from backend
        errorMessage = error.response.data.detail || 'Server error';
      } else if (error.request) {
        errorMessage = 'Cannot connect to server';
      }
      
      // Display error in toast
      toast.error(errorMessage, {
        duration: 6000,
        style: {
          maxWidth: '500px',
        }
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Reset
  const resetApp = () => {
    setAudioFile(null);
    setAudioURL(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="emotion-detector">
      {/* Recording Indicator Banner */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            className="recording-banner"
            initial={{ y: -100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -100, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="recording-banner-content">
              <div className="recording-dot"></div>
              <span className="recording-banner-text">
                🎙️ RECORDING - {String(Math.floor(recordingTime / 60)).padStart(1, '0')}:{String(recordingTime % 60).padStart(2, '0')}
              </span>
              <Mic size={20} className="recording-banner-icon" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <motion.div 
        className="header"
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="logo">
          <Sparkles size={32} className="logo-icon" />
          <h1>BanglaSER</h1>
        </div>
        <p className="subtitle">Bangla Speech Emotion Recognition</p>
        <p className="tagline">🇧🇩 AI-powered voice emotion detection for Bengali speakers</p>
      </motion.div>

      {/* Main Card */}
      <motion.div 
        className="main-card"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        {/* Upload Section */}
        <div className="upload-section">
          <h2>Record or Upload Audio</h2>
          
          <div className="button-group">
            {/* Record Button */}
            <motion.button
              className={`action-btn record-btn ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Mic size={24} className={isRecording ? 'mic-pulse' : ''} />
              {isRecording ? (
                <span className="recording-text">
                  <span className="recording-dot-inline">●</span>
                  <span>REC {String(Math.floor(recordingTime / 60)).padStart(1, '0')}:{String(recordingTime % 60).padStart(2, '0')}</span>
                </span>
              ) : (
                'Start Recording'
              )}
            </motion.button>

            {/* Upload Button */}
            <motion.button
              className="action-btn upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={isRecording || isProcessing}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Upload size={24} />
              Upload Audio
            </motion.button>
            
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/wav,audio/mpeg,audio/mp3,audio/mp4,audio/x-m4a,audio/ogg,audio/flac,audio/webm,audio/*"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
          </div>

          {/* Audio Player */}
          <AnimatePresence>
            {audioURL && (
              <motion.div 
                className="audio-player-container"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <div className="audio-player">
                  <Volume2 size={20} />
                  <audio controls src={audioURL} className="audio-element" />
                  <button className="reset-btn" onClick={resetApp}>
                    <X size={18} />
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Analyze Button */}
          {audioFile && !result && !isProcessing && (
            <motion.button
              className="analyze-btn"
              onClick={() => analyzeAudio()}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <Sparkles size={20} />
              Analyze Emotion
            </motion.button>
          )}
        </div>

        {/* Processing State */}
        <AnimatePresence>
          {isProcessing && (
            <motion.div 
              className="processing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Loader className="spinner" size={40} />
              <p>Analyzing emotion...</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        <AnimatePresence>
          {result && (
            <motion.div 
              className="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
            >
              <h2>Detection Results</h2>
              
              {/* Top Emotion */}
              <motion.div 
                className="top-emotion"
                style={{ borderColor: result.top_emotion.color }}
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2 }}
              >
                <div className="emotion-badge" style={{ background: result.top_emotion.color }}>
                  {result.top_emotion.label_en}
                </div>
                <div className="emotion-label">
                  <span className="emotion-en">{result.top_emotion.label_en}</span>
                  <span className="emotion-bn">{result.top_emotion.label_bn}</span>
                </div>
                <div className="confidence">
                  <span className="confidence-value">{result.top_emotion.confidence}%</span>
                  <span className="confidence-label">Confidence</span>
                </div>
              </motion.div>

              {/* All Emotions */}
              <div className="all-emotions">
                <h3>Detailed Analysis</h3>
                <div className="emotion-bars">
                  {Object.entries(result.all_emotions)
                    .sort((a, b) => b[1].percentage - a[1].percentage)
                    .map(([emotion, data], index) => (
                      <motion.div
                        key={emotion}
                        className="emotion-bar"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <div className="emotion-info">
                          <span className="emotion-name">
                            {data.label_en} <span className="emotion-bn-small">({data.label_bn})</span>
                          </span>
                          <span className="emotion-percentage">{data.percentage}%</span>
                        </div>
                        <div className="bar-container">
                          <motion.div 
                            className="bar-fill"
                            style={{ backgroundColor: data.color }}
                            initial={{ width: 0 }}
                            animate={{ width: `${data.percentage}%` }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                          />
                        </div>
                      </motion.div>
                    ))}
                </div>
              </div>

              {/* Try Again Button */}
              <motion.button
                className="try-again-btn"
                onClick={resetApp}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Try Another Audio
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Footer */}
      <motion.div 
        className="footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <p>
          Developed by{' '}
          <a href="mailto:bsse1302@iit.du.ac.bd" className="developer-link">Swadhin(1302)</a>,{' '}
          <a href="mailto:bsse1314@iit.du.ac.bd" className="developer-link">Amit(1314)</a>,{' '}
          <a href="mailto:bsse1325@iit.du.ac.bd" className="developer-link">Rony(1325)</a>. 
          Feel free to contact with us
        </p>
      </motion.div>
    </div>
  );
};

export default EmotionDetector;
