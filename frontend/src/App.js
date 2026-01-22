import React from 'react';
import { Toaster } from 'react-hot-toast';
import EmotionDetector from './components/EmotionDetector';
import './App.css';

function App() {
  return (
    <div className="App">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 3000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
      <EmotionDetector />
    </div>
  );
}

export default App;
