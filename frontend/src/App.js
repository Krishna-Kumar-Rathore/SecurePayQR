// frontend/src/App.js - FIXED VERSION
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Scanner from './components/Scanner';
import Results from './components/Results';
import Analytics from './components/Analytics';
import About from './components/About';
import { ModelProvider } from './context/ModelContext';
import { DetectionProvider } from './context/DetectionContext';

function App() {
  return (
    <ModelProvider>
      <DetectionProvider>
        <Router>
          <div className="min-h-screen bg-gray-50">
            <Header />
            
            <main className="max-w-7xl mx-auto px-4 py-6">
              <Routes>
                <Route path="/" element={<Scanner />} />
                <Route path="/scanner" element={<Scanner />} />
                <Route path="/results" element={<Results />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/about" element={<About />} />
              </Routes>
            </main>
            
            <footer className="bg-white border-t mt-12">
              <div className="max-w-7xl mx-auto px-4 py-6">
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <div>
                    <p>SecurePayQR - CNN-LSTM Based QR Code Fraud Detection</p>
                    <p>Powered by Deep Learning & Computer Vision</p>
                  </div>
                  <div className="text-right">
                    <p>Model: CNN-LSTM Architecture</p>
                    <p>Inference: ONNX Runtime Web</p>
                  </div>
                </div>
              </div>
            </footer>
          </div>
        </Router>
      </DetectionProvider>
    </ModelProvider>
  );
}

export default App;