// frontend/src/components/QRDetector.js
import React, { useRef, useEffect, useState } from 'react';

const QRDetector = ({ onQRDetected, isScanning }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [hasCamera, setHasCamera] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const scanIntervalRef = useRef(null);

  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setHasCamera(true);
          setCameraError(null);
        }
        
      } catch (error) {
        console.error('Camera access error:', error);
        setCameraError('Camera access denied or not available');
        setHasCamera(false);
      }
    };

    initCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const startScanning = async () => {
      if (!isScanning || !hasCamera || !videoRef.current) return;

      // Simple QR detection simulation
      scanIntervalRef.current = setInterval(() => {
        try {
          const canvas = canvasRef.current;
          const video = videoRef.current;
          
          if (!canvas || !video || video.readyState !== 4) return;
          
          const ctx = canvas.getContext('2d');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          
          // Simulate QR detection (in real app, use ZXing or similar)
          const imageData = canvas.toDataURL('image/png');
          
          // Mock QR detection for demo
          if (Math.random() < 0.1) { // 10% chance to "detect" QR
            const mockQRData = {
              text: `upi://pay?pa=demo${Math.floor(Math.random() * 1000)}@paytm&pn=Test Merchant&am=${Math.floor(Math.random() * 1000) + 100}`,
              imageData: imageData,
              timestamp: new Date().toISOString()
            };
            
            onQRDetected(mockQRData);
            clearInterval(scanIntervalRef.current);
          }
        } catch (error) {
          console.error('Scanning error:', error);
        }
      }, 500);
    };

    if (isScanning) {
      startScanning();
    } else {
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
      }
    }

    return () => {
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
      }
    };
  }, [isScanning, hasCamera, onQRDetected]);

  if (cameraError) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-2">📷</div>
          <p className="text-gray-600">{cameraError}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry Camera Access
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="relative">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-64 md:h-96 bg-black rounded-lg object-cover"
      />
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Scanner overlay */}
      <div className="absolute inset-4">
        <div className={`scanner-overlay ${isScanning ? 'scan-animation' : ''}`}>
          {/* Corner indicators */}
          <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-blue-500"></div>
          <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-blue-500"></div>
          <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-blue-500"></div>
          <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-blue-500"></div>
        </div>
      </div>
      
      {/* Status indicator */}
      <div className="absolute bottom-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
        {isScanning ? '🔍 Scanning...' : '⏸️ Paused'}
      </div>
      
      {/* Mock detection hint */}
      {isScanning && (
        <div className="absolute top-2 right-2 bg-blue-600 bg-opacity-75 text-white px-3 py-1 rounded text-xs">
          Demo Mode: Random detection
        </div>
      )}
    </div>
  );
};

export default QRDetector;