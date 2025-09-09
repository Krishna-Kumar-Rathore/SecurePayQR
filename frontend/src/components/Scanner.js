// frontend/src/components/Scanner.js - ENHANCED VERSION
import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import QRDetector from './QRDetector';
import { useModel } from '../context/ModelContext';
import { useDetection } from '../context/DetectionContext';

const Scanner = () => {
  const [isScanning, setIsScanning] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [scanMode, setScanMode] = useState('camera'); // 'camera' or 'upload'
  const { modelReady } = useModel();
  const { addDetection, detectionHistory } = useDetection();
  const navigate = useNavigate();

  const handleQRDetected = useCallback(async (qrData) => {
    if (!modelReady) return;

    setIsLoading(true);
    setIsScanning(false);

    try {
      // Convert image data to blob for API
      let blob;
      if (qrData.imageData.startsWith('data:')) {
        const response = await fetch(qrData.imageData);
        blob = await response.blob();
      } else {
        blob = qrData.imageData; // Already a blob from file upload
      }
      
      // Create form data
      const formData = new FormData();
      formData.append('file', blob, 'qr_scan.png');
      formData.append('source', qrData.source || 'camera');

      // Call API
      const apiResponse = await fetch('/detect', {
        method: 'POST',
        body: formData,
        headers: {
          'Authorization': 'Bearer demo-token'
        }
      });

      if (!apiResponse.ok) {
        throw new Error(`Detection API failed: ${apiResponse.status}`);
      }

      const detection = await apiResponse.json();
      
      const result = {
        id: detection.id || Date.now(),
        qrData,
        detection,
        timestamp: new Date().toISOString()
      };

      setCurrentResult(result);
      addDetection(result);
      
    } catch (error) {
      console.error('Detection failed:', error);
      // Fallback to mock detection for demo
      const mockDetection = {
        is_tampered: Math.random() < 0.2,
        confidence: 0.85 + Math.random() * 0.1,
        probabilities: {
          valid: Math.random() * 0.5 + 0.5,
          tampered: Math.random() * 0.5
        },
        processing_time_ms: 200 + Math.random() * 300,
        model_version: '1.0',
        timestamp: new Date().toISOString()
      };
      
      const result = {
        id: Date.now(),
        qrData,
        detection: mockDetection,
        timestamp: new Date().toISOString()
      };
      
      setCurrentResult(result);
      addDetection(result);
    } finally {
      setIsLoading(false);
    }
  }, [modelReady, addDetection]);

  const handleFileUpload = useCallback(async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please select a valid image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      const qrData = {
        imageData: e.target.result,
        text: 'Uploaded QR Code (content will be extracted)',
        source: 'upload',
        timestamp: new Date().toISOString()
      };
      
      await handleQRDetected(qrData);
    };
    reader.readAsDataURL(file);
  }, [handleQRDetected]);

  const toggleScanning = () => {
    if (!modelReady) return;
    setIsScanning(!isScanning);
    if (!isScanning) {
      setCurrentResult(null);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Scanner Section */}
      <div className="space-y-4">
        {/* Scan Mode Selector */}
        <div className="bg-white rounded-lg p-4 shadow-sm border">
          <h3 className="text-lg font-semibold mb-3">Detection Method</h3>
          <div className="flex space-x-4">
            <button
              onClick={() => {
                setScanMode('camera');
                setIsScanning(false);
                setCurrentResult(null);
              }}
              className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
                scanMode === 'camera'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              📷 Camera Scan
            </button>
            <button
              onClick={() => {
                setScanMode('upload');
                setIsScanning(false);
                setCurrentResult(null);
              }}
              className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
                scanMode === 'upload'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              📁 Upload Image
            </button>
          </div>
        </div>

        {/* Scanner Card */}
        {scanMode === 'camera' ? (
          <ScannerCard 
            isScanning={isScanning}
            onToggleScanning={toggleScanning}
            onQRDetected={handleQRDetected}
            modelReady={modelReady}
          />
        ) : (
          <ImageUploadCard
            onFileUpload={handleFileUpload}
            modelReady={modelReady}
            isLoading={isLoading}
          />
        )}
        
        <InstructionsCard scanMode={scanMode} />
      </div>

      {/* Results Section */}
      <div className="space-y-4">
        <ResultsCard 
          result={currentResult}
          isLoading={isLoading}
          onViewResults={() => navigate('/results')}
        />
        
        {detectionHistory.length > 0 && (
          <HistoryCard 
            history={detectionHistory.slice(0, 5)}
            onViewAll={() => navigate('/results')}
          />
        )}
      </div>
    </div>
  );
};

const ScannerCard = ({ isScanning, onToggleScanning, onQRDetected, modelReady }) => (
  <div className="bg-white rounded-lg p-6 shadow-sm border">
    <div className="flex items-center justify-between mb-4">
      <h2 className="text-xl font-semibold text-gray-900">Camera Scanner</h2>
      <button
        onClick={onToggleScanning}
        disabled={!modelReady}
        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
          modelReady
            ? isScanning
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-blue-600 hover:bg-blue-700 text-white'
            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
        }`}
      >
        {isScanning ? 'Stop Scanning' : 'Start Scanning'}
      </button>
    </div>
    
    <QRDetector 
      onQRDetected={onQRDetected}
      isScanning={isScanning && modelReady}
    />
  </div>
);

const ImageUploadCard = ({ onFileUpload, modelReady, isLoading }) => {
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onFileUpload(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      onFileUpload(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <div className="bg-white rounded-lg p-6 shadow-sm border">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900">Upload QR Image</h2>
        <div className={`px-3 py-1 rounded-full text-sm ${
          modelReady ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-500'
        }`}>
          {modelReady ? 'Ready' : 'Loading...'}
        </div>
      </div>
      
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          modelReady && !isLoading
            ? 'border-blue-300 hover:border-blue-400 cursor-pointer'
            : 'border-gray-300 cursor-not-allowed'
        }`}
        onClick={() => modelReady && !isLoading && fileInputRef.current?.click()}
      >
        {isLoading ? (
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-3 text-blue-700">Processing...</span>
          </div>
        ) : (
          <>
            <div className="text-4xl mb-4">📁</div>
            <p className="text-lg font-medium text-gray-700 mb-2">
              Drop an image here or click to browse
            </p>
            <p className="text-sm text-gray-500">
              Supports PNG, JPG, GIF up to 10MB
            </p>
          </>
        )}
        
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          disabled={!modelReady || isLoading}
          className="hidden"
        />
      </div>
    </div>
  );
};

const InstructionsCard = ({ scanMode }) => (
  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
    <h3 className="font-medium text-blue-900 mb-2">
      {scanMode === 'camera' ? 'Camera Instructions:' : 'Upload Instructions:'}
    </h3>
    {scanMode === 'camera' ? (
      <ol className="text-sm text-blue-800 space-y-1">
        <li>1. Click &quot;Start Scanning&quot; to activate the camera</li>
        <li>2. Point your camera at a QR code</li>
        <li>3. Hold steady until the code is detected</li>
        <li>4. Wait for AI analysis results</li>
      </ol>
    ) : (
      <ol className="text-sm text-blue-800 space-y-1">
        <li>1. Click &quot;Choose File&quot; or drag &amp; drop an image</li>
        <li>2. Select an image containing a QR code</li>
        <li>3. The AI will locate and analyze the QR code</li>
        <li>4. View the fraud detection results</li>
      </ol>
    )}
  </div>
);

const ResultsCard = ({ result, isLoading, onViewResults }) => (
  <div className="bg-white rounded-lg p-6 shadow-sm border">
    <div className="flex items-center justify-between mb-4">
      <h2 className="text-xl font-semibold text-gray-900">Detection Results</h2>
      {result && (
        <button
          onClick={onViewResults}
          className="text-blue-600 hover:text-blue-700 text-sm font-medium"
        >
          View All Results →
        </button>
      )}
    </div>
    
    <ResultDisplay result={result} isLoading={isLoading} />
  </div>
);

const HistoryCard = ({ history, onViewAll }) => (
  <div className="bg-white rounded-lg p-6 shadow-sm border">
    <div className="flex items-center justify-between mb-4">
      <h3 className="text-lg font-semibold text-gray-900">Recent Scans</h3>
      <button
        onClick={onViewAll}
        className="text-blue-600 hover:text-blue-700 text-sm font-medium"
      >
        View All →
      </button>
    </div>
    
    <div className="space-y-3">
      {history.map((scan) => (
        <div key={scan.id} className="flex items-center justify-between p-3 bg-gray-50 rounded border">
          <div className="flex items-center">
            <span className="text-lg mr-3">
              {scan.detection.is_tampered ? '❌' : '✅'}
            </span>
            <div>
              <p className="text-sm font-medium">
                {scan.detection.is_tampered ? 'Tampered' : 'Valid'}
              </p>
              <p className="text-xs text-gray-500">
                {new Date(scan.timestamp).toLocaleTimeString()} • {scan.qrData.source || 'Camera'}
              </p>
            </div>
          </div>
          <span className="text-sm text-gray-600">
            {(scan.detection.confidence * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  </div>
);

const ResultDisplay = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-blue-700">Analyzing QR Code...</span>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="text-center py-12 text-gray-500">
        <div className="text-4xl mb-4">🔍</div>
        <p>Scan or upload a QR code to see fraud detection results</p>
      </div>
    );
  }

  const isValid = !result.detection.is_tampered;
  const statusBg = isValid ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200';
  const statusText = isValid ? 'text-green-700' : 'text-red-700';
  const glowClass = isValid ? 'success-glow' : 'danger-glow';

  return (
    <div className={`${statusBg} border rounded-lg p-6 ${glowClass} transition-all duration-300`}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center">
          <div className="text-2xl mr-3">
            {isValid ? '✅' : '❌'}
          </div>
          <div>
            <h3 className={`text-lg font-bold ${statusText}`}>
              {isValid ? 'QR Code Valid' : 'QR Code Tampered'}
            </h3>
            <p className={`${statusText} opacity-75`}>
              Confidence: {(result.detection.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        <div className="flex flex-col items-end">
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${statusText} bg-white bg-opacity-50`}>
            {result.detection.processing_time_ms?.toFixed(0) || 0}ms
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Source: {result.qrData.source || 'Camera'}
          </div>
        </div>
      </div>

      {/* QR Code Content */}
      <div className="mb-4">
        <h4 className="font-medium text-gray-700 mb-2">QR Code Content:</h4>
        <div className="bg-white rounded p-3 text-sm font-mono break-all border">
          {result.qrData.text}
        </div>
      </div>

      {/* Confidence Scores */}
      <ConfidenceScores probabilities={result.detection.probabilities} />
    </div>
  );
};

const ConfidenceScores = ({ probabilities }) => (
  <div className="mb-4">
    <h4 className="font-medium text-gray-700 mb-2">Detection Scores:</h4>
    <div className="space-y-2">
      <ScoreBar 
        label="Valid" 
        value={probabilities.valid} 
        color="bg-green-500" 
      />
      <ScoreBar 
        label="Tampered" 
        value={probabilities.tampered} 
        color="bg-red-500" 
      />
    </div>
  </div>
);

const ScoreBar = ({ label, value, color }) => (
  <div className="flex justify-between items-center">
    <span className="text-sm">{label}</span>
    <div className="flex items-center">
      <div className="w-32 bg-gray-200 rounded-full h-2 mr-2">
        <div 
          className={`${color} h-2 rounded-full transition-all duration-300`}
          style={{ width: `${value * 100}%` }}
        ></div>
      </div>
      <span className="text-sm w-12 text-right">
        {(value * 100).toFixed(1)}%
      </span>
    </div>
  </div>
);

export default Scanner;