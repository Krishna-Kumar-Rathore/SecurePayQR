// frontend/src/context/DetectionContext.js
import React, { createContext, useContext, useState, useEffect } from 'react';

const DetectionContext = createContext();

export const useDetection = () => {
  const context = useContext(DetectionContext);
  if (!context) {
    throw new Error('useDetection must be used within a DetectionProvider');
  }
  return context;
};

export const DetectionProvider = ({ children }) => {
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [statistics, setStatistics] = useState({
    totalScans: 0,
    validScans: 0,
    tamperedScans: 0,
    averageConfidence: 0,
    averageProcessingTime: 0
  });

  // Safe localStorage operations with error handling
  const saveToLocalStorage = (data) => {
    try {
      // Remove image data to save space
      const cleanedData = data.map(result => ({
        ...result,
        qrData: {
          ...result.qrData,
          imageData: undefined // Remove large base64 image data
        }
      }));

      localStorage.setItem('securepayqr-detections', JSON.stringify(cleanedData));
    } catch (error) {
      if (error.name === 'QuotaExceededError') {
        console.warn('Storage quota exceeded, keeping only recent results');
        // Keep only 5 most recent results
        const limitedData = data.slice(0, 5).map(result => ({
          ...result,
          qrData: {
            ...result.qrData,
            imageData: undefined
          }
        }));
        
        try {
          localStorage.setItem('securepayqr-detections', JSON.stringify(limitedData));
          // Update state to match what we actually stored
          setDetectionHistory(limitedData);
        } catch (secondError) {
          console.error('Failed to save even limited data:', secondError);
          // Clear everything if still failing
          localStorage.removeItem('securepayqr-detections');
        }
      } else {
        console.error('Failed to save to localStorage:', error);
      }
    }
  };

  // Load from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('securepayqr-detections');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setDetectionHistory(parsed);
      } catch (error) {
        console.error('Failed to parse stored detections:', error);
        // Clear corrupted data
        localStorage.removeItem('securepayqr-detections');
      }
    }
  }, []);

  // Save to localStorage when detections change (with size limit)
  useEffect(() => {
    if (detectionHistory.length > 0) {
      // Limit to maximum 20 results to prevent storage issues
      const limitedHistory = detectionHistory.slice(0, 20);
      
      // Only save if we had to limit, update state
      if (limitedHistory.length !== detectionHistory.length) {
        setDetectionHistory(limitedHistory);
      } else {
        saveToLocalStorage(limitedHistory);
      }
      
      updateStatistics();
    }
  }, [detectionHistory]);

  const updateStatistics = () => {
    const totalScans = detectionHistory.length;
    const validScans = detectionHistory.filter(d => !d.detection.is_tampered).length;
    const tamperedScans = totalScans - validScans;
    
    const totalConfidence = detectionHistory.reduce((sum, d) => sum + d.detection.confidence, 0);
    const averageConfidence = totalScans > 0 ? totalConfidence / totalScans : 0;
    
    const totalProcessingTime = detectionHistory.reduce((sum, d) => sum + (d.detection.processing_time_ms || 0), 0);
    const averageProcessingTime = totalScans > 0 ? totalProcessingTime / totalScans : 0;

    setStatistics({
      totalScans,
      validScans,
      tamperedScans,
      averageConfidence,
      averageProcessingTime
    });
  };

  // Add new detection (properly integrated with React state)
  const addDetection = (newDetection) => {
    setDetectionHistory(prevHistory => {
      // Add new detection to the beginning, limit to 20 results
      const updated = [newDetection, ...prevHistory].slice(0, 20);
      return updated;
    });
  };

  const clearHistory = () => {
    setDetectionHistory([]);
    localStorage.removeItem('securepayqr-detections');
  };

  const value = {
    detectionHistory,
    statistics,
    addDetection,
    clearHistory
  };

  return (
    <DetectionContext.Provider value={value}>
      {children}
    </DetectionContext.Provider>
  );
};