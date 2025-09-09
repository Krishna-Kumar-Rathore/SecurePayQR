#!/usr/bin/env python3
"""
SecurePayQR: FastAPI Backend
RESTful API for QR code fraud detection with model serving
"""

import os
import io
import json
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, List
import logging

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('securepayqr_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('securepayqr_request_duration_seconds', 'Request duration')
DETECTION_COUNT = Counter('securepayqr_detections_total', 'Total detections', ['result'])
MODEL_INFERENCE_TIME = Histogram('securepayqr_model_inference_seconds', 'Model inference time')

# Pydantic models
class QRDetectionRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    include_features: bool = Field(False, description="Include intermediate model features")

class QRDetectionResponse(BaseModel):
    is_tampered: bool
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    model_version: str
    timestamp: str
    features: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float

class ModelInfo(BaseModel):
    model_name: str
    version: str
    architecture: str
    input_shape: List[int]
    output_shape: List[int]
    parameters_count: int
    model_size_mb: float

# Model Manager
class SecurePayQRModelManager:
    """Manages ML model loading and inference"""
    
    def __init__(self, model_path: str = "models/securepayqr_model.onnx"):
        self.model_path = model_path
        self.session = None
        self.is_loaded = False
        self.model_info = {}
        self.load_model()
    
    def load_model(self):
        """Load ONNX model for inference"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return
            
            # Load ONNX model
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Load model metadata
            model_info_path = Path(self.model_path).parent / "model_info.json"
            if model_info_path.exists():
                with open(model_info_path, 'r') as f:
                    self.model_info = json.load(f)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize to model input size
        image = image.convert('RGB')
        image = image.resize((256, 256))
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Add batch dimension and transpose to NCHW
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    async def detect_tampering(self, image: Image.Image) -> Dict:
        """Run inference on image"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            input_array = self.preprocess_image(image)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_array})
            
            # Post-process outputs
            logits = outputs[0]
            probabilities = self._softmax(logits[0])
            
            prediction = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            inference_time = time.time() - start_time
            MODEL_INFERENCE_TIME.observe(inference_time)
            
            result = {
                'is_tampered': bool(prediction == 1),
                'confidence': confidence,
                'probabilities': {
                    'valid': float(probabilities[0]),
                    'tampered': float(probabilities[1])
                },
                'processing_time_ms': inference_time * 1000,
                'model_version': self.model_info.get('version', '1.0'),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update metrics
            DETECTION_COUNT.labels(result='tampered' if result['is_tampered'] else 'valid').inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    
    @staticmethod
    def _softmax(x):
        """Apply softmax to logits"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

# Initialize FastAPI app
app = FastAPI(
    title="SecurePayQR API",
    description="AI-powered QR code fraud detection service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global variables
model_manager = SecurePayQRModelManager()
app_start_time = time.time()

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

# Authentication (basic implementation)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token validation - implement proper auth for production"""
    if not credentials:
        return None
    
    # In production, validate JWT token here
    token = credentials.credentials
    if token != "demo-token":  # Replace with proper validation
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"user_id": "demo_user"}

# Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
        version="1.0.0",
        uptime_seconds=uptime
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = model_manager.model_info
    return ModelInfo(
        model_name="SecurePayQR CNN-LSTM",
        version=info.get('version', '1.0'),
        architecture="CNN-LSTM",
        input_shape=info.get('input_shape', [1, 3, 256, 256]),
        output_shape=info.get('output_shape', [1, 2]),
        parameters_count=info.get('total_parameters', 0),
        model_size_mb=info.get('model_size_mb', 0.0)
    )

@app.post("/detect", response_model=QRDetectionResponse)
async def detect_qr_tampering(
    file: UploadFile = File(..., description="QR code image file"),
    include_features: bool = False,
    user = Depends(get_current_user)
):
    """Detect QR code tampering from uploaded image"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    try:
        # Load and validate image
        image = Image.open(io.BytesIO(content))
        
        # Run detection
        result = await model_manager.detect_tampering(image)
        
        # Add features if requested (placeholder for demo)
        if include_features:
            result['features'] = {
                'cnn_features_shape': [1, 512],
                'lstm_features_shape': [1, 512],
                'feature_extraction_time_ms': 50.0
            }
        
        return QRDetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    user = Depends(get_current_user)
):
    """Batch detection for multiple QR codes"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            content = await file.read()
            image = Image.open(io.BytesIO(content))
            result = await model_manager.detect_tampering(image)
            result['file_index'] = i
            result['filename'] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                'file_index': i,
                'filename': file.filename,
                'error': str(e),
                'is_tampered': None,
                'confidence': 0.0
            })
    
    return {'results': results}

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/stats")
async def get_statistics():
    """Get API usage statistics"""
    # This would typically query a database
    return {
        'total_requests': 0,  # Replace with actual metrics
        'total_detections': 0,
        'tampered_detected': 0,
        'average_processing_time_ms': 0.0,
        'model_accuracy': 0.95,  # From validation
        'uptime_hours': (time.time() - app_start_time) / 3600
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("SecurePayQR API starting up...")
    
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Log model status
    if model_manager.is_loaded:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded - detection endpoints will not work")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("SecurePayQR API shutting down...")

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )