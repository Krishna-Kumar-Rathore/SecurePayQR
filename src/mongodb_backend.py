#!/usr/bin/env python3
"""
SecurePayQR: FastAPI Backend with MongoDB
RESTful API for QR code fraud detection with MongoDB database
"""

import os
import io
import json
import time
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# MongoDB imports
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from bson import ObjectId
import bcrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('securepayqr_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('securepayqr_request_duration_seconds', 'Request duration')
DETECTION_COUNT = Counter('securepayqr_detections_total', 'Total detections', ['result'])
MODEL_INFERENCE_TIME = Histogram('securepayqr_model_inference_seconds', 'Model inference time')

# MongoDB Models
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

# Pydantic models
class QRDetectionRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    include_features: bool = Field(False, description="Include intermediate model features")

class QRDetectionResponse(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    is_tampered: bool
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    model_version: str
    timestamp: datetime
    features: Optional[Dict] = None
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class DetectionLog(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: Optional[str] = None
    qr_content: Optional[str] = None
    is_tampered: bool
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    model_version: str
    client_ip: str
    user_agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    features: Optional[Dict] = None
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class User(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password_hash: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class APIStats(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
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

# Database Manager
class MongoDBManager:
    """MongoDB connection and operations manager"""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URL', 
            'mongodb://mongodb:27017'
        )
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.is_connected = False
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            # Test connection
            await self.client.admin.command('ping')
            self.database = self.client.securepayqr
            self.is_connected = True
            
            # Create indexes
            await self.create_indexes()
            
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("Disconnected from MongoDB")
    
    async def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Users collection indexes
            await self.database.users.create_index("username", unique=True)
            await self.database.users.create_index("email", unique=True)
            
            # Detection logs indexes
            await self.database.detection_logs.create_index("user_id")
            await self.database.detection_logs.create_index("timestamp")
            await self.database.detection_logs.create_index("is_tampered")
            
            # API stats indexes
            await self.database.api_stats.create_index("endpoint")
            await self.database.api_stats.create_index("timestamp")
            await self.database.api_stats.create_index([("endpoint", 1), ("timestamp", -1)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    async def create_default_user(self):
        """Create default admin user"""
        try:
            existing_user = await self.database.users.find_one({"username": "admin"})
            if not existing_user:
                password_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
                admin_user = User(
                    username="admin",
                    email="admin@securepayqr.com",
                    password_hash=password_hash
                )
                
                await self.database.users.insert_one(admin_user.model_dump(by_alias=True))
                logger.info("Default admin user created")
        except DuplicateKeyError:
            logger.info("Admin user already exists")
        except Exception as e:
            logger.error(f"Error creating default user: {e}")

# Model Manager (same as before but with MongoDB logging)
class SecurePayQRModelManager:
    """Manages ML model loading and inference with MongoDB logging"""
    
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
    
    async def detect_tampering(self, image: Image.Image, log_to_db: bool = True) -> Dict:
        """Run inference on image and optionally log to MongoDB"""
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
                'timestamp': datetime.now(timezone.utc)
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
    title="SecurePayQR API with MongoDB",
    description="AI-powered QR code fraud detection service with MongoDB backend",
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
db_manager = MongoDBManager()
app_start_time = time.time()

# Dependency to get database
async def get_database():
    if not db_manager.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected")
    return db_manager.database

# Middleware for metrics and API stats logging
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    # Log API stats to MongoDB (fire and forget)
    if db_manager.is_connected:
        try:
            api_stat = APIStats(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=duration * 1000,
                request_size_bytes=int(request.headers.get('content-length', 0)),
                response_size_bytes=len(response.body) if hasattr(response, 'body') else None
            )
            
            # Insert without waiting (fire and forget)
            asyncio.create_task(
                db_manager.database.api_stats.insert_one(api_stat.model_dump(by_alias=True))
            )
        except Exception as e:
            logger.warning(f"Failed to log API stats: {e}")
    
    return response

# Authentication (basic implementation)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token validation - implement proper JWT for production"""
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
        database_connected=db_manager.is_connected,
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
    request: Request,
    file: UploadFile = File(..., description="QR code image file"),
    include_features: bool = False,
    user = Depends(get_current_user),
    db = Depends(get_database)
):
    """Detect QR code tampering from uploaded image and log to MongoDB"""
    
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
        
        # Log detection to MongoDB
        detection_log = DetectionLog(
            user_id=user.get('user_id') if user else None,
            qr_content=None,  # Could extract QR content here
            is_tampered=result['is_tampered'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time_ms=result['processing_time_ms'],
            model_version=result['model_version'],
            client_ip=request.client.host,
            user_agent=request.headers.get('user-agent'),
            timestamp=result['timestamp'],
            features=result.get('features')
        )
        
        # Insert detection log
        insert_result = await db.detection_logs.insert_one(detection_log.model_dump(by_alias=True))
        result['id'] = str(insert_result.inserted_id)
        
        return QRDetectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect/batch")
async def detect_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    user = Depends(get_current_user),
    db = Depends(get_database)
):
    """Batch detection for multiple QR codes"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    detection_logs = []
    
    for i, file in enumerate(files):
        try:
            content = await file.read()
            image = Image.open(io.BytesIO(content))
            result = await model_manager.detect_tampering(image)
            result['file_index'] = i
            result['filename'] = file.filename
            
            # Create detection log
            detection_log = DetectionLog(
                user_id=user.get('user_id') if user else None,
                is_tampered=result['is_tampered'],
                confidence=result['confidence'],
                probabilities=result['probabilities'],
                processing_time_ms=result['processing_time_ms'],
                model_version=result['model_version'],
                client_ip=request.client.host,
                user_agent=request.headers.get('user-agent'),
                timestamp=result['timestamp']
            )
            
            detection_logs.append(detection_log.model_dump(by_alias=True))
            results.append(result)
            
        except Exception as e:
            results.append({
                'file_index': i,
                'filename': file.filename,
                'error': str(e),
                'is_tampered': None,
                'confidence': 0.0
            })
    
    # Bulk insert detection logs
    if detection_logs:
        await db.detection_logs.insert_many(detection_logs)
    
    return {'results': results}

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/stats")
async def get_statistics(db = Depends(get_database)):
    """Get API usage statistics from MongoDB"""
    try:
        # Aggregate statistics
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_detections": {"$sum": 1},
                    "tampered_count": {
                        "$sum": {"$cond": [{"$eq": ["$is_tampered", True]}, 1, 0]}
                    },
                    "avg_processing_time": {"$avg": "$processing_time_ms"},
                    "avg_confidence": {"$avg": "$confidence"}
                }
            }
        ]
        
        detection_stats = await db.detection_logs.aggregate(pipeline).to_list(1)
        
        # API request statistics
        api_pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_requests": {"$sum": 1},
                    "avg_response_time": {"$avg": "$response_time_ms"}
                }
            }
        ]
        
        api_stats = await db.api_stats.aggregate(api_pipeline).to_list(1)
        
        stats = {
            'total_requests': api_stats[0]['total_requests'] if api_stats else 0,
            'total_detections': detection_stats[0]['total_detections'] if detection_stats else 0,
            'tampered_detected': detection_stats[0]['tampered_count'] if detection_stats else 0,
            'average_processing_time_ms': detection_stats[0]['avg_processing_time'] if detection_stats else 0.0,
            'average_confidence': detection_stats[0]['avg_confidence'] if detection_stats else 0.0,
            'model_accuracy': 0.95,  # From validation
            'uptime_hours': (time.time() - app_start_time) / 3600
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {
            'total_requests': 0,
            'total_detections': 0,
            'tampered_detected': 0,
            'average_processing_time_ms': 0.0,
            'average_confidence': 0.0,
            'model_accuracy': 0.95,
            'uptime_hours': (time.time() - app_start_time) / 3600
        }

@app.get("/detection-logs")
async def get_detection_logs(
    skip: int = 0,
    limit: int = 100,
    user = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get detection logs with pagination"""
    try:
        cursor = db.detection_logs.find().sort("timestamp", -1).skip(skip).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string
        for log in logs:
            log['_id'] = str(log['_id'])
            
        return {'logs': logs, 'skip': skip, 'limit': limit}
        
    except Exception as e:
        logger.error(f"Failed to get detection logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("SecurePayQR API starting up...")
    
    # Connect to MongoDB
    await db_manager.connect()
    
    # Create default user
    if db_manager.is_connected:
        await db_manager.create_default_user()
    
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Log model status
    if model_manager.is_loaded:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded - detection endpoints will not work")
    
    logger.info("Startup completed successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("SecurePayQR API shutting down...")
    await db_manager.disconnect()

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )