# SecurePayQR: AI-Powered QR Code Fraud Detection

AI-powered QR code fraud detection system using CNN-LSTM deep learning architecture to identify tampered and malicious QR codes in real-time. Features a FastAPI backend, React frontend with camera integration, and comprehensive monitoring for secure payment verification.

## Key Features

- **CNN-LSTM Hybrid Model**: Advanced deep learning architecture for high-accuracy fraud detection (96%+ accuracy)
- **Real-time Detection**: Sub-second inference with ONNX Runtime optimization
- **Web Interface**: React frontend with camera scanning and file upload capabilities
- **Production Ready**: FastAPI backend, PostgreSQL database, Docker containerization
- **Monitoring**: Prometheus metrics and Grafana dashboards

## Tech Stack

**Backend**: Python, PyTorch, FastAPI, MongoDB, Docker 
**Frontend**: React, Tailwind CSS, WebRTC Camera API
**Infrastructure**: Docker
**ML Pipeline**: PyTorch training, automated model deployment


## Project Structure

```
SecurePayQR/
├── src/                          # Python source code
│   ├── fastapi_backend.py        # API backend
│   ├── cnn_lstm_model.py         # ML model architecture
│   ├── training_pipeline.py     # Model training
│   └── dataset_creation_script.py # Dataset generation
├── frontend/                     # React frontend
│   ├── src/components/           # React components
│   └── src/context/              # State management
├── config/                       # Configuration files
├── tests/                        # Test suite
├── monitoring/                   # Prometheus & Grafana configs
├── docker-compose.yml            # Docker services
├── requirements.txt              # Python dependencies
└── .env                          # Environment variables
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (for local development)
- Node.js 16+ (for frontend development)

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/Krishna-Kumar-Rathore/SecurePayQR.git
cd SecurePayQR
```

2. **Environment Setup**
```bash
cp .env.example .env
# Edit .env with your configurations
```

3. **Docker Deployment**
```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Production
docker-compose up -d --build
```

4. **Local Development (Alternative)**
```bash
# Backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/fastapi_backend.py

# Frontend
cd frontend
npm install
npm start
```

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Web App | http://localhost:3000 | Main application |
| API | http://localhost:8000 | Backend API |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| Grafana | http://localhost:3000 | Monitoring dashboard |

## Usage

### Web Interface
1. Open the web application
2. Choose detection method (Camera or Upload)
3. Scan/upload QR code image
4. View fraud detection results with confidence scores

### API Usage

**Health Check**
```bash
curl http://localhost:8000/health
```

**QR Detection**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Authorization: Bearer demo-token" \
  -F "file=@qr_code.png"
```

**Response**
```json
{
  "is_tampered": false,
  "confidence": 0.967,
  "probabilities": {
    "valid": 0.967,
    "tampered": 0.033
  },
  "processing_time_ms": 284.5,
  "model_version": "1.0"
}
```





## Model Architecture

The CNN-LSTM hybrid model combines:
- **CNN (MobileNetV3)**: Spatial feature extraction from QR code images
- **LSTM**: Sequential pattern analysis for tampering detection
- **Feature Fusion**: Multi-layer network combining CNN and LSTM outputs

**Performance Metrics**:
- Accuracy: 96.2%
- Precision: 97.8%
- Recall: 95.4%
- Inference Time: <300ms

## Development

### Training New Models
```bash
python src/training_pipeline.py --config config/train_config.json --use_wandb
```

### Running Tests
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### API Development
The FastAPI backend provides endpoints for:
- QR code fraud detection
- Batch processing
- Model information
- Health monitoring
- Prometheus metrics

## Docker Services

- **API**: FastAPI backend with ML model
- **Database**: PostgreSQL for data storage
- **Cache**: Redis for performance optimization
- **Proxy**: Nginx reverse proxy
- **Monitoring**: Prometheus + Grafana stack

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/Krishna-Kumar-Rathore/SecurePayQR/issues)
- **Documentation**: [API Docs](http://localhost:8000/docs)

---

Built for secure digital payment verification using advanced AI techniques.