#!/usr/bin/env python3
"""
SecurePayQR: Comprehensive Testing Framework
Unit tests, integration tests, and performance tests for the entire system
"""

import pytest
import asyncio
import os
import io
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import tempfile
import requests

# Import modules to test
import sys
sys.path.append('src')

from dataset_creation_script import QRDatasetGenerator
from cnn_lstm_model import SecurePayQRModel, QRTamperDetector, create_securepayqr_model
from training_pipeline import QRCodeDataset, QRTrainer
from fastapi_backend import app, model_manager

# Test Configuration
TEST_CONFIG = {
    'test_data_dir': 'tests/test_data',
    'temp_model_path': 'tests/temp_model.onnx',
    'sample_qr_codes': ['upi://pay?pa=test@paytm&pn=Test&am=100']
}

# Fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_qr_image():
    """Create a sample QR code image for testing"""
    # Create a simple test image
    image = Image.new('RGB', (256, 256), color='white')
    # Add some QR-like patterns
    pixels = image.load()
    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            if (i + j) % 16 == 0:
                for x in range(i, min(i+4, 256)):
                    for y in range(j, min(j+4, 256)):
                        pixels[x, y] = (0, 0, 0)
    return image

@pytest.fixture
def test_client():
    """Create FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def mock_model():
    """Create mock ML model for testing"""
    model = Mock()
    model.predict.return_value = torch.tensor([0])  # Valid QR code
    model.predict_proba.return_value = torch.tensor([[0.9, 0.1]])
    return model

# Dataset Tests
class TestQRDatasetGenerator:
    """Test QR dataset generation functionality"""
    
    def test_generator_initialization(self, temp_dir):
        """Test dataset generator initialization"""
        generator = QRDatasetGenerator(temp_dir)
        
        assert generator.output_dir.exists()
        assert (generator.output_dir / "valid").exists()
        assert (generator.output_dir / "tampered").exists()
        assert (generator.output_dir / "metadata").exists()
    
    def test_valid_qr_generation(self, temp_dir):
        """Test valid QR code generation"""
        generator = QRDatasetGenerator(temp_dir)
        
        # Generate small number for testing
        valid_codes = generator.generate_valid_qr_codes(num_codes=10)
        
        assert len(valid_codes) == 10
        assert all(code['label'] == 'valid' for code in valid_codes)
        assert all(os.path.exists(generator.output_dir / "valid" / code['filename']) 
                  for code in valid_codes)
    
    def test_tampering_methods(self, temp_dir):
        """Test different tampering methods"""
        generator = QRDatasetGenerator(temp_dir)
        
        # Create a test image
        test_image = Image.new('RGB', (256, 256), color='white')
        
        # Test each tampering method
        for method in generator.tampering_methods:
            tampered = generator._apply_tampering(test_image, method)
            assert isinstance(tampered, Image.Image)
            assert tampered.size == test_image.size
    
    def test_dataset_metadata_creation(self, temp_dir):
        """Test metadata file creation"""
        generator = QRDatasetGenerator(temp_dir)
        
        valid_codes = generator.generate_valid_qr_codes(num_codes=5)
        tampered_codes = generator.generate_tampered_qr_codes(valid_codes, tampering_ratio=0.5)
        
        generator.save_dataset_metadata(valid_codes, tampered_codes)
        
        # Check metadata files exist
        assert (generator.output_dir / "metadata" / "dataset_metadata.json").exists()
        assert (generator.output_dir / "metadata" / "dataset_stats.json").exists()
        
        # Validate metadata content
        with open(generator.output_dir / "metadata" / "dataset_stats.json") as f:
            stats = json.load(f)
        
        assert stats['valid_samples'] == 5
        assert stats['tampered_samples'] >= 2  # Should be around 2-3 for 50% ratio

# Model Tests
class TestSecurePayQRModel:
    """Test CNN-LSTM model functionality"""
    
    def test_model_creation(self):
        """Test model instantiation"""
        model = create_securepayqr_model(
            input_channels=3,
            cnn_feature_dim=256,
            lstm_hidden_dim=128,
            num_classes=2
        )
        
        assert isinstance(model, SecurePayQRModel)
        assert model.num_classes == 2
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = create_securepayqr_model(
            cnn_feature_dim=256,
            lstm_hidden_dim=128
        )
        
        # Test input
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 256, 256)
        
        logits, features = model(test_input)
        
        assert logits.shape == (batch_size, 2)
        assert 'cnn_features' in features
        assert 'lstm_features' in features
        assert 'fused_features' in features
    
    def test_model_prediction(self):
        """Test model prediction functionality"""
        model = create_securepayqr_model()
        model.eval()
        
        test_input = torch.randn(1, 3, 256, 256)
        
        # Test prediction methods
        prediction = model.predict(test_input)
        probabilities = model.predict_proba(test_input)
        
        assert prediction.shape == (1,)
        assert probabilities.shape == (1, 2)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    
    def test_sequence_extractor(self):
        """Test QR sequence extraction"""
        from cnn_lstm_model import QRSequenceExtractor
        
        extractor = QRSequenceExtractor(patch_size=8, stride=4)
        test_input = torch.randn(2, 3, 256, 256)
        
        sequence = extractor.extract_scanning_sequence(test_input)
        
        assert len(sequence.shape) == 3  # (batch, seq_len, features)
        assert sequence.shape[0] == 2  # batch size
        assert sequence.shape[2] == 3 * 8 * 8  # channels * patch_size^2

# Training Tests
class TestTrainingPipeline:
    """Test training pipeline functionality"""
    
    def test_dataset_loading(self, temp_dir):
        """Test QR dataset loading"""
        # Create minimal test dataset
        os.makedirs(f"{temp_dir}/valid", exist_ok=True)
        os.makedirs(f"{temp_dir}/tampered", exist_ok=True)
        os.makedirs(f"{temp_dir}/metadata", exist_ok=True)
        
        # Create test metadata
        metadata = [
            {'filename': 'test1.png', 'label': 'valid', 'upi_string': 'test'},
            {'filename': 'test2.png', 'label': 'tampered', 'upi_string': 'test', 'tampering_method': 'overlay'}
        ]
        
        with open(f"{temp_dir}/metadata/dataset_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create test images
        test_img = Image.new('RGB', (256, 256), color='white')
        test_img.save(f"{temp_dir}/valid/test1.png")
        test_img.save(f"{temp_dir}/tampered/test2.png")
        
        # Test dataset loading
        dataset = QRCodeDataset(temp_dir)
        
        assert len(dataset) == 2
        
        sample = dataset[0]
        assert 'image' in sample
        assert 'label' in sample
        assert sample['label'] in [0, 1]
    
    def test_training_config_loading(self):
        """Test training configuration loading"""
        from training_pipeline import load_config
        
        config = load_config()
        
        assert 'model' in config
        assert 'training' in config
        assert 'data' in config
        assert config['model']['num_classes'] == 2
    
    @patch('training_pipeline.wandb')
    def test_trainer_initialization(self, mock_wandb, temp_dir):
        """Test trainer initialization"""
        config = {
            'output_dir': temp_dir,
            'model': {
                'input_channels': 3,
                'cnn_feature_dim': 256,
                'lstm_hidden_dim': 128,
                'lstm_layers': 2,
                'num_classes': 2,
                'dropout': 0.3
            },
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'use_class_weights': False
            },
            'logging': {
                'use_wandb': False
            }
        }
        
        trainer = QRTrainer(config)
        
        assert trainer.config == config
        assert isinstance(trainer.model, SecurePayQRModel)
        assert trainer.optimizer is not None

# API Tests
class TestFastAPIBackend:
    """Test FastAPI backend functionality"""
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'version' in data
    
    def test_model_info_endpoint(self, test_client):
        """Test model info endpoint"""
        response = test_client.get("/model/info")
        
        # May return 503 if model not loaded in test environment
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'model_name' in data
            assert 'architecture' in data
    
    def test_detection_endpoint_with_file(self, test_client, sample_qr_image):
        """Test QR detection endpoint with file upload"""
        # Convert image to bytes
        img_bytes = io.BytesIO()
        sample_qr_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"file": ("test.png", img_bytes, "image/png")}
        
        response = test_client.post("/detect", files=files)
        
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'is_tampered' in data
            assert 'confidence' in data
            assert 'probabilities' in data
    
    def test_detection_endpoint_invalid_file(self, test_client):
        """Test detection endpoint with invalid file"""
        files = {"file": ("test.txt", io.StringIO("not an image"), "text/plain")}
        
        response = test_client.post("/detect", files=files)
        assert response.status_code == 400
    
    def test_batch_detection_endpoint(self, test_client, sample_qr_image):
        """Test batch detection endpoint"""
        # Create multiple test images
        img_bytes1 = io.BytesIO()
        img_bytes2 = io.BytesIO()
        sample_qr_image.save(img_bytes1, format='PNG')
        sample_qr_image.save(img_bytes2, format='PNG')
        img_bytes1.seek(0)
        img_bytes2.seek(0)
        
        files = [
            ("files", ("test1.png", img_bytes1, "image/png")),
            ("files", ("test2.png", img_bytes2, "image/png"))
        ]
        
        response = test_client.post("/detect/batch", files=files)
        
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'results' in data
            assert len(data['results']) == 2
    
    def test_metrics_endpoint(self, test_client):
        """Test Prometheus metrics endpoint"""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_stats_endpoint(self, test_client):
        """Test statistics endpoint"""
        response = test_client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        expected_keys = ['total_requests', 'total_detections', 'tampered_detected', 
                        'average_processing_time_ms', 'model_accuracy', 'uptime_hours']
        
        for key in expected_keys:
            assert key in data

# Performance Tests
class TestPerformance:
    """Test performance and load handling"""
    
    def test_model_inference_speed(self, mock_model):
        """Test model inference speed"""
        detector = QRTamperDetector()
        detector.model = mock_model
        detector.model.eval = Mock()
        
        # Create test image
        test_image = Image.new('RGB', (256, 256), color='white')
        
        # Measure inference time
        start_time = time.time()
        result = detector.detect_tampering(test_image)
        inference_time = time.time() - start_time
        
        # Should be fast with mock model
        assert inference_time < 1.0  # Less than 1 second
    
    def test_concurrent_requests(self, test_client, sample_qr_image):
        """Test concurrent API requests"""
        import concurrent.futures
        
        def make_request():
            img_bytes = io.BytesIO()
            sample_qr_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            files = {"file": ("test.png", img_bytes, "image/png")}
            return test_client.post("/detect", files=files)
        
        # Test 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in futures]
        
        # All requests should complete (may be 503 if model not loaded)
        for response in responses:
            assert response.status_code in [200, 503]
    
    def test_memory_usage(self):
        """Test memory usage during model operations"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use model
        model = create_securepayqr_model()
        test_input = torch.randn(4, 3, 256, 256)  # Larger batch
        
        with torch.no_grad():
            for _ in range(10):
                logits, _ = model(test_input)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del model, test_input, logits
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should not grow excessively
        memory_growth = peak_memory - initial_memory
        assert memory_growth < 2000  # Less than 2GB growth
        
        # Memory should be mostly freed
        memory_retained = final_memory - initial_memory
        assert memory_retained < 500  # Less than 500MB retained

# Integration Tests
class TestIntegration:
    """Test end-to-end integration"""
    
    def test_full_pipeline_integration(self, temp_dir):
        """Test complete pipeline from dataset to model to API"""
        # 1. Generate small test dataset
        generator = QRDatasetGenerator(temp_dir)
        valid_codes = generator.generate_valid_qr_codes(num_codes=5)
        tampered_codes = generator.generate_tampered_qr_codes(valid_codes, 0.5)
        generator.save_dataset_metadata(valid_codes, tampered_codes)
        
        # 2. Test dataset loading
        dataset = QRCodeDataset(temp_dir)
        assert len(dataset) > 0
        
        # 3. Test model creation and inference
        model = create_securepayqr_model()
        model.eval()
        
        sample = dataset[0]
        test_input = sample['image'].unsqueeze(0)
        
        with torch.no_grad():
            logits, features = model(test_input)
            prediction = model.predict(test_input)
        
        assert logits.shape == (1, 2)
        assert prediction.shape == (1,)
    
    @pytest.mark.asyncio
    async def test_api_model_integration(self, sample_qr_image):
        """Test API and model integration"""
        # Mock the model manager for testing
        with patch('fastapi_backend.model_manager') as mock_manager:
            mock_manager.is_loaded = True
            mock_manager.detect_tampering.return_value = {
                'is_tampered': False,
                'confidence': 0.95,
                'probabilities': {'valid': 0.95, 'tampered': 0.05},
                'processing_time_ms': 250.0,
                'model_version': '1.0',
                'timestamp': '2024-01-01 00:00:00'
            }
            
            # Test API call
            test_client = TestClient(app)
            
            img_bytes = io.BytesIO()
            sample_qr_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            files = {"file": ("test.png", img_bytes, "image/png")}
            response = test_client.post("/detect", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data['is_tampered'] == False
            assert data['confidence'] == 0.95

# Test Utilities
class TestUtilities:
    """Test utility functions and helpers"""
    
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        from fastapi_backend import SecurePayQRModelManager
        
        manager = SecurePayQRModelManager()
        test_image = Image.new('RGB', (512, 512), color='white')
        
        processed = manager.preprocess_image(test_image)
        
        assert processed.shape == (1, 3, 256, 256)
        assert processed.dtype == np.float32
        assert processed.min() >= -5.0  # Reasonable normalization range
        assert processed.max() <= 5.0
    
    def test_softmax_implementation(self):
        """Test softmax function"""
        from fastapi_backend import SecurePayQRModelManager
        
        test_logits = np.array([2.0, 1.0, 0.1])
        probabilities = SecurePayQRModelManager._softmax(test_logits)
        
        assert len(probabilities) == 3
        assert np.abs(probabilities.sum() - 1.0) < 1e-6
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

# Test Configuration and Fixtures
@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment"""
    # Create test directories
    os.makedirs('tests/test_data', exist_ok=True)
    os.makedirs('tests/models', exist_ok=True)
    
    yield
    
    # Cleanup is handled by temp directories

if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])