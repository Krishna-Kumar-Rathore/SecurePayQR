#!/usr/bin/env python3
"""
SecurePayQR: CNN-LSTM Model Architecture
Combines CNN spatial features with LSTM sequential analysis for QR fraud detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class QRSequenceExtractor:
    """Extract sequential patterns from QR codes for LSTM processing"""
    
    def __init__(self, patch_size: int = 8, stride: int = 4):
        self.patch_size = patch_size
        self.stride = stride
    
    def extract_scanning_sequence(self, qr_image: torch.Tensor) -> torch.Tensor:
        """
        Extract QR code in scanning order (zigzag pattern similar to QR reading)
        Args:
            qr_image: (B, C, H, W) tensor
        Returns:
            sequence: (B, seq_len, patch_features) tensor
        """
        B, C, H, W = qr_image.shape
        
        # Extract patches using unfold
        patches = qr_image.unfold(2, self.patch_size, self.stride)\
                         .unfold(3, self.patch_size, self.stride)  # (B, C, H_patches, W_patches, patch_size, patch_size)
        
        B, C, H_p, W_p, P1, P2 = patches.shape
        
        # Flatten patches for sequential processing
        patches = patches.reshape(B, C, H_p * W_p, P1 * P2)  # (B, C, num_patches, patch_features)
        
        # Create zigzag scanning order (mimicking QR reader pattern)
        sequence_indices = self._create_zigzag_indices(H_p, W_p)
        
        # Reorder patches according to scanning pattern
        patches = patches[:, :, sequence_indices, :]  # (B, C, num_patches, patch_features)
        
        # Combine channel and feature dimensions
        patches = patches.reshape(B, H_p * W_p, C * P1 * P2)  # (B, seq_len, features)
        
        return patches
    
    def _create_zigzag_indices(self, H: int, W: int) -> torch.Tensor:
        """Create zigzag scanning pattern indices"""
        indices = []
        
        # Create zigzag pattern (left-to-right, then right-to-left alternating)
        for i in range(H):
            if i % 2 == 0:  # Even rows: left to right
                for j in range(W):
                    indices.append(i * W + j)
            else:  # Odd rows: right to left
                for j in range(W-1, -1, -1):
                    indices.append(i * W + j)
        
        return torch.tensor(indices, dtype=torch.long)

class CNNFeatureExtractor(nn.Module):
    """CNN backbone for spatial feature extraction"""
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        
        # Use MobileNetV3-Small as backbone (efficient for web deployment)
        self.backbone = mobilenet_v3_small(pretrained=True)
        
        # Modify first conv for grayscale input if needed
        if input_channels == 1:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
        
        # Remove classifier and get feature dimension
        self.backbone.classifier = nn.Identity()
        
        # Add custom feature head
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # Fixed spatial size
            nn.Flatten(),
            nn.Linear(576 * 7 * 7, feature_dim),  # MobileNetV3-Small outputs 576 channels
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image
        Returns:
            features: (B, feature_dim) spatial features
        """
        features = self.backbone.features(x)
        features = self.feature_head(features)
        return features

class LSTMSequenceAnalyzer(nn.Module):
    """LSTM for analyzing QR code sequential patterns"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, 
                 dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM for better pattern recognition
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism for focusing on important sequence parts
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, input_dim) sequence features
        Returns:
            output: (B, output_dim) sequence representation
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)  # (B, seq_len, hidden_dim * 2)
        
        # Attention-weighted pooling
        attention_weights = self.attention(lstm_out)  # (B, seq_len, 1)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # (B, hidden_dim * 2)
        
        return attended_output

class SecurePayQRModel(nn.Module):
    """
    Complete CNN-LSTM model for QR code fraud detection
    Combines spatial CNN features with sequential LSTM analysis
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 cnn_feature_dim: int = 512,
                 lstm_hidden_dim: int = 256,
                 lstm_layers: int = 2,
                 patch_size: int = 8,
                 stride: int = 4,
                 num_classes: int = 2,  # valid vs tampered
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Sequential pattern extractor
        self.sequence_extractor = QRSequenceExtractor(patch_size, stride)
        
        # CNN for spatial features
        self.cnn_extractor = CNNFeatureExtractor(input_channels, cnn_feature_dim)
        
        # Calculate sequence feature dimension
        sequence_feature_dim = input_channels * patch_size * patch_size
        
        # LSTM for sequential analysis
        self.lstm_analyzer = LSTMSequenceAnalyzer(
            sequence_feature_dim, lstm_hidden_dim, lstm_layers, dropout
        )
        
        # Fusion layer combining CNN and LSTM features
        total_feature_dim = cnn_feature_dim + self.lstm_analyzer.output_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with detailed outputs for analysis
        
        Args:
            x: (B, C, H, W) input QR code images
            
        Returns:
            logits: (B, num_classes) classification logits
            features: dict with intermediate features for analysis
        """
        batch_size = x.size(0)
        
        # Extract spatial features using CNN
        cnn_features = self.cnn_extractor(x)  # (B, cnn_feature_dim)
        
        # Extract sequential features
        sequence_features = self.sequence_extractor.extract_scanning_sequence(x)  # (B, seq_len, patch_features)
        
        # Analyze sequences with LSTM
        lstm_features = self.lstm_analyzer(sequence_features)  # (B, lstm_output_dim)
        
        # Fuse CNN and LSTM features
        combined_features = torch.cat([cnn_features, lstm_features], dim=1)  # (B, total_feature_dim)
        fused_features = self.fusion_layer(combined_features)  # (B, 256)
        
        # Classification
        logits = self.classifier(fused_features)  # (B, num_classes)
        
        # Collect features for analysis/visualization
        features = {
            'cnn_features': cnn_features,
            'lstm_features': lstm_features,
            'fused_features': fused_features,
            'sequence_features': sequence_features
        }
        
        return logits, features
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities"""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)

class QRTamperDetector:
    """High-level interface for QR tamper detection"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = SecurePayQRModel(
            input_channels=3,
            cnn_feature_dim=512,
            lstm_hidden_dim=256,
            num_classes=2
        ).to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
    
    def detect_tampering(self, image: np.ndarray, return_confidence: bool = True) -> dict:
        """
        Detect if QR code is tampered
        
        Args:
            image: RGB image array (H, W, 3)
            return_confidence: whether to return confidence scores
            
        Returns:
            result: dict with prediction and confidence
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, features = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        result = {
            'is_tampered': bool(prediction.item()),
            'class_names': ['valid', 'tampered'],
            'predicted_class': 'tampered' if prediction.item() else 'valid'
        }
        
        if return_confidence:
            result.update({
                'confidence': float(probabilities.max().item()),
                'probabilities': {
                    'valid': float(probabilities[0, 0].item()),
                    'tampered': float(probabilities[0, 1].item())
                }
            })
        
        return result
    
    def batch_detect(self, images: list) -> list:
        """Detect tampering for batch of images"""
        results = []
        for image in images:
            result = self.detect_tampering(image)
            results.append(result)
        return results

# Model factory function
def create_securepayqr_model(**kwargs) -> SecurePayQRModel:
    """Factory function to create SecurePayQR model with custom parameters"""
    default_params = {
        'input_channels': 3,
        'cnn_feature_dim': 512,
        'lstm_hidden_dim': 256,
        'lstm_layers': 2,
        'patch_size': 8,
        'stride': 4,
        'num_classes': 2,
        'dropout': 0.3
    }
    default_params.update(kwargs)
    
    return SecurePayQRModel(**default_params)

# Export for ONNX conversion
def export_to_onnx(model: SecurePayQRModel, output_path: str, input_shape: Tuple[int, ...] = (1, 3, 256, 256)):
    """Export trained model to ONNX format for web deployment"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['qr_image'],
        output_names=['logits'],
        dynamic_axes={
            'qr_image': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to ONNX: {output_path}")

if __name__ == "__main__":
    # Test model creation
    model = create_securepayqr_model()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    logits, features = model(dummy_input)
    
    print(f"Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"CNN features shape: {features['cnn_features'].shape}")
    print(f"LSTM features shape: {features['lstm_features'].shape}")
    print(f"Fused features shape: {features['fused_features'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (approx): {total_params * 4 / 1024 / 1024:.2f} MB")