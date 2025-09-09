#!/usr/bin/env python3
"""
SecurePayQR: Training Pipeline
Complete training script with validation, metrics tracking, and model export
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import wandb  # For experiment tracking
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings

# Import our model
# from cnn_lstm_model import SecurePayQRModel, export_to_onnx
from .cnn_lstm_model import SecurePayQRModel, export_to_onnx

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QRCodeDataset(Dataset):
    """Dataset class for QR code images"""
    
    def __init__(self, dataset_dir: str, split: str = 'train', transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.split = split
        
        # Load metadata
        metadata_file = self.dataset_dir / "metadata" / "dataset_metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter by split if provided
        self.samples = self.metadata
        
        # Create label mapping
        self.label_to_idx = {'valid': 0, 'tampered': 1}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        if sample['label'] == 'valid':
            image_path = self.dataset_dir / "valid" / sample['filename']
        else:
            image_path = self.dataset_dir / "tampered" / sample['filename']
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_to_idx[sample['label']]
        
        return {
            'image': image,
            'label': label,
            'filename': sample['filename'],
            'tampering_method': sample.get('tampering_method', 'none')
        }

class QRTrainer:
    """Trainer class for SecurePayQR model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = SecurePayQRModel(
            input_channels=config['model']['input_channels'],
            cnn_feature_dim=config['model']['cnn_feature_dim'],
            lstm_hidden_dim=config['model']['lstm_hidden_dim'],
            lstm_layers=config['model']['lstm_layers'],
            num_classes=config['model']['num_classes'],
            dropout=config['model']['dropout']
        ).to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function with class weights for imbalanced data
        if config['training']['use_class_weights']:
            class_weights = torch.FloatTensor([1.0, 2.0]).to(self.device)  # Higher weight for tampered
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Initialize wandb if enabled
        if config['logging']['use_wandb']:
            wandb.init(
                project="securepayqr",
                config=config,
                name=config['experiment_name']
            )
            wandb.watch(self.model)
    
    def create_data_loaders(self):
        """Create train and validation data loaders"""
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create full dataset
        full_dataset = QRCodeDataset(
            self.config['data']['dataset_dir'],
            transform=train_transform
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate model and return metrics"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # ROC AUC for binary classification
        roc_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'precision_valid': precision_per_class[0],
            'recall_valid': recall_per_class[0],
            'f1_valid': f1_per_class[0],
            'precision_tampered': precision_per_class[1],
            'recall_tampered': recall_per_class[1],
            'f1_tampered': f1_per_class[1]
        }
        
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Create data loaders
        self.create_data_loaders()
        
        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Print metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['roc_auc']:.4f}")
            
            # Log to wandb
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': self.config
                }
                
                torch.save(checkpoint, self.output_dir / 'best_model.pth')
                logger.info(f"Saved best model with val_loss: {val_metrics['loss']:.4f}")
            
            # Early stopping
            if epoch > 10 and self.early_stopping_check():
                logger.info("Early stopping triggered")
                break
        
        # Load best model and save final version
        self.model.load_state_dict(self.best_model_state)
        self.save_final_model()
        
        # Generate training plots
        self.plot_training_history()
        
        # Final evaluation
        final_metrics = self.validate()
        logger.info(f"\nFinal Validation Metrics:")
        for key, value in final_metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        return final_metrics
    
    def early_stopping_check(self, patience: int = 10) -> bool:
        """Check if early stopping should be triggered"""
        if len(self.val_losses) < patience:
            return False
        
        recent_losses = self.val_losses[-patience:]
        return all(loss >= self.best_val_loss for loss in recent_losses)
    
    def save_final_model(self):
        """Save final model in multiple formats"""
        
        # Save PyTorch model
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        torch.save(final_checkpoint, self.output_dir / 'securepayqr_final.pth')
        
        # Export to ONNX for web deployment
        try:
            onnx_path = self.output_dir / 'securepayqr_model.onnx'
            export_to_onnx(self.model, str(onnx_path))
            logger.info(f"Model exported to ONNX: {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
        
        # Save model info
        model_info = {
            'model_size_mb': sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'input_shape': [1, 3, 256, 256],
            'output_shape': [1, 2],
            'class_names': ['valid', 'tampered']
        }
        
        with open(self.output_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, '_last_lr'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            ax3.plot(epochs[:len(lrs)], lrs, 'orange', label='Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.legend()
            ax3.grid(True)
        
        # Model architecture summary
        ax4.text(0.1, 0.8, f"Model: SecurePayQR CNN-LSTM", fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.7, f"Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}", fontsize=10)
        ax4.text(0.1, 0.6, f"Best Val Loss: {self.best_val_loss:.4f}", fontsize=10)
        ax4.text(0.1, 0.5, f"Final Val Acc: {self.val_accuracies[-1]:.4f}", fontsize=10)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Model Summary')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.config['logging']['use_wandb']:
            wandb.log({"training_history": wandb.Image(str(self.output_dir / 'training_history.png'))})

def load_config(config_path: str = None) -> Dict:
    """Load training configuration"""
    
    default_config = {
        'experiment_name': 'securepayqr_v1',
        'output_dir': 'outputs',
        'data': {
            'dataset_dir': 'qr_dataset'
        },
        'model': {
            'input_channels': 3,
            'cnn_feature_dim': 512,
            'lstm_hidden_dim': 256,
            'lstm_layers': 2,
            'num_classes': 2,
            'dropout': 0.3
        },
        'training': {
            'num_epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'num_workers': 4,
            'use_class_weights': True
        },
        'logging': {
            'use_wandb': False
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Deep merge configs
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(default_config, user_config)
    
    return default_config

def main():
    parser = argparse.ArgumentParser(description="Train SecurePayQR model")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--dataset_dir", type=str, default="qr_dataset", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset_dir:
        config['data']['dataset_dir'] = args.dataset_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.use_wandb:
        config['logging']['use_wandb'] = True
    
    # Create trainer and start training
    trainer = QRTrainer(config)
    final_metrics = trainer.train()
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {config['output_dir']}/best_model.pth")
    print(f"ONNX model saved to: {config['output_dir']}/securepayqr_model.onnx")

if __name__ == "__main__":
    main()