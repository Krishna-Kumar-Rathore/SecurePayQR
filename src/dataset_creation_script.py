#!/usr/bin/env python3
"""
SecurePayQR: Dataset Creation Pipeline
Creates synthetic tampered QR codes for training CNN-LSTM model
"""

import os
import cv2
import numpy as np
import qrcode
import random
import json
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Any
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QRDatasetGenerator:
    """Generate valid and tampered QR codes for training"""
    
    def __init__(self, output_dir: str = "qr_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "valid").mkdir(exist_ok=True)
        (self.output_dir / "tampered").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # QR code parameters
        self.qr_sizes = [256, 512, 1024]  # Various QR sizes
        self.error_corrections = [
            qrcode.constants.ERROR_CORRECT_L,  # ~7%
            qrcode.constants.ERROR_CORRECT_M,  # ~15%
            qrcode.constants.ERROR_CORRECT_Q,  # ~25%
            qrcode.constants.ERROR_CORRECT_H   # ~30%
        ]
        
        # Tampering parameters
        self.tampering_methods = [
            'digital_overlay',
            'module_manipulation',
            'print_scan_simulation',
            'environmental_attack',
            'partial_occlusion',
            'gradient_overlay',
            'logo_insertion'
        ]
        
    def generate_valid_qr_codes(self, num_codes: int = 1000) -> List[Dict]:
        """Generate valid QR codes with various content"""
        
        # UPI payment templates
        upi_templates = [
            "upi://pay?pa={}@{}&pn={}&am={}&cu=INR",
            "upi://pay?pa={}@{}&pn={}",
            "upi://pay?pa={}@{}&pn={}&tr={}",
            "upi://pay?pa={}@{}&pn={}&am={}&tr={}&cu=INR"
        ]
        
        # Sample data for UPI generation
        vpa_names = ["deepak", "rajesh", "priya", "amit", "sneha", "vijay", "anita"]
        bank_codes = ["paytm", "ybl", "okhdfcbank", "axl", "ibl", "apl"]
        merchant_names = ["Tea Center", "Grocery Store", "Restaurant", "Medical Store", 
                         "Electronics Shop", "Clothing Store", "Service Center"]
        amounts = ["100", "250", "500", "750", "1000", "1500", "2000"]
        
        valid_codes = []
        
        logger.info(f"Generating {num_codes} valid QR codes...")
        
        for i in tqdm(range(num_codes)):
            # Generate UPI string
            vpa = f"{random.choice(vpa_names)}{random.randint(1000, 9999)}"
            bank = random.choice(bank_codes)
            merchant = random.choice(merchant_names)
            
            if random.random() < 0.7:  # 70% with amount
                amount = random.choice(amounts)
                transaction_id = f"TXN{random.randint(100000, 999999)}"
                upi_string = random.choice(upi_templates).format(
                    vpa, bank, merchant, amount, transaction_id
                )
            else:
                upi_string = random.choice(upi_templates[:2]).format(vpa, bank, merchant)
            
            # Generate QR code
            qr_size = random.choice(self.qr_sizes)
            error_correction = random.choice(self.error_corrections)
            
            qr = qrcode.QRCode(
                version=None,  # Auto-determine version
                error_correction=error_correction,
                box_size=max(1, qr_size // 37),  # Adjust box size for target resolution
                border=4,
            )
            qr.add_data(upi_string)
            qr.make(fit=True)
            
            # Create image
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_img = qr_img.resize((qr_size, qr_size), Image.Resampling.LANCZOS)
            
            # Add some natural variations
            qr_img = self._add_natural_variations(qr_img)
            
            # Save image
            filename = f"valid_qr_{i:05d}.png"
            filepath = self.output_dir / "valid" / filename
            qr_img.save(filepath)
            
            # Store metadata
            metadata = {
                'filename': filename,
                'upi_string': upi_string,
                'size': qr_size,
                'error_correction': error_correction,
                'label': 'valid',
                'tampering_method': None
            }
            valid_codes.append(metadata)
            
        return valid_codes
    
    def _add_natural_variations(self, img: Image.Image) -> Image.Image:
        """Add natural variations to QR codes"""
        
        # Random brightness/contrast
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
            
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Slight blur (simulating camera focus)
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
            
        # JPEG compression artifacts
        if random.random() < 0.4:
            # Convert to JPEG and back to simulate compression
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=random.randint(85, 98))
            buffer.seek(0)
            img = Image.open(buffer)
            
        return img
    
    def generate_tampered_qr_codes(self, valid_codes: List[Dict], 
                                   tampering_ratio: float = 1.0) -> List[Dict]:
        """Generate tampered versions of valid QR codes"""
        
        num_tampered = int(len(valid_codes) * tampering_ratio)
        tampered_codes = []
        
        logger.info(f"Generating {num_tampered} tampered QR codes...")
        
        for i in tqdm(range(num_tampered)):
            # Select random valid QR code
            source_code = random.choice(valid_codes)
            source_path = self.output_dir / "valid" / source_code['filename']
            
            # Load source image
            source_img = Image.open(source_path)
            
            # Apply random tampering method
            tampering_method = random.choice(self.tampering_methods)
            tampered_img = self._apply_tampering(source_img, tampering_method)
            
            # Save tampered image
            filename = f"tampered_qr_{i:05d}_{tampering_method}.png"
            filepath = self.output_dir / "tampered" / filename
            tampered_img.save(filepath)
            
            # Store metadata
            metadata = {
                'filename': filename,
                'source_filename': source_code['filename'],
                'upi_string': source_code['upi_string'],
                'size': source_code['size'],
                'label': 'tampered',
                'tampering_method': tampering_method
            }
            tampered_codes.append(metadata)
            
        return tampered_codes
    
    def _apply_tampering(self, img: Image.Image, method: str) -> Image.Image:
        """Apply specific tampering method"""
        
        if method == 'digital_overlay':
            return self._digital_overlay(img)
        elif method == 'module_manipulation':
            return self._module_manipulation(img)
        elif method == 'print_scan_simulation':
            return self._print_scan_simulation(img)
        elif method == 'environmental_attack':
            return self._environmental_attack(img)
        elif method == 'partial_occlusion':
            return self._partial_occlusion(img)
        elif method == 'gradient_overlay':
            return self._gradient_overlay(img)
        elif method == 'logo_insertion':
            return self._logo_insertion(img)
        else:
            return img
    
    def _digital_overlay(self, img: Image.Image) -> Image.Image:
        """Simulate sticker/overlay attacks"""
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Random overlay shapes
        overlay_type = random.choice(['rectangle', 'circle', 'text'])
        
        if overlay_type == 'rectangle':
            x1, y1 = random.randint(0, w//3), random.randint(0, h//3)
            x2, y2 = x1 + random.randint(w//8, w//4), y1 + random.randint(h//8, h//4)
            color = random.choice(['white', 'black', 'red', 'blue'])
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
        elif overlay_type == 'circle':
            x, y = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
            radius = random.randint(w//16, w//8)
            color = random.choice(['white', 'black', 'red', 'blue'])
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
            
        elif overlay_type == 'text':
            try:
                from PIL import ImageFont
                # Use default font if available
                font_size = random.randint(12, 24)
                text = random.choice(['PAID', 'FAKE', 'SCAN ME', '★', '€', '$'])
                x, y = random.randint(0, w//2), random.randint(0, h//2)
                draw.text((x, y), text, fill='red')
            except:
                # Fallback to simple overlay if font not available
                x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
                x2, y2 = x1 + w//8, y1 + h//16
                draw.rectangle([x1, y1, x2, y2], fill='red')
        
        return img
    
    def _module_manipulation(self, img: Image.Image) -> Image.Image:
        """Manipulate individual QR modules"""
        img_array = np.array(img.convert('L'))
        h, w = img_array.shape
        
        # Estimate module size
        module_size = max(1, min(w, h) // 37)  # Standard QR has ~37 modules per side
        
        # Randomly flip some modules
        num_flips = random.randint(5, 20)
        for _ in range(num_flips):
            x = random.randint(0, w - module_size)
            y = random.randint(0, h - module_size)
            
            # Flip the module
            region = img_array[y:y+module_size, x:x+module_size]
            img_array[y:y+module_size, x:x+module_size] = 255 - region
        
        return Image.fromarray(img_array).convert('RGB')
    
    def _print_scan_simulation(self, img: Image.Image) -> Image.Image:
        """Simulate print-scan degradation"""
        # Add noise
        img_array = np.array(img)
        noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        
        # Add blur
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # JPEG compression
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=random.randint(60, 85))
        buffer.seek(0)
        img = Image.open(buffer)
        
        return img
    
    def _environmental_attack(self, img: Image.Image) -> Image.Image:
        """Simulate environmental conditions"""
        # Lighting changes
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.3, 1.8))
        
        # Add perspective distortion (simple skew)
        w, h = img.size
        skew_x = random.randint(-w//20, w//20)
        skew_y = random.randint(-h//20, h//20)
        
        # Create transform matrix for perspective
        transform = (1, skew_x/w, 0, skew_y/h, 1, 0)
        img = img.transform(img.size, Image.Transform.AFFINE, transform)
        
        return img
    
    def _partial_occlusion(self, img: Image.Image) -> Image.Image:
        """Simulate partial covering/damage"""
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Random occlusion from edges
        edge = random.choice(['top', 'bottom', 'left', 'right', 'corner'])
        
        if edge == 'top':
            occlusion_height = random.randint(h//10, h//4)
            draw.rectangle([0, 0, w, occlusion_height], fill='white')
        elif edge == 'bottom':
            occlusion_height = random.randint(h//10, h//4)
            draw.rectangle([0, h-occlusion_height, w, h], fill='white')
        elif edge == 'left':
            occlusion_width = random.randint(w//10, w//4)
            draw.rectangle([0, 0, occlusion_width, h], fill='white')
        elif edge == 'right':
            occlusion_width = random.randint(w//10, w//4)
            draw.rectangle([w-occlusion_width, 0, w, h], fill='white')
        elif edge == 'corner':
            corner_size = random.randint(min(w,h)//8, min(w,h)//4)
            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            if corner == 'tl':
                draw.rectangle([0, 0, corner_size, corner_size], fill='white')
            elif corner == 'tr':
                draw.rectangle([w-corner_size, 0, w, corner_size], fill='white')
            elif corner == 'bl':
                draw.rectangle([0, h-corner_size, corner_size, h], fill='white')
            elif corner == 'br':
                draw.rectangle([w-corner_size, h-corner_size, w, h], fill='white')
        
        return img
    
    def _gradient_overlay(self, img: Image.Image) -> Image.Image:
        """Add gradient overlay to simulate lighting issues"""
        w, h = img.size
        
        # Create gradient mask
        gradient = Image.new('L', (w, h))
        draw = ImageDraw.Draw(gradient)
        
        # Random gradient direction
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            for x in range(w):
                intensity = int(255 * x / w)
                draw.line([(x, 0), (x, h)], fill=intensity)
        elif direction == 'vertical':
            for y in range(h):
                intensity = int(255 * y / h)
                draw.line([(0, y), (w, y)], fill=intensity)
        else:  # diagonal
            for x in range(w):
                for y in range(h):
                    intensity = int(255 * (x + y) / (w + h))
                    draw.point((x, y), fill=intensity)
        
        # Apply gradient overlay
        img = img.convert('RGBA')
        overlay = Image.new('RGBA', (w, h), (128, 128, 128, 100))
        overlay.putalpha(gradient)
        
        return Image.alpha_composite(img, overlay).convert('RGB')
    
    def _logo_insertion(self, img: Image.Image) -> Image.Image:
        """Insert fake logos/watermarks"""
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Simple geometric "logo"
        logo_size = random.randint(w//12, w//6)
        x = random.randint(logo_size, w - logo_size)
        y = random.randint(logo_size, h - logo_size)
        
        logo_type = random.choice(['circle', 'square', 'triangle'])
        color = random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
        
        if logo_type == 'circle':
            draw.ellipse([x-logo_size//2, y-logo_size//2, 
                         x+logo_size//2, y+logo_size//2], fill=color)
        elif logo_type == 'square':
            draw.rectangle([x-logo_size//2, y-logo_size//2, 
                           x+logo_size//2, y+logo_size//2], fill=color)
        elif logo_type == 'triangle':
            points = [(x, y-logo_size//2), 
                     (x-logo_size//2, y+logo_size//2), 
                     (x+logo_size//2, y+logo_size//2)]
            draw.polygon(points, fill=color)
        
        return img
    
    def save_dataset_metadata(self, valid_codes: List[Dict], tampered_codes: List[Dict]):
        """Save dataset metadata and statistics"""
        
        all_metadata = valid_codes + tampered_codes
        
        # Save complete metadata
        with open(self.output_dir / "metadata" / "dataset_metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Generate statistics
        stats = {
            'total_samples': len(all_metadata),
            'valid_samples': len(valid_codes),
            'tampered_samples': len(tampered_codes),
            'tampering_methods': {},
            'size_distribution': {},
            'error_correction_distribution': {}
        }
        
        for method in self.tampering_methods:
            count = sum(1 for code in tampered_codes if code['tampering_method'] == method)
            stats['tampering_methods'][method] = count
        
        for code in all_metadata:
            size = code['size']
            stats['size_distribution'][size] = stats['size_distribution'].get(size, 0) + 1
            
            if 'error_correction' in code:
                ec = code['error_correction']
                stats['error_correction_distribution'][ec] = stats['error_correction_distribution'].get(ec, 0) + 1
        
        # Save statistics
        with open(self.output_dir / "metadata" / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset created successfully!")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Valid: {stats['valid_samples']}, Tampered: {stats['tampered_samples']}")

def main():
    parser = argparse.ArgumentParser(description="Generate QR Code dataset for SecurePayQR")
    parser.add_argument("--num_valid", type=int, default=1000, help="Number of valid QR codes")
    parser.add_argument("--tampering_ratio", type=float, default=1.0, help="Ratio of tampered to valid codes")
    parser.add_argument("--output_dir", type=str, default="qr_dataset", help="Output directory")
    
    args = parser.parse_args()
    
    # Create generator
    generator = QRDatasetGenerator(args.output_dir)
    
    # Generate valid QR codes
    valid_codes = generator.generate_valid_qr_codes(args.num_valid)
    
    # Generate tampered QR codes
    tampered_codes = generator.generate_tampered_qr_codes(valid_codes, args.tampering_ratio)
    
    # Save metadata
    generator.save_dataset_metadata(valid_codes, tampered_codes)
    
    print(f"\nDataset generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Valid QR codes: {len(valid_codes)}")
    print(f"Tampered QR codes: {len(tampered_codes)}")

if __name__ == "__main__":
    main()