"""
Installation:
pip install torch torchvision
pip install opencv-python pillow numpy scikit-learn
pip install transformers  # For CLIP (optional)
# OR
pip install open-clip-torch  # Alternative CLIP (optional)

If CLIP libraries fail, the script will fallback to ResNet50 or color histograms.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

# For visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Try different CLIP implementations
try:
    import open_clip as clip
    CLIP_AVAILABLE = True
except ImportError:
    try:
        from transformers import CLIPProcessor, CLIPModel
        CLIP_AVAILABLE = "transformers"
    except ImportError:
        CLIP_AVAILABLE = False

import hashlib
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


def structural_similarity(img1_path: str, img2_path: str, size=(256, 256)) -> float:
    """Compare two images using SSIM (robust to dimming/rotation)."""
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0.0
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)

        # Enhanced to handle brightness variations
        img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
        
        score, _ = ssim(img1_norm, img2_norm, full=True)
        return float(score)
    except Exception:
        return 0.0


def rotation_invariant_similarity(img1_path: str, img2_path: str) -> float:
    """Compare images with rotation invariance for orchid dataset."""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            return 0.0
            
        # Resize for faster processing
        size = (256, 256)
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)
        
        # Convert to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Normalize brightness
        img1_gray = cv2.normalize(img1_gray, None, 0, 255, cv2.NORM_MINMAX)
        img2_gray = cv2.normalize(img2_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Try multiple rotations
        angles = [0, 90, 180, 270]
        max_score = 0.0
        
        for angle in angles:
            if angle == 0:
                rotated = img2_gray
            else:
                M = cv2.getRotationMatrix2D((size[0]//2, size[1]//2), angle, 1)
                rotated = cv2.warpAffine(img2_gray, M, size)
                
            score, _ = ssim(img1_gray, rotated, full=True)
            max_score = max(max_score, score)
            
        return float(max_score)
    except Exception as e:
        print(f"Error in rotation comparison: {e}")
        return 0.0


def foreground_focused_similarity(img1_path: str, img2_path: str) -> float:
    """Compare images focusing on the foreground object (orchid) rather than background."""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            return 0.0
            
        # Resize for faster processing
        size = (256, 256)
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)
        
        # Extract foreground using simple thresholding
        # This works well for orchid images with controlled backgrounds
        
        # Convert to HSV for better segmentation
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel which often separates flower from background
        img1_s = img1_hsv[:,:,1]
        img2_s = img2_hsv[:,:,1]
        
        # Threshold to get foreground mask
        _, img1_mask = cv2.threshold(img1_s, 30, 255, cv2.THRESH_BINARY)
        _, img2_mask = cv2.threshold(img2_s, 30, 255, cv2.THRESH_BINARY)
        
        # Apply mask to original grayscale image
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        img1_fg = cv2.bitwise_and(img1_gray, img1_gray, mask=img1_mask)
        img2_fg = cv2.bitwise_and(img2_gray, img2_gray, mask=img2_mask)
        
        # Compare foregrounds
        score, _ = ssim(img1_fg, img2_fg, full=True)
        return float(score)
    except Exception as e:
        print(f"Error in foreground comparison: {e}")
        return 0.0


class ImageQualityAssessment:
    """Advanced Image Quality Assessment optimized for DINOv2-based duplicate detection"""

    @staticmethod
    def laplacian_variance(image_path: str) -> float:
        """Calculate sharpness using Laplacian variance"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.0
            return cv2.Laplacian(image, cv2.CV_64F).var()
        except Exception:
            return 0.0

    @staticmethod
    def brightness_contrast_score(image_path: str) -> Tuple[float, float]:
        """Calculate brightness and contrast metrics"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.0, 0.0

            brightness = np.mean(image)
            contrast = np.std(image)
            return float(brightness), float(contrast)
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def brisque_score(image_path: str) -> float:
        """Simplified BRISQUE-like score for noise assessment"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 100.0  # High score = bad quality

            # Convert to float
            image = image.astype(np.float32)

            # Calculate local mean
            mu = cv2.GaussianBlur(image, (7, 7), 1.166)
            mu_sq = mu * mu

            # Calculate local variance
            sigma = cv2.GaussianBlur(image * image, (7, 7), 1.166)
            sigma = (sigma - mu_sq) ** 0.5

            # Avoid division by zero
            sigma[sigma == 0] = 1

            # Calculate structural information
            structdis = (image - mu) / sigma

            # Simple quality score (lower is better)
            quality_score = np.mean(np.abs(structdis))
            return float(quality_score)
        except Exception:
            return 100.0

    @staticmethod
    def resolution_score(image_path: str) -> float:
        """Score based on resolution (higher resolution = better)"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                return float(width * height)
        except Exception:
            return 0.0
    
    @staticmethod
    def edge_density_score(image_path: str) -> float:
        """Calculate edge density for detail assessment"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.0
            
            # Apply Canny edge detection
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
            return float(edge_density)
        except Exception:
            return 0.0
    
    @staticmethod
    def color_richness_score(image_path: str) -> float:
        """Calculate color richness and saturation"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return 0.0
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate saturation statistics
            saturation = hsv[:, :, 1]
            mean_saturation = np.mean(saturation)
            
            # Calculate color histogram diversity
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # Normalize histograms
            hist_h = hist_h / np.sum(hist_h)
            hist_s = hist_s / np.sum(hist_s)
            
            # Calculate entropy (diversity)
            entropy_h = -np.sum(hist_h * np.log(hist_h + 1e-10))
            entropy_s = -np.sum(hist_s * np.log(hist_s + 1e-10))
            
            # Combine metrics
            color_score = (mean_saturation / 255.0) * 0.4 + (entropy_h / np.log(180)) * 0.3 + (entropy_s / np.log(256)) * 0.3
            return float(color_score)
        except Exception:
            return 0.0
    
    @staticmethod
    def exposure_quality_score(image_path: str) -> float:
        """Assess exposure quality (avoid over/under exposure)"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return 0.0
            
            # Calculate histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten() / np.sum(hist)
            
            # Check for clipping (over/under exposure)
            underexposed = np.sum(hist[:10])  # Very dark pixels
            overexposed = np.sum(hist[245:])  # Very bright pixels
            
            # Penalize extreme exposure
            clipping_penalty = (underexposed + overexposed) * 2
            
            # Reward good distribution
            mid_range = np.sum(hist[50:200])  # Well-exposed range
            
            exposure_score = max(0.0, mid_range - clipping_penalty)
            return float(exposure_score)
        except Exception:
            return 0.0
    
    @classmethod
    def comprehensive_quality_score(cls, image_path: str) -> Dict[str, float]:
        """Calculate comprehensive quality metrics optimized for DINOv2"""
        try:
            # Get individual metrics
            sharpness = cls.laplacian_variance(image_path)
            brightness, contrast = cls.brightness_contrast_score(image_path)
            noise_score = cls.brisque_score(image_path)  # Lower is better
            resolution = cls.resolution_score(image_path)
            edge_density = cls.edge_density_score(image_path)
            color_richness = cls.color_richness_score(image_path)
            exposure_quality = cls.exposure_quality_score(image_path)
            
            # Normalize scores (0-1 scale, higher is better)
            # Sharpness: normalize by typical range
            sharpness_norm = min(1.0, sharpness / 1000.0)
            
            # Brightness: optimal around 127, penalize extremes
            brightness_norm = 1.0 - abs(brightness - 127) / 127.0
            
            # Contrast: normalize by typical range
            contrast_norm = min(1.0, contrast / 80.0)
            
            # Noise: invert since lower is better, normalize
            noise_norm = max(0.0, 1.0 - (noise_score / 10.0))
            
            # Resolution: normalize by megapixels (assume 12MP as reference)
            resolution_norm = min(1.0, resolution / 12000000.0)
            
            # Edge density, color richness, and exposure are already 0-1
            
            # Calculate weighted composite score optimized for DINOv2
            # DINOv2 benefits from sharp, well-exposed, colorful images
            composite_score = (
                sharpness_norm * 0.25 +      # Sharpness is crucial
                brightness_norm * 0.15 +     # Good exposure
                contrast_norm * 0.15 +       # Good contrast
                noise_norm * 0.15 +          # Low noise
                resolution_norm * 0.10 +     # Higher resolution
                edge_density * 0.10 +        # Rich details
                color_richness * 0.05 +      # Color information
                exposure_quality * 0.05       # Proper exposure
            )
            
            return {
                'composite_score': float(composite_score),
                'sharpness': float(sharpness_norm),
                'brightness': float(brightness_norm),
                'contrast': float(contrast_norm),
                'noise_quality': float(noise_norm),
                'resolution': float(resolution_norm),
                'edge_density': float(edge_density),
                'color_richness': float(color_richness),
                'exposure_quality': float(exposure_quality),
                'raw_sharpness': float(sharpness),
                'raw_brightness': float(brightness),
                'raw_contrast': float(contrast),
                'raw_noise': float(noise_score),
                'raw_resolution': float(resolution)
            }
        except Exception as e:
            print(f"Error calculating quality score for {image_path}: {e}")
            return {
                'composite_score': 0.0,
                'sharpness': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'noise_quality': 0.0,
                'resolution': 0.0,
                'edge_density': 0.0,
                'color_richness': 0.0,
                'exposure_quality': 0.0,
                'raw_sharpness': 0.0,
                'raw_brightness': 0.0,
                'raw_contrast': 0.0,
                'raw_noise': 100.0,
                'raw_resolution': 0.0
            }


class DuplicateDetector:
    """Detect duplicate and near-duplicate images"""

    @staticmethod
    def perceptual_hash(image_path: str, hash_size: int = 8) -> str:
        """Generate perceptual hash for image"""
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale and resize
                img = img.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
                pixels = np.array(img).flatten()

                # Calculate average
                avg = pixels.mean()

                # Generate hash
                hash_bits = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
                return hash_bits
        except Exception:
            return ''

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes"""
        if len(hash1) != len(hash2):
            return len(hash1)  # Maximum distance if lengths differ
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    @staticmethod
    def file_hash(image_path: str) -> str:
        """Generate MD5 hash of file content for exact duplicate detection"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ''


class ContentComplexityAnalyzer:
    """Analyze image content complexity to distinguish between full plants, leaves, and plant parts"""
    
    @staticmethod
    def analyze_content_complexity(image_path: str) -> Dict[str, float]:
        """Analyze image content complexity and return various metrics"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'complexity_score': 0.5, 'object_count': 0, 'detail_level': 0.5, 'completeness_score': 0.5}
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate various complexity metrics
            edge_complexity = ContentComplexityAnalyzer._calculate_edge_complexity(gray)
            texture_complexity = ContentComplexityAnalyzer._calculate_texture_complexity(gray)
            color_complexity = ContentComplexityAnalyzer._calculate_color_complexity(hsv)
            shape_complexity = ContentComplexityAnalyzer._calculate_shape_complexity(gray)
            object_count = ContentComplexityAnalyzer._estimate_object_count(gray)
            completeness_score = ContentComplexityAnalyzer._assess_completeness(image, gray)
            
            # Combine metrics into overall complexity score
            complexity_score = (
                edge_complexity * 0.25 +
                texture_complexity * 0.25 +
                color_complexity * 0.2 +
                shape_complexity * 0.15 +
                min(object_count / 5.0, 1.0) * 0.15  # Normalize object count
            )
            
            return {
                'complexity_score': min(complexity_score, 1.0),
                'edge_complexity': edge_complexity,
                'texture_complexity': texture_complexity,
                'color_complexity': color_complexity,
                'shape_complexity': shape_complexity,
                'object_count': object_count,
                'completeness_score': completeness_score,
                'detail_level': (edge_complexity + texture_complexity) / 2
            }
            
        except Exception as e:
            print(f"Error analyzing content complexity for {image_path}: {e}")
            return {'complexity_score': 0.5, 'object_count': 0, 'detail_level': 0.5, 'completeness_score': 0.5}
    
    @staticmethod
    def _calculate_edge_complexity(gray_image: np.ndarray) -> float:
        """Calculate edge complexity using Canny edge detection"""
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return min(edge_density * 10, 1.0)  # Normalize to 0-1
    
    @staticmethod
    def _calculate_texture_complexity(gray_image: np.ndarray) -> float:
        """Calculate texture complexity using local binary patterns"""
        # Simple texture analysis using standard deviation of local patches
        h, w = gray_image.shape
        patch_size = 16
        texture_scores = []
        
        for i in range(0, h - patch_size, patch_size // 2):
            for j in range(0, w - patch_size, patch_size // 2):
                patch = gray_image[i:i+patch_size, j:j+patch_size]
                texture_scores.append(np.std(patch))
        
        if texture_scores:
            avg_texture = np.mean(texture_scores)
            return min(avg_texture / 50.0, 1.0)  # Normalize
        return 0.5
    
    @staticmethod
    def _calculate_color_complexity(hsv_image: np.ndarray) -> float:
        """Calculate color complexity based on hue distribution"""
        hue_channel = hsv_image[:, :, 0]
        # Calculate histogram of hues
        hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])
        # Normalize histogram
        hist = hist.flatten() / np.sum(hist)
        # Calculate entropy (diversity of colors)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return min(entropy / 5.0, 1.0)  # Normalize
    
    @staticmethod
    def _calculate_shape_complexity(gray_image: np.ndarray) -> float:
        """Calculate shape complexity using contour analysis"""
        # Find contours
        contours, _ = cv2.findContours(cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Analyze contour complexity
        complexity_scores = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Calculate contour perimeter to area ratio
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                if area > 0:
                    complexity = perimeter / (2 * np.sqrt(np.pi * area))  # Normalized by circle
                    complexity_scores.append(min(complexity / 3.0, 1.0))
        
        return np.mean(complexity_scores) if complexity_scores else 0.5
    
    @staticmethod
    def _estimate_object_count(gray_image: np.ndarray) -> int:
        """Estimate number of distinct objects/regions in the image"""
        # Use watershed algorithm for object separation
        # Apply threshold
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(opening)
        
        # Filter out background and very small components
        unique_labels, counts = np.unique(labels, return_counts=True)
        significant_objects = sum(1 for i, count in enumerate(counts) if i > 0 and count > 500)  # Skip background (label 0)
        
        return significant_objects
    
    @staticmethod
    def _assess_completeness(image: np.ndarray, gray_image: np.ndarray) -> float:
        """Assess if the image shows complete objects vs partial/cropped objects"""
        h, w = gray_image.shape
        
        # Check if main objects touch image borders (indicating cropping)
        border_thickness = min(h, w) // 20
        
        # Create border mask
        border_mask = np.zeros_like(gray_image)
        border_mask[:border_thickness, :] = 255  # Top
        border_mask[-border_thickness:, :] = 255  # Bottom
        border_mask[:, :border_thickness] = 255  # Left
        border_mask[:, -border_thickness:] = 255  # Right
        
        # Find significant contours
        contours, _ = cv2.findContours(cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5
        
        # Check if major contours intersect with borders
        border_intersections = 0
        total_significant_contours = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (h * w * 0.01):  # Only consider contours that are at least 1% of image
                total_significant_contours += 1
                # Create contour mask
                contour_mask = np.zeros_like(gray_image)
                cv2.fillPoly(contour_mask, [contour], 255)
                
                # Check intersection with border
                intersection = cv2.bitwise_and(contour_mask, border_mask)
                if np.sum(intersection) > 0:
                    border_intersections += 1
        
        if total_significant_contours == 0:
            return 0.5
        
        # Higher completeness score means less cropping (fewer border intersections)
        completeness = 1.0 - (border_intersections / total_significant_contours)
        return completeness
    
    @staticmethod
    def classify_content_type(complexity_metrics: Dict[str, float], threshold: float = 0.6) -> str:
        """Classify content as 'complex' (full plants) or 'simple' (leaves, parts)"""
        complexity_score = complexity_metrics.get('complexity_score', 0.5)
        completeness_score = complexity_metrics.get('completeness_score', 0.5)
        object_count = complexity_metrics.get('object_count', 0)
        
        # Weighted decision based on multiple factors
        if (complexity_score > threshold and 
            completeness_score > 0.6 and 
            object_count >= 2):
            return 'complex'  # Full plant with multiple parts
        elif complexity_score > threshold * 1.2:  # Very high complexity
            return 'complex'
        else:
            return 'simple'  # Leaf or plant part


class ImageEmbeddings:
    """Extract image embeddings for semantic similarity using DINOv2 (superior to CLIP)"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.processor = None

        # Try DINOv2 first (best for image similarity)
        try:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
            self.model.eval()
            self.method = "dinov2"
            print("Using DINOv2-base for superior image embeddings")
        except Exception as e:
            print(f"DINOv2-base failed: {e}, trying DINOv2-small...")
            
            # Try DINOv2 small (faster alternative)
            try:
                from transformers import AutoImageProcessor, AutoModel
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
                self.model = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device)
                self.model.eval()
                self.method = "dinov2_small"
                print("Using DINOv2-small for embeddings")
            except Exception as e:
                print(f"DINOv2-small failed: {e}, trying CLIP fallback...")
                
                # Fallback to CLIP (legacy)
                if CLIP_AVAILABLE == True:
                    try:
                        # OpenCLIP
                        self.model, _, self.preprocess = clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
                        self.model = self.model.to(self.device)
                        self.model.eval()
                        self.method = "openclip"
                        print("Using OpenCLIP for embeddings (fallback)")
                    except Exception as e:
                        print(f"OpenCLIP failed: {e}")
                        self.model = None

                elif CLIP_AVAILABLE == "transformers":
                    try:
                        # HuggingFace transformers CLIP
                        from transformers import CLIPProcessor, CLIPModel
                        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                        self.model.eval()
                        self.method = "transformers"
                        print("Using HuggingFace CLIP for embeddings (fallback)")
                    except Exception as e:
                        print(f"Transformers CLIP failed: {e}")
                        self.model = None

                # Final fallback to ResNet if all else fails
                if self.model is None:
                    try:
                        from torchvision import models, transforms
                        print("All advanced models failed, using ResNet50 for embeddings...")
                        self.model = models.resnet50(pretrained=True)
                        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove classifier
                        self.model = self.model.to(self.device)
                        self.model.eval()

                        self.preprocess = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        self.method = "resnet"
                    except Exception as e:
                        print(f"ResNet fallback failed: {e}")
                        self.model = None
                        self.method = "none"
                        print("Using color histogram features as final fallback")

    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract embedding for an image using DINOv2 or fallback models"""
        if self.model is None:
            # Fallback to simple histogram features
            return self._extract_histogram_features(image_path)

        try:
            image = Image.open(image_path).convert('RGB')

            if self.method in ["dinov2", "dinov2_small"]:
                # DINOv2 processing
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use the last hidden state and take the CLS token (first token)
                    image_features = outputs.last_hidden_state[:, 0, :]
                    image_features = F.normalize(image_features, p=2, dim=1)
                return image_features.cpu().numpy().flatten()

            elif self.method == "openclip":
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensor)
                    image_features = F.normalize(image_features, p=2, dim=1)
                return image_features.cpu().numpy().flatten()

            elif self.method == "transformers":
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    image_features = F.normalize(image_features, p=2, dim=1)
                return image_features.cpu().numpy().flatten()

            elif self.method == "resnet":
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model(image_tensor)
                    features = features.view(features.size(0), -1)
                    features = F.normalize(features, p=2, dim=1)
                return features.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return self._extract_histogram_features(image_path)

    def _extract_histogram_features(self, image_path: str) -> Optional[np.ndarray]:
        """Fallback: extract simple color histogram features"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Calculate histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

            # Normalize and concatenate
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)

            features = np.concatenate([hist_h, hist_s, hist_v])
            return features.astype(np.float32)

        except Exception as e:
            print(f"Error extracting histogram features from {image_path}: {e}")
            return None


class DuplicateGroup:
    """Represents a group of duplicate/similar images"""

    def __init__(self, group_id: int):
        self.group_id = group_id
        self.images = []
        self.best_image_idx = None

    def add_image(self, image_idx: int, image_data: dict):
        """Add an image to this duplicate group"""
        self.images.append({
            'idx': image_idx,
            'data': image_data
        })

    def find_best_image(self):
        """Find the highest quality image in this group using smart selection logic"""
        if not self.images:
            return None

        # Multi-criteria selection with weighted scoring
        best_image = None
        best_score = -1
        selection_reasons = []

        for image in self.images:
            data = image['data']
            metrics = data.get('quality_metrics', {})
            
            # Calculate weighted composite score
            composite_score = (
                data['quality_score'] * 0.4 +                    # Overall quality (40%)
                metrics.get('sharpness', 0) * 0.15 +            # Sharpness (15%)
                metrics.get('resolution', 0) / 10000.0 * 0.15 + # Resolution (15%)
                metrics.get('edge_density', 0) * 0.1 +          # Edge density (10%)
                metrics.get('color_richness', 0) * 0.1 +        # Color richness (10%)
                metrics.get('exposure_quality', 0) * 0.1        # Exposure quality (10%)
            )
            
            # Bonus for balanced characteristics
            brightness = metrics.get('brightness', 128)
            if 100 <= brightness <= 155:  # Well-exposed images
                composite_score += 0.05
            
            contrast = metrics.get('contrast', 0)
            if contrast > 30:  # Good contrast
                composite_score += 0.03
                
            # Penalty for extreme noise
            noise_quality = metrics.get('noise_quality', 0.5)
            if noise_quality < 0.2:  # Very noisy (low quality)
                composite_score -= 0.1
            
            if composite_score > best_score:
                best_score = composite_score
                best_image = image
                
                # Generate selection reasoning
                reasons = []
                if data['quality_score'] > 0.7:
                    reasons.append("high overall quality")
                if metrics.get('sharpness', 0) > 500:
                    reasons.append("excellent sharpness")
                if metrics.get('resolution', 0) > 5000:
                    reasons.append("high resolution")
                if metrics.get('edge_density', 0) > 0.15:
                    reasons.append("rich detail")
                if metrics.get('color_richness', 0) > 0.6:
                    reasons.append("vibrant colors")
                if 100 <= brightness <= 155:
                    reasons.append("optimal exposure")
                    
                selection_reasons = reasons if reasons else ["best available quality"]

        if best_image:
            self.best_image_idx = best_image['idx']
            # Store selection reasoning for debugging/reporting
            best_image['selection_reasons'] = selection_reasons
            
        return best_image

    def get_all_indices(self) -> List[int]:
        """Get all image indices in this group"""
        return [img['idx'] for img in self.images]


class SmartImageSplitter:
    """Advanced image splitter with DINOv2 integration and multi-level duplicate detection"""

    def __init__(self,
                 train_ratio: float = 0.8,
                 test_ratio: float = 0.2,
                 similarity_threshold: float = 0.90,  # Optimized for DINOv2
                 hash_threshold: int = 5,
                 remove_exact_duplicates: bool = True,
                 keep_near_duplicates: bool = True,
                 rotation_similarity_threshold: float = 0.80,  # DINOv2 better at rotations
                 foreground_similarity_threshold: float = 0.75,  # DINOv2 better at object focus
                 cross_validation_folds: int = 5,
                 # New DINOv2-optimized thresholds
                 exact_duplicate_threshold: float = 0.99,
                 near_duplicate_threshold: float = 0.95,
                 variation_threshold: float = 0.85,
                 semantic_threshold: float = 0.80,
                 # Content-based filtering parameters
                 enable_content_filtering: bool = False,
                 detailed_to_train_ratio: float = 0.7,
                 simple_to_test_ratio: float = 0.6,
                 complexity_threshold: float = 0.6,
                 config_file: str = None):

        # Load configuration from file if provided
        if config_file:
            config = self._load_config(config_file)
            # Override defaults with config values
            train_ratio = config.get('train_ratio', train_ratio)
            keep_near_duplicates = config.get('keep_near_duplicates', keep_near_duplicates)
            
            # Load DINOv2 duplicate resolution settings
            dup_config = config.get('duplicate_resolution', {})
            exact_duplicate_threshold = dup_config.get('exact_duplicate_threshold', exact_duplicate_threshold)
            near_duplicate_threshold = dup_config.get('near_duplicate_threshold', near_duplicate_threshold)
            variation_threshold = dup_config.get('variation_threshold', variation_threshold)
            semantic_threshold = dup_config.get('semantic_threshold', semantic_threshold)
            
            # Load content-based filtering settings
            content_config = config.get('content_filtering', {})
            enable_content_filtering = content_config.get('enable_content_filtering', enable_content_filtering)
            detailed_to_train_ratio = content_config.get('detailed_to_train_ratio', detailed_to_train_ratio)
            simple_to_test_ratio = content_config.get('simple_to_test_ratio', simple_to_test_ratio)
            complexity_threshold = content_config.get('complexity_threshold', complexity_threshold)

        assert abs(train_ratio + test_ratio - 1.0) < 1e-6, "Train and test ratios must sum to 1.0"

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.similarity_threshold = similarity_threshold
        self.hash_threshold = hash_threshold
        self.remove_exact_duplicates = remove_exact_duplicates
        self.keep_near_duplicates = keep_near_duplicates
        self.rotation_similarity_threshold = rotation_similarity_threshold
        self.foreground_similarity_threshold = foreground_similarity_threshold
        self.cross_validation_folds = cross_validation_folds
        
        # New multi-level thresholds for DINOv2
        self.exact_duplicate_threshold = exact_duplicate_threshold
        self.near_duplicate_threshold = near_duplicate_threshold
        self.variation_threshold = variation_threshold
        self.semantic_threshold = semantic_threshold
        
        # Content-based filtering settings
        self.enable_content_filtering = enable_content_filtering
        self.detailed_to_train_ratio = detailed_to_train_ratio
        self.simple_to_test_ratio = simple_to_test_ratio
        self.complexity_threshold = complexity_threshold

        self.embedding_model = ImageEmbeddings()  # Now supports DINOv2
        self.quality_assessor = ImageQualityAssessment()
        self.duplicate_detector = DuplicateDetector()
        self.complexity_analyzer = ContentComplexityAnalyzer()

        self.image_data = []
        self.duplicate_groups = []
        self.exact_duplicates = []
        self.unique_images = []  # Images that will actually be used
        
        # Model evaluation metrics
        self.model_performance = {}
        self.best_model_config = None
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            return {}
        
    def analyze_images(self, image_folder: str) -> None:
        """Analyze all images in the folder"""
        print("🔍 Analyzing images...")

        # image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.GIF', '.TIFF'}
        image_extensions = {'.JPG'}  # Restricted for testing
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(Path(image_folder).glob(f'*{ext}'))

        print(f"Found {len(image_paths)} images")

        # Extract features for each image
        for i, image_path in enumerate(image_paths):
            if i % 50 == 0:
                print(f"Processing image {i + 1}/{len(image_paths)}")

            # Extract embedding
            embedding = self.embedding_model.extract_embedding(str(image_path))
            if embedding is None:
                continue

            # Calculate comprehensive quality metrics
            quality_metrics = self.quality_assessor.comprehensive_quality_score(str(image_path))
            
            # Generate hashes
            phash = self.duplicate_detector.perceptual_hash(str(image_path))
            file_hash = self.duplicate_detector.file_hash(str(image_path))
            
            # Analyze content complexity if content filtering is enabled
            complexity_metrics = None
            content_type = 'unknown'
            if self.enable_content_filtering:
                try:
                    complexity_metrics = self.complexity_analyzer.analyze_content_complexity(str(image_path))
                    content_type = self.complexity_analyzer.classify_content_type(
                        complexity_metrics, self.complexity_threshold
                    )
                except Exception as e:
                    print(f"Warning: Could not analyze content complexity for {image_path}: {e}")
                    complexity_metrics = {
                        'overall_complexity': 0.5,
                        'completeness_score': 0.5,
                        'edge_complexity': 0.5,
                        'texture_complexity': 0.5,
                        'color_complexity': 0.5,
                        'shape_complexity': 0.5,
                        'object_count': 1
                    }
                    content_type = 'simple'

            # Use the comprehensive quality score
            quality_score = quality_metrics['composite_score']

            image_data_entry = {
                'path': str(image_path),
                'embedding': embedding,
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'sharpness': quality_metrics['sharpness'],
                'brightness': quality_metrics['brightness'],
                'contrast': quality_metrics['contrast'],
                'brisque': quality_metrics['noise_quality'],
                'resolution': quality_metrics['resolution'],
                'edge_density': quality_metrics['edge_density'],
                'color_richness': quality_metrics['color_richness'],
                'exposure_quality': quality_metrics['exposure_quality'],
                'phash': phash,
                'file_hash': file_hash,
                'original_index': i
            }
            
            # Add content complexity data if available
            if complexity_metrics:
                image_data_entry.update({
                    'complexity_metrics': complexity_metrics,
                    'content_type': content_type,
                    'complexity_score': complexity_metrics['complexity_score'],
                    'completeness_score': complexity_metrics['completeness_score']
                })
            
            self.image_data.append(image_data_entry)

    def detect_duplicates_advanced(self) -> None:
        """Multi-level duplicate detection optimized for DINOv2"""
        print("🔍 Performing multi-level duplicate detection with DINOv2...")

        if not self.image_data:
            return

        all_groups = []
        
        # Level 1: Exact duplicates (file hash + perceptual hash)
        print("Level 1: Detecting exact duplicates...")
        exact_groups = self._detect_exact_duplicates()
        all_groups.extend(exact_groups)
        print(f"Found {len(exact_groups)} exact duplicate groups")

        # Level 2: Near duplicates (DINOv2 + perceptual hash)
        print("Level 2: Detecting near duplicates...")
        near_groups = self._detect_near_duplicates()
        all_groups.extend(near_groups)
        print(f"Found {len(near_groups)} near duplicate groups")

        # Level 3: Variations (rotations, crops, lighting)
        print("Level 3: Detecting variations...")
        variation_groups = self._detect_variations()
        all_groups.extend(variation_groups)
        print(f"Found {len(variation_groups)} variation groups")

        # Level 4: Semantic duplicates (same object, different angle)
        print("Level 4: Detecting semantic duplicates...")
        semantic_groups = self._detect_semantic_duplicates()
        all_groups.extend(semantic_groups)
        print(f"Found {len(semantic_groups)} semantic duplicate groups")

        # Merge overlapping groups
        print("Merging overlapping groups...")
        merged_groups = self._merge_overlapping_groups(all_groups)
        
        self.exact_duplicates = merged_groups
        print(f"Final result: {len(merged_groups)} unique duplicate groups")

    def _detect_exact_duplicates(self) -> List[List[int]]:
        """Detect exact duplicates using file hash and perceptual hash"""
        exact_duplicate_groups = defaultdict(list)
        perceptual_duplicate_groups = defaultdict(list)
        
        # File hash duplicates (identical files)
        for i, data in enumerate(self.image_data):
            if data['file_hash']:
                exact_duplicate_groups[data['file_hash']].append(i)
        
        # Perceptual hash duplicates (visually identical)
        for i, data in enumerate(self.image_data):
            if data['phash']:
                perceptual_duplicate_groups[data['phash']].append(i)
        
        # Combine both types of exact duplicates
        all_groups = []
        all_groups.extend([indices for indices in exact_duplicate_groups.values() if len(indices) > 1])
        all_groups.extend([indices for indices in perceptual_duplicate_groups.values() if len(indices) > 1])
        
        return self._merge_overlapping_groups(all_groups)

    def _detect_near_duplicates(self) -> List[List[int]]:
        """Detect near duplicates using DINOv2 embeddings and perceptual hash with high threshold"""
        all_groups = []
        
        # Method 1: DINOv2 embeddings for semantic similarity
        if self.embedding_model and self.embedding_model.method != "none":
            embeddings = np.array([data['embedding'] for data in self.image_data])
            similarity_matrix = cosine_similarity(embeddings)
            dinov2_groups = self._find_similarity_groups(similarity_matrix, self.near_duplicate_threshold)
            all_groups.extend(dinov2_groups)
        
        # Method 2: Perceptual hash for visual similarity (catch obvious duplicates)
        perceptual_groups = self._detect_perceptual_near_duplicates()
        all_groups.extend(perceptual_groups)
        
        return self._merge_overlapping_groups(all_groups)
    
    def _detect_perceptual_near_duplicates(self) -> List[List[int]]:
        """Detect near duplicates using perceptual hash hamming distance"""
        groups = []
        processed = set()
        
        for i in range(len(self.image_data)):
            if i in processed or not self.image_data[i]['phash']:
                continue
                
            group = [i]
            hash1 = self.image_data[i]['phash']
            
            for j in range(i + 1, len(self.image_data)):
                if j in processed or not self.image_data[j]['phash']:
                    continue
                    
                hash2 = self.image_data[j]['phash']
                hamming_dist = DuplicateDetector.hamming_distance(hash1, hash2)
                
                # Very low hamming distance indicates near-identical images
                if hamming_dist <= 3:  # More aggressive threshold for obvious duplicates
                    group.append(j)
            
            if len(group) > 1:
                groups.append(group)
                processed.update(group)
                
        return groups

    def _detect_variations(self) -> List[List[int]]:
        """Detect variations (rotations, crops, lighting) using DINOv2 with medium threshold"""
        if not self.embedding_model or self.embedding_model.method == "none":
            return []
            
        embeddings = np.array([data['embedding'] for data in self.image_data])
        similarity_matrix = cosine_similarity(embeddings)
        
        return self._find_similarity_groups(similarity_matrix, self.variation_threshold)

    def _detect_semantic_duplicates(self) -> List[List[int]]:
        """Detect semantic duplicates (same object, different angle) using DINOv2 with low threshold"""
        if not self.embedding_model or self.embedding_model.method == "none":
            return []
            
        embeddings = np.array([data['embedding'] for data in self.image_data])
        similarity_matrix = cosine_similarity(embeddings)
        
        return self._find_similarity_groups(similarity_matrix, self.semantic_threshold)

    def _find_similarity_groups(self, similarity_matrix: np.ndarray, threshold: float) -> List[List[int]]:
        """Find groups of similar images based on similarity matrix and threshold"""
        groups = []
        processed = set()

        for i in range(len(similarity_matrix)):
            if i in processed:
                continue

            group = [i]
            for j in range(i + 1, len(similarity_matrix)):
                if j in processed:
                    continue

                if similarity_matrix[i, j] > threshold:
                    group.append(j)

            if len(group) > 1:
                groups.append(group)
                processed.update(group)

        return groups

    def _merge_overlapping_groups(self, groups: List[List[int]]) -> List[List[int]]:
        """Merge overlapping duplicate groups"""
        if not groups:
            return []

        # Convert to sets for easier manipulation
        group_sets = [set(group) for group in groups]
        merged = []

        while group_sets:
            current = group_sets.pop(0)

            # Find all groups that overlap with current
            overlapping = []
            non_overlapping = []

            for group_set in group_sets:
                if current & group_set:  # If there's intersection
                    overlapping.append(group_set)
                else:
                    non_overlapping.append(group_set)

            # Merge all overlapping groups
            for overlapping_group in overlapping:
                current |= overlapping_group

            merged.append(list(current))
            group_sets = non_overlapping

        return merged

    def create_unique_dataset(self) -> None:
        """Create dataset with unique images, handling duplicates appropriately"""
        print("🎯 Creating unique dataset...")

        # Track which images to remove completely
        images_to_remove = set()
        
        # Handle exact duplicates - always remove extras, keep best
        if self.remove_exact_duplicates and self.exact_duplicates:
            print(f"Removing exact duplicates, keeping best quality image from each group...")
            for group in self.exact_duplicates:
                # Keep only the best quality image from each exact duplicate group
                best_idx = max(group, key=lambda idx: self.image_data[idx]['quality_score'])
                # Mark others for removal
                for idx in group:
                    if idx != best_idx:
                        images_to_remove.add(idx)

        # Handle near-duplicates based on keep_near_duplicates setting
        if not self.keep_near_duplicates and self.duplicate_groups:
            print(f"Removing near-duplicates, keeping best quality image from each group...")
            for group in self.duplicate_groups:
                # Keep only the best quality image from each group
                best_idx = max(group, key=lambda idx: self.image_data[idx]['quality_score'])
                # Mark others for removal
                for idx in group:
                    if idx != best_idx:
                        images_to_remove.add(idx)
        elif self.keep_near_duplicates:
            print(f"Keeping near-duplicates for smart splitting...")
            # Don't remove near-duplicates, but we'll handle them in splitting

        # Create unique dataset (removing only marked images)
        self.unique_images = []
        original_to_unique_mapping = {}  # Track index mapping
        
        for i, data in enumerate(self.image_data):
            if i not in images_to_remove:
                original_to_unique_mapping[i] = len(self.unique_images)
                # Add mapping for splitting algorithm
                data_copy = data.copy()
                data_copy['original_index'] = i
                self.unique_images.append(data_copy)

        # Update duplicate groups to use new indices in unique dataset
        if self.keep_near_duplicates:
            updated_duplicate_groups = []
            for group in self.duplicate_groups:
                # Map original indices to unique dataset indices
                updated_group = []
                for orig_idx in group:
                    if orig_idx in original_to_unique_mapping:
                        updated_group.append(original_to_unique_mapping[orig_idx])
                
                # Only keep groups with multiple images still present
                if len(updated_group) > 1:
                    updated_duplicate_groups.append(updated_group)
            
            self.duplicate_groups = updated_duplicate_groups
            print(f"Updated duplicate groups for splitting: {len(self.duplicate_groups)} groups")

        print(f"Original images: {len(self.image_data)}")
        print(f"Unique images for splitting: {len(self.unique_images)}")
        print(f"Images removed as duplicates: {len(self.image_data) - len(self.unique_images)}")

    def smart_split_with_duplicate_handling(self) -> Dict[str, List[Dict]]:
        """Intelligently split images ensuring duplicates stay together"""
        print("🧠 Performing smart train/test split with duplicate handling...")

        if not self.unique_images:
            raise ValueError("No unique images found. Run create_unique_dataset() first.")

        # Use the updated duplicate groups from create_unique_dataset
        active_duplicate_groups = []
        if self.keep_near_duplicates and hasattr(self, 'duplicate_groups'):
            # duplicate_groups already contains updated indices for unique dataset
            active_duplicate_groups = self.duplicate_groups.copy()
        
        # Also check for any remaining exact duplicate groups
        if self.exact_duplicates:
            for group in self.exact_duplicates:
                # Map original indices to unique dataset indices
                mapped_group = []
                for orig_idx in group:
                    for i, unique_img in enumerate(self.unique_images):
                        if unique_img.get('original_index') == orig_idx:
                            mapped_group.append(i)
                            break
                
                if len(mapped_group) > 1:  # Only care about groups with multiple images
                    active_duplicate_groups.append(mapped_group)

        print(f"Active duplicate groups in unique dataset: {len(active_duplicate_groups)}")

        n_total = len(self.unique_images)
        n_train = int(n_total * self.train_ratio)

        # Sort by quality score (descending)
        sorted_indices = sorted(range(len(self.unique_images)),
                              key=lambda i: self.unique_images[i]['quality_score'],
                              reverse=True)

        # Initialize assignments
        assignments = {}  # index -> 'train' or 'test'
        assigned_count = {'train': 0, 'test': 0}

        # Step 1: Handle duplicate groups with proportional allocation
        # Sort groups by size (largest first) for better distribution
        sorted_groups = sorted(active_duplicate_groups, key=len, reverse=True)
        
        for group in sorted_groups:
            group_size = len(group)
            
            # Calculate how many images from this group should go to train
            # Based on current remaining capacity and target ratio
            remaining_train_capacity = n_train - assigned_count['train']
            remaining_test_capacity = (n_total - n_train) - assigned_count['test']
            total_remaining = remaining_train_capacity + remaining_test_capacity
            
            if total_remaining <= 0:
                # No capacity left, shouldn't happen but safety check
                continue
                
            # Calculate target train allocation for this group
            if group_size <= remaining_train_capacity:
                # Group fits entirely in train if we want it there
                target_train_ratio = remaining_train_capacity / total_remaining
            else:
                # Group is larger than remaining train capacity
                target_train_ratio = self.train_ratio
            
            # Decide split based on which gives better overall balance
            if (remaining_train_capacity >= group_size and 
                target_train_ratio >= 0.5) or remaining_test_capacity < group_size:
                split = 'train'
            else:
                split = 'test'
            
            # Assign all images in group to same split (duplicates must stay together)
            for img_idx in group:
                if img_idx not in assignments:
                    assignments[img_idx] = split
                    assigned_count[split] += 1
                    
            print(f"📦 Assigned duplicate group of size {group_size} to {split} "
                  f"(train: {assigned_count['train']}/{n_train}, test: {assigned_count['test']}/{n_total-n_train})")

        # Step 2: Apply content-based filtering if enabled
        if self.enable_content_filtering:
            print(f"🎯 Applying content-based filtering...")
            
            # Analyze content complexity for remaining unassigned images
            unassigned_indices = [idx for idx in sorted_indices if idx not in assignments]
            content_analysis = {}
            
            for img_idx in unassigned_indices:
                image_path = self.unique_images[img_idx]['path']
                try:
                    complexity_metrics = self.complexity_analyzer.analyze_content_complexity(image_path)
                    content_type = self.complexity_analyzer.classify_content_type(
                        complexity_metrics, self.complexity_threshold
                    )
                    content_analysis[img_idx] = {
                        'complexity_score': complexity_metrics['overall_complexity'],
                        'content_type': content_type,
                        'completeness': complexity_metrics['completeness_score']
                    }
                except Exception as e:
                    print(f"Warning: Could not analyze content for {image_path}: {e}")
                    content_analysis[img_idx] = {
                        'complexity_score': 0.5,
                        'content_type': 'simple',
                        'completeness': 0.5
                    }
            
            # Separate images by content type
            detailed_images = [idx for idx in unassigned_indices 
                             if content_analysis[idx]['content_type'] == 'detailed']
            simple_images = [idx for idx in unassigned_indices 
                           if content_analysis[idx]['content_type'] == 'simple']
            
            print(f"   Detailed/complex images: {len(detailed_images)}")
            print(f"   Simple images: {len(simple_images)}")
            
            # Calculate how many detailed images should go to train
            remaining_train_capacity = n_train - assigned_count['train']
            remaining_test_capacity = (n_total - n_train) - assigned_count['test']
            
            detailed_to_train = min(
                len(detailed_images),
                int(len(detailed_images) * self.detailed_to_train_ratio),
                remaining_train_capacity
            )
            
            # Sort detailed images by completeness (full plants first to train)
            detailed_images.sort(
                key=lambda idx: content_analysis[idx]['completeness'], 
                reverse=True
            )
            
            # Assign detailed images
            for i, img_idx in enumerate(detailed_images):
                if i < detailed_to_train and assigned_count['train'] < n_train:
                    assignments[img_idx] = 'train'
                    assigned_count['train'] += 1
                else:
                    assignments[img_idx] = 'test'
                    assigned_count['test'] += 1
            
            # Calculate how many simple images should go to test
            remaining_train_capacity = n_train - assigned_count['train']
            simple_to_test = min(
                len(simple_images),
                int(len(simple_images) * self.simple_to_test_ratio),
                assigned_count['test'] + len(simple_images) - (n_total - n_train)
            )
            
            # Sort simple images by complexity (simplest first to test)
            simple_images.sort(
                key=lambda idx: content_analysis[idx]['complexity_score']
            )
            
            # Assign simple images
            for i, img_idx in enumerate(simple_images):
                if i < simple_to_test and assigned_count['test'] < (n_total - n_train):
                    assignments[img_idx] = 'test'
                    assigned_count['test'] += 1
                else:
                    assignments[img_idx] = 'train'
                    assigned_count['train'] += 1
            
            print(f"   Assigned {detailed_to_train}/{len(detailed_images)} detailed images to train")
            print(f"   Assigned {simple_to_test}/{len(simple_images)} simple images to test")
        
        # Step 3: Assign any remaining images (fallback)
        for img_idx in sorted_indices:
            if img_idx not in assignments:
                # Assign based on current needs
                if assigned_count['train'] < n_train:
                    assignments[img_idx] = 'train'
                    assigned_count['train'] += 1
                else:
                    assignments[img_idx] = 'test'
                    assigned_count['test'] += 1
        
        # Step 4: Rebalance if ratio is significantly off
        actual_train_ratio = assigned_count['train'] / n_total
        ratio_difference = abs(actual_train_ratio - self.train_ratio)
        
        if ratio_difference > 0.05:  # If more than 5% off target
            print(f"⚖️ Rebalancing splits - current ratio: {actual_train_ratio:.3f}, target: {self.train_ratio:.3f}")
            
            # Find moveable groups (not individual images to maintain duplicate integrity)
            moveable_groups = []
            for group in sorted_groups:
                if len(group) <= abs(assigned_count['train'] - n_train):
                    moveable_groups.append(group)
            
            # Sort by size (smallest first for fine-tuning)
            moveable_groups.sort(key=len)
            
            for group in moveable_groups:
                current_split = assignments[group[0]]  # All in group have same split
                group_size = len(group)
                
                # Check if moving this group improves the balance
                if current_split == 'train' and assigned_count['train'] > n_train:
                    # Move from train to test
                    new_train_count = assigned_count['train'] - group_size
                    new_ratio = new_train_count / n_total
                    if abs(new_ratio - self.train_ratio) < ratio_difference:
                        for img_idx in group:
                            assignments[img_idx] = 'test'
                        assigned_count['train'] -= group_size
                        assigned_count['test'] += group_size
                        print(f"🔄 Moved group of {group_size} from train to test")
                        break
                        
                elif current_split == 'test' and assigned_count['train'] < n_train:
                    # Move from test to train
                    new_train_count = assigned_count['train'] + group_size
                    new_ratio = new_train_count / n_total
                    if abs(new_ratio - self.train_ratio) < ratio_difference:
                        for img_idx in group:
                            assignments[img_idx] = 'train'
                        assigned_count['train'] += group_size
                        assigned_count['test'] -= group_size
                        print(f"🔄 Moved group of {group_size} from test to train")
                        break

        # Create final splits
        splits = {
            'train': [self.unique_images[i] for i in range(len(self.unique_images))
                     if assignments.get(i) == 'train'],
            'test': [self.unique_images[i] for i in range(len(self.unique_images))
                    if assignments.get(i) == 'test']
        }
        
        # Final allocation summary
        final_train_ratio = len(splits['train']) / n_total
        final_test_ratio = len(splits['test']) / n_total
        
        print(f"\n📊 Final Split Results:")
        print(f"   Train: {len(splits['train'])} images ({final_train_ratio:.1%})")
        print(f"   Test:  {len(splits['test'])} images ({final_test_ratio:.1%})")
        print(f"   Target ratio: {self.train_ratio:.1%} train / {1-self.train_ratio:.1%} test")
        print(f"   Ratio deviation: {abs(final_train_ratio - self.train_ratio):.1%}")
        
        # Count how many duplicate groups were handled
        groups_in_train = 0
        groups_in_test = 0
        for group in active_duplicate_groups:
            if assignments.get(group[0]) == 'train':
                groups_in_train += 1
            else:
                groups_in_test += 1
        
        print(f"   Duplicate groups: {groups_in_train} in train, {groups_in_test} in test")

        # Verify no duplicates across splits
        self._verify_no_cross_split_duplicates(splits)

        # Print statistics
        for split_name, split_data in splits.items():
            if len(split_data) > 0:
                avg_quality = np.mean([d['quality_score'] for d in split_data])
                print(f"{split_name}: {len(split_data)} images, avg quality: {avg_quality:.3f}")

        return splits

    def _verify_no_cross_split_duplicates(self, splits: Dict[str, List[Dict]]) -> None:
        """Verify that no duplicates exist across train/test splits using multiple hash methods"""
        print("✅ Verifying no cross-split duplicates...")

        train_file_hashes = set()
        train_perceptual_hashes = set()
        
        # Collect hashes from train set
        for img_data in splits['train']:
            if img_data.get('file_hash'):
                train_file_hashes.add(img_data['file_hash'])
            if img_data.get('phash'):
                train_perceptual_hashes.add(img_data['phash'])

        # Check test set for conflicts
        file_conflicts = 0
        perceptual_conflicts = 0
        
        for img_data in splits['test']:
            # Check file hash conflicts (exact duplicates)
            if img_data.get('file_hash') and img_data['file_hash'] in train_file_hashes:
                file_conflicts += 1
                print(f"⚠ Found exact duplicate across splits: {img_data['path']}")
            
            # Check perceptual hash conflicts (visually identical)
            if img_data.get('phash') and img_data['phash'] in train_perceptual_hashes:
                perceptual_conflicts += 1
                print(f"⚠ Found perceptual duplicate across splits: {img_data['path']}")
        
        total_conflicts = file_conflicts + perceptual_conflicts
        if total_conflicts == 0:
            print("✅ No cross-split duplicates found!")
        else:
            print(f"❌ Found {total_conflicts} cross-split duplicates ({file_conflicts} exact, {perceptual_conflicts} perceptual)!")
            print("🔧 Consider adjusting duplicate detection thresholds or removing these duplicates.")

    def copy_files(self, splits: Dict[str, List[Dict]], output_folder: str) -> None:
        """Copy files to train/test folders"""
        print("📁 Copying files...")

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        total_copied = 0
        copy_stats = {}

        for split_name, split_data in splits.items():
            split_folder = output_path / split_name
            split_folder.mkdir(parents=True, exist_ok=True)

            print(f"Copying {len(split_data)} images to {split_name} folder...")

            copied_in_split = 0
            for data in split_data:
                src_path = Path(data['path'])
                dst_path = split_folder / src_path.name

                # Handle naming conflicts in destination
                if dst_path.exists():
                    # Check if it's actually the same file (by comparing file hashes)
                    try:
                        import hashlib
                        def get_file_hash(filepath):
                            with open(filepath, 'rb') as f:
                                return hashlib.md5(f.read()).hexdigest()
                        
                        src_hash = get_file_hash(src_path)
                        dst_hash = get_file_hash(dst_path)
                        
                        if src_hash == dst_hash:
                            # Same file already exists, skip copying to avoid duplicate
                            print(f"Skipping {src_path.name} - identical file already exists in {split_name}")
                            continue
                    except Exception:
                        # If hash comparison fails, proceed with renaming
                        pass
                    
                    # Different files with same name, add counter
                    counter = 1
                    original_dst = dst_path
                    while dst_path.exists():
                        stem = original_dst.stem
                        suffix = original_dst.suffix
                        dst_path = split_folder / f"{stem}_{counter}{suffix}"
                        counter += 1

                try:
                    shutil.copy2(src_path, dst_path)
                    copied_in_split += 1
                    total_copied += 1
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")

            copy_stats[split_name] = copied_in_split

        print(f"✅ Successfully copied {total_copied} images")

        # Verify the copy was successful
        train_count = len(list((output_path / 'train').glob('*'))) if (output_path / 'train').exists() else 0
        test_count = len(list((output_path / 'test').glob('*'))) if (output_path / 'test').exists() else 0

        print(f"📊 Final count - Train: {train_count}, Test: {test_count}, Total: {train_count + test_count}")

        # Save comprehensive analysis
        analysis_file = output_path / 'split_analysis.json'
        analysis_data = {
            'processing_summary': {
                'original_images_found': len(self.image_data),
                'unique_images_processed': len(self.unique_images),
                'images_copied': total_copied,
                'exact_duplicate_groups': len(self.exact_duplicates),
                'near_duplicate_groups': len(self.duplicate_groups),
                'remove_exact_duplicates': self.remove_exact_duplicates,
                'keep_near_duplicates': self.keep_near_duplicates
            },
            'duplicate_analysis': {
                'exact_duplicates': [
                    {
                        'group_id': i,
                        'size': len(group),
                        'images': [self.image_data[idx]['path'] for idx in group if idx < len(self.image_data)]
                    }
                    for i, group in enumerate(self.exact_duplicates)
                ],
                'near_duplicate_groups': [
                    {
                        'group_id': i,
                        'size': len(group),
                        'images': [self.image_data[idx]['path'] for idx in group if idx < len(self.image_data)]
                    }
                    for i, group in enumerate(self.duplicate_groups)
                ]
            },
            'split_results': {
                split_name: {
                    'count': len(split_data),
                    'avg_quality': float(np.mean([d['quality_score'] for d in split_data])) if split_data else 0.0,
                    'avg_sharpness': float(np.mean([d['sharpness'] for d in split_data])) if split_data else 0.0,
                    'avg_resolution': float(np.mean([d['resolution'] for d in split_data])) if split_data else 0.0,
                    'files_copied': copy_stats.get(split_name, 0)
                }
                for split_name, split_data in splits.items()
            },
            'verification': {
                'train_folder_count': train_count,
                'test_folder_count': test_count,
                'total_output_count': train_count + test_count,
                'expected_unique_count': len(self.unique_images)
            },
            'settings': {
                'train_ratio': self.train_ratio,
                'test_ratio': self.test_ratio,
                'similarity_threshold': self.similarity_threshold,
                'hash_threshold': self.hash_threshold,
                'embedding_method': self.embedding_model.method
            }
        }

        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print(f"📊 Detailed analysis saved to {analysis_file}")

        # Alert if there's a count mismatch
        if train_count + test_count != len(self.unique_images):
            print(f"⚠ WARNING: Expected {len(self.unique_images)} images but found {train_count + test_count} in output folders!")
        else:
            print("✅ Image count verification passed!")

        # Summary report
        print("\n" + "="*60)
        print("📋 PROCESSING SUMMARY")
        print("="*60)
        print(f"Original images found: {len(self.image_data)}")
        print(f"Exact duplicate groups: {len(self.exact_duplicates)}")
        print(f"Near-duplicate groups: {len(self.duplicate_groups)}")
        print(f"Unique images processed: {len(self.unique_images)}")
        print(f"Images successfully copied: {total_copied}")
        print(f"Train set: {train_count} images ({train_count/total_copied*100:.1f}%)")
        print(f"Test set: {test_count} images ({test_count/total_copied*100:.1f}%)")
        print("="*60)

    def evaluate_model_configurations(self) -> None:
        """Evaluate different model configurations to find the best one"""
        print("\n🧪 Evaluating different model configurations...")
        
        # Define different configurations to test with meaningful parameter variations
        configurations = [
            # Default configuration
            {
                'name': 'Default',
                'train_ratio': self.train_ratio,
                'test_ratio': self.test_ratio,
                'similarity_threshold': self.similarity_threshold,
                'hash_threshold': self.hash_threshold,
                'exact_duplicate_threshold': getattr(self, 'exact_duplicate_threshold', 0.99),
                'near_duplicate_threshold': getattr(self, 'near_duplicate_threshold', 0.95)
            },
            # Conservative - strict duplicate detection
            {
                'name': 'Conservative',
                'train_ratio': self.train_ratio,
                'test_ratio': self.test_ratio,
                'similarity_threshold': 0.95,  # Much higher threshold
                'hash_threshold': 3,  # Stricter hash matching
                'exact_duplicate_threshold': 0.99,
                'near_duplicate_threshold': 0.98  # Very strict near-duplicate detection
            },
            # Aggressive - loose duplicate detection
            {
                'name': 'Aggressive',
                'train_ratio': self.train_ratio,
                'test_ratio': self.test_ratio,
                'similarity_threshold': 0.75,  # Much lower threshold
                'hash_threshold': 8,  # More lenient hash matching
                'exact_duplicate_threshold': 0.95,
                'near_duplicate_threshold': 0.85  # Loose near-duplicate detection
            },
            # Balanced split focused
            {
                'name': 'Balanced Split',
                'train_ratio': 0.7,  # Different split ratio
                'test_ratio': 0.3,
                'similarity_threshold': 0.88,
                'hash_threshold': 4,
                'exact_duplicate_threshold': 0.98,
                'near_duplicate_threshold': 0.92
            },
            # Quality focused - prioritize high-quality images
            {
                'name': 'Quality Focused',
                'train_ratio': self.train_ratio,
                'test_ratio': self.test_ratio,
                'similarity_threshold': 0.85,
                'hash_threshold': 6,
                'exact_duplicate_threshold': 0.97,
                'near_duplicate_threshold': 0.90,
                'enable_content_filtering': True,
                'complexity_threshold': 0.7  # Higher complexity threshold
            }
        ]
        
        # Evaluate each configuration
        best_score = -1
        best_config = None
        
        for config in configurations:
            print(f"\nEvaluating configuration: {config['name']}")
            metrics = self.evaluate_model_performance(config)
            
            # Store performance metrics
            self.model_performance[config['name']] = metrics
            
            # Print key metrics
            print(f"  Train quality: {metrics['train_quality_avg']:.4f}")
            print(f"  Test quality: {metrics['test_quality_avg']:.4f}")
            print(f"  Quality difference: {metrics['train_test_quality_diff']:.4f}")
            print(f"  Overall score: {metrics['overall_score']:.4f}")
            
            # Update best configuration
            if metrics['overall_score'] > best_score:
                best_score = metrics['overall_score']
                best_config = config
        
        # Store best configuration
        if best_config:
            print(f"\n🏆 Best configuration: {best_config['name']} (score: {best_score:.4f})")
            self.best_model_config = best_config
            
        # Visualize performance comparison
        self.visualize_model_performance()
        
    def visualize_model_performance(self) -> None:
        """Visualize performance comparison between different model configurations"""
        if not MATPLOTLIB_AVAILABLE or not self.model_performance:
            return
            
        try:
            # Create figure with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Model Configuration Performance Comparison', fontsize=16)
            
            # Extract data for plotting
            config_names = list(self.model_performance.keys())
            train_quality = [self.model_performance[name]['train_quality_avg'] for name in config_names]
            test_quality = [self.model_performance[name]['test_quality_avg'] for name in config_names]
            quality_diff = [self.model_performance[name]['train_test_quality_diff'] for name in config_names]
            overall_scores = [self.model_performance[name]['overall_score'] for name in config_names]
            
            # Plot train vs test quality
            axs[0, 0].bar(config_names, train_quality, label='Train Quality', alpha=0.7, color='blue')
            axs[0, 0].bar(config_names, test_quality, label='Test Quality', alpha=0.7, color='green')
            axs[0, 0].set_title('Train vs Test Quality')
            axs[0, 0].set_ylabel('Quality Score')
            axs[0, 0].legend()
            axs[0, 0].set_xticklabels(config_names, rotation=45, ha='right')
            
            # Plot quality difference
            axs[0, 1].bar(config_names, quality_diff, color='orange')
            axs[0, 1].set_title('Train/Test Quality Difference')
            axs[0, 1].set_ylabel('Absolute Difference')
            axs[0, 1].set_xticklabels(config_names, rotation=45, ha='right')
            
            # Plot overall scores
            axs[1, 0].bar(config_names, overall_scores, color='purple')
            axs[1, 0].set_title('Overall Performance Score')
            axs[1, 0].set_ylabel('Score')
            axs[1, 0].set_xticklabels(config_names, rotation=45, ha='right')
            
            # Highlight best configuration
            best_config = max(config_names, key=lambda x: self.model_performance[x]['overall_score'])
            best_idx = config_names.index(best_config)
            
            # Create a summary table
            axs[1, 1].axis('tight')
            axs[1, 1].axis('off')
            table_data = [
                ['Configuration', 'Train Quality', 'Test Quality', 'Difference', 'Score'],
            ]
            
            for name in config_names:
                metrics = self.model_performance[name]
                row = [
                    name,
                    f"{metrics['train_quality_avg']:.4f}",
                    f"{metrics['test_quality_avg']:.4f}",
                    f"{metrics['train_test_quality_diff']:.4f}",
                    f"{metrics['overall_score']:.4f}"
                ]
                table_data.append(row)
                
            table = axs[1, 1].table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Highlight best configuration in the table
            for i in range(len(table_data[0])):
                table[(best_idx + 1, i)].set_facecolor('#90EE90')  # Light green
                
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save figure
            plt.savefig('model_performance_comparison.png')
            print("\n📊 Performance visualization saved as 'model_performance_comparison.png'")
            
            # Close figure to free memory
            plt.close(fig)
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
        
    def evaluate_model_performance(self, model_config: Dict) -> Dict:
        """Evaluate model performance using cross-validation
        
        Args:
            model_config: Dictionary containing model configuration parameters
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"📊 Evaluating model configuration: {model_config}")
        
        # Store original parameters
        original_params = {
            'train_ratio': self.train_ratio,
            'test_ratio': self.test_ratio,
            'similarity_threshold': self.similarity_threshold,
            'hash_threshold': self.hash_threshold,
            'remove_exact_duplicates': self.remove_exact_duplicates,
            'keep_near_duplicates': self.keep_near_duplicates,
            'rotation_similarity_threshold': self.rotation_similarity_threshold,
            'foreground_similarity_threshold': self.foreground_similarity_threshold,
            'exact_duplicate_threshold': getattr(self, 'exact_duplicate_threshold', 0.99),
            'near_duplicate_threshold': getattr(self, 'near_duplicate_threshold', 0.95),
            'variation_threshold': getattr(self, 'variation_threshold', 0.85),
            'semantic_threshold': getattr(self, 'semantic_threshold', 0.80),
            'enable_content_filtering': getattr(self, 'enable_content_filtering', False),
            'complexity_threshold': getattr(self, 'complexity_threshold', 0.6)
        }
        
        # Apply model configuration - skip 'name' parameter
        for param, value in model_config.items():
            if param != 'name' and hasattr(self, param):
                setattr(self, param, value)
                print(f"  Setting {param} = {value}")
        
        # Prepare for cross-validation
        if not self.unique_images:
            raise ValueError("No unique images found. Run create_unique_dataset() first.")
            
        n_samples = len(self.unique_images)
        fold_size = n_samples // self.cross_validation_folds
        
        # Metrics to track
        metrics = {
            'train_quality_avg': [],
            'test_quality_avg': [],
            'train_test_quality_diff': [],
            'duplicate_detection_rate': [],
            'cross_split_duplicates': []
        }
        
        # Perform cross-validation
        for fold in range(self.cross_validation_folds):
            print(f"Fold {fold+1}/{self.cross_validation_folds}")
            
            # Create fold indices
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.cross_validation_folds - 1 else n_samples
            
            # Create train/test splits for this fold
            test_indices = list(range(test_start, test_end))
            train_indices = [i for i in range(n_samples) if i not in test_indices]
            
            # Create splits
            splits = {
                'train': [self.unique_images[i] for i in train_indices],
                'test': [self.unique_images[i] for i in test_indices]
            }
            
            # Calculate metrics
            train_quality = np.mean([img['quality_score'] for img in splits['train']])
            test_quality = np.mean([img['quality_score'] for img in splits['test']])
            quality_diff = abs(train_quality - test_quality)
            
            # Check for duplicates across splits
            train_hashes = set(img['file_hash'] for img in splits['train'] if img['file_hash'])
            test_hashes = set(img['file_hash'] for img in splits['test'] if img['file_hash'])
            cross_split_duplicates = len(train_hashes.intersection(test_hashes))
            
            # Calculate duplicate detection rate (higher is better)
            duplicate_groups_found = len(self.duplicate_groups) + len(self.exact_duplicates)
            total_images = len(self.image_data)
            duplicate_detection_rate = duplicate_groups_found / total_images if total_images > 0 else 0
            
            # Store metrics
            metrics['train_quality_avg'].append(train_quality)
            metrics['test_quality_avg'].append(test_quality)
            metrics['train_test_quality_diff'].append(quality_diff)
            metrics['duplicate_detection_rate'].append(duplicate_detection_rate)
            metrics['cross_split_duplicates'].append(cross_split_duplicates)
        
        # Calculate average metrics across folds
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[key] = np.mean(values)
            
        # Calculate overall score (higher is better)
        # We want: high quality, small train/test difference, high duplicate detection, low cross-split duplicates
        avg_train_quality = avg_metrics['train_quality_avg']
        avg_test_quality = avg_metrics['test_quality_avg']
        avg_quality = (avg_train_quality + avg_test_quality) / 2
        quality_diff = avg_metrics['train_test_quality_diff']
        duplicate_rate = avg_metrics['duplicate_detection_rate']
        cross_duplicates = avg_metrics['cross_split_duplicates']
        
        # Weighted score calculation
        overall_score = (
            avg_quality * 0.4 +                      # Higher quality is better
            (1.0 - quality_diff) * 0.3 +            # Lower difference is better
            duplicate_rate * 0.2 +                  # Higher duplicate detection is better
            (1.0 - cross_duplicates / 10.0) * 0.1   # Lower cross-split duplicates is better
        )
        
        avg_metrics['overall_score'] = overall_score
        
        # Restore original parameters
        for param, value in original_params.items():
            setattr(self, param, value)
            
        return avg_metrics

    def run_complete_pipeline(self, input_folder: str, output_folder: str, evaluate_models: bool = False) -> None:
        """Run the complete processing pipeline"""
        print("🚀 Starting Smart Image Dataset Splitting Pipeline")
        print("="*60)

        try:
            # Step 1: Analyze images
            self.analyze_images(input_folder)

            if not self.image_data:
                print("❌ No valid images found!")
                return

            # Step 2: Detect duplicates
            self.detect_duplicates_advanced()

            # Step 3: Create unique dataset
            self.create_unique_dataset()

            if not self.unique_images:
                print("❌ No unique images remain after duplicate processing!")
                return
                
            # Step 3.5: Evaluate different model configurations if requested
            if evaluate_models:
                self.evaluate_model_configurations()
                # Apply best model configuration if found
                if self.best_model_config:
                    print("\n🔄 Applying best model configuration...")
                    for param, value in self.best_model_config.items():
                        if hasattr(self, param):
                            setattr(self, param, value)

            # Step 4: Smart split
            splits = self.smart_split_with_duplicate_handling()

            # Step 5: Copy files
            self.copy_files(splits, output_folder)

            print("\n🎉 Dataset splitting completed successfully!")
            print("✅ No duplicates will exist across train/test splits!")
            
            # Print best model configuration if available
            if self.best_model_config:
                print("\n🏆 Best model configuration:")
                for param, value in self.best_model_config.items():
                    print(f"  - {param}: {value}")

        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Example usage with enhanced configuration"""

    # Configuration - UPDATE THESE PATHS
    input_folder = r"C:\Users\user\Desktop\ok"  # Your input folder
    output_folder = r"C:\Users\user\Desktop\ko"  # Your output folder
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Smart Image Dataset Splitter')
    parser.add_argument('--input', type=str, help='Input folder path', default=input_folder)
    parser.add_argument('--output', type=str, help='Output folder path', default=output_folder)
    parser.add_argument('--train-ratio', type=float, help='Training set ratio', default=0.75)
    parser.add_argument('--evaluate', action='store_true', help='Evaluate different model configurations')
    parser.add_argument('--cross-validation-folds', type=int, help='Number of cross-validation folds', default=5)
    parser.add_argument('--similarity-threshold', type=float, help='Semantic similarity threshold (0-1)', default=0.87)
    parser.add_argument('--hash-threshold', type=int, help='Perceptual hash difference threshold', default=5)
    parser.add_argument('--keep-duplicates', action='store_true', help='Keep near-duplicates in dataset')
    parser.add_argument('--rotation-threshold', type=float, help='Rotation similarity threshold', default=0.82)
    parser.add_argument('--foreground-threshold', type=float, help='Foreground similarity threshold', default=0.85)
    args = parser.parse_args()

    # Initialize enhanced splitter
    splitter = SmartImageSplitter(
        train_ratio=args.train_ratio,                      # Training ratio
        test_ratio=1.0 - args.train_ratio,                 # Testing ratio
        similarity_threshold=args.similarity_threshold,    # Semantic similarity threshold (0-1)
        hash_threshold=args.hash_threshold,                # Perceptual hash difference threshold
        remove_exact_duplicates=False,                      # Remove exact file duplicates (keep best quality)
        keep_near_duplicates=args.keep_duplicates,         # Keep near-duplicates but ensure they don't split across train/test
        rotation_similarity_threshold=args.rotation_threshold,       # Threshold for detecting rotated variants
        foreground_similarity_threshold=args.foreground_threshold,   # Threshold for detecting same orchid with different backgrounds
        cross_validation_folds=args.cross_validation_folds           # Number of cross-validation folds
    )

    # Run complete pipeline with model evaluation if requested
    splitter.run_complete_pipeline(args.input, args.output, evaluate_models=args.evaluate)


if __name__ == "__main__":
    # Run main configuration
    main()