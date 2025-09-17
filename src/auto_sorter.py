"""Auto-sorter integration for Splintez dataset browser"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PyQt5 import QtCore, QtWidgets

# Try to import ML dependencies with graceful fallbacks
try:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from sklearn.metrics.pairwise import cosine_similarity
    from skimage.metrics import structural_similarity as ssim
    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML dependencies not available: {e}")
    ML_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class DummyArray:
        def __init__(self, *args, **kwargs):
            pass
    np = type('numpy', (), {'array': DummyArray, 'mean': lambda x: 0, 'std': lambda x: 0})()

# Try different CLIP implementations
CLIP_AVAILABLE = False
if ML_AVAILABLE:
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


class AutoSortWorker(QtCore.QThread):
    """Worker thread for auto-sorting to prevent UI blocking"""
    
    progress_updated = QtCore.pyqtSignal(int)  # progress percentage
    status_updated = QtCore.pyqtSignal(str)  # status message
    finished = QtCore.pyqtSignal(dict)  # results dictionary
    error_occurred = QtCore.pyqtSignal(str)  # error message
    
    def __init__(self, train_folder, test_folder, labels, settings, test_mode=False, test_label=None):
        super().__init__()
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.labels = labels
        self.settings = settings
        self.test_mode = test_mode
        self.test_label = test_label
        self._is_cancelled = False
        
    def cancel(self):
        """Cancel the sorting operation"""
        self._is_cancelled = True
        
    def run(self):
        """Run the auto-sorting process"""
        try:
            import time
            start_time = time.time()
            
            if not ML_AVAILABLE:
                self.error_occurred.emit("ML dependencies not available. Please install: torch, torchvision, opencv-python, pillow, numpy, scikit-learn, scikit-image")
                return
            
            # Determine which labels to process
            labels_to_process = [self.test_label] if self.test_mode and self.test_label else self.labels
            
            total_labels = len(labels_to_process)
            
            results = {
                'processed_labels': [],
                'labels_processed': 0,
                'total_images': 0,
                'duplicates_found': 0,
                'moved_to_test': 0,
                'train_images': 0,
                'test_images': 0,
                'processing_time': 'N/A',
                'details': ''
            }
            
            for i, label in enumerate(labels_to_process):
                if self._is_cancelled:
                    self.error_occurred.emit("Operation cancelled")
                    return
                    
                self.progress_updated.emit(int((i / total_labels) * 100))
                self.status_updated.emit(f"Processing label: {label}")
                
                # Process this label
                label_results = self.process_label(label)
                
                if label_results:
                    results['processed_labels'].append(label)
                    results['labels_processed'] += 1
                    results['total_images'] += label_results.get('total_images', 0)
                    results['duplicates_found'] += label_results.get('duplicates_removed', 0)
                    results['moved_to_test'] += label_results.get('test_images', 0)
                    results['train_images'] += label_results.get('train_images', 0)
                    results['test_images'] += label_results.get('test_images', 0)
                    
                    # Add to details
                    details_line = f"Label '{label}': {label_results.get('total_images', 0)} images processed, {label_results.get('duplicates_removed', 0)} duplicates removed\n"
                    results['details'] += details_line
                    
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            results['processing_time'] = f"{processing_time:.1f} seconds"
            
            # Add summary to details
            if results['details']:
                results['details'] += f"\n--- Summary ---\n"
                results['details'] += f"Total labels processed: {results['labels_processed']}\n"
                results['details'] += f"Total images processed: {results['total_images']}\n"
                results['details'] += f"Total duplicates found: {results['duplicates_found']}\n"
                results['details'] += f"Processing completed in {results['processing_time']}"
            
            self.progress_updated.emit(100)
            self.status_updated.emit("Auto-sorting completed successfully!")
            self.finished.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"Error during auto-sorting: {str(e)}")
            
    def process_label(self, label):
        """Process a single label folder - collect from both train and test, then re-split"""
        try:
            # Get train and test folders for this label
            train_label_folder = os.path.join(self.train_folder, label)
            test_label_folder = os.path.join(self.test_folder, label)
            
            # Check if either folder exists and has images
            train_exists = os.path.exists(train_label_folder)
            test_exists = os.path.exists(test_label_folder)
            
            if not train_exists and not test_exists:
                return None
                
            # Initialize the proper ML splitter
            # sys.path.append(str(Path(__file__).parent.parent / "ML"))
            # from sorter import SmartImageSplitter
            from ML.sorter import SmartImageSplitter
            
            # Get config file path
            config_file = str(Path(__file__).parent.parent / "auto_sort_settings.json")
            
            splitter = SmartImageSplitter(
                train_ratio=self.settings['train_ratio'],
                test_ratio=1.0 - self.settings['train_ratio'],
                # Map old parameter names to new DINOv2-optimized thresholds
                exact_duplicate_threshold=0.99,  # Always strict for exact duplicates
                near_duplicate_threshold=self.settings['similarity_threshold'],  # Use similarity_threshold for near duplicates
                variation_threshold=self.settings['rotation_similarity_threshold'],  # Use rotation threshold for variations
                semantic_threshold=self.settings['foreground_similarity_threshold'],  # Use foreground threshold for semantic
                hash_threshold=self.settings['hash_threshold'],
                remove_exact_duplicates=self.settings['remove_exact_duplicates'],
                keep_near_duplicates=self.settings['keep_near_duplicates'],
                cross_validation_folds=self.settings['cross_validation_folds'],
                # Content-based filtering parameters
                enable_content_filtering=self.settings.get('enable_content_filtering', False),
                detailed_to_train_ratio=self.settings.get('content_balance_ratio', 0.7),
                simple_to_test_ratio=0.6,  # Default value
                complexity_threshold=self.settings.get('complexity_threshold', 0.6),
                config_file=config_file  # Pass config file for DINOv2 settings
            )
            
            # Collect all image paths from both folders
            image_paths = []
            
            # Get images from train folder
            if os.path.exists(train_label_folder):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']:
                    image_paths.extend(Path(train_label_folder).glob(ext))
                    image_paths.extend(Path(train_label_folder).glob(ext.upper()))
            
            # Get images from test folder
            if os.path.exists(test_label_folder):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']:
                    image_paths.extend(Path(test_label_folder).glob(ext))
                    image_paths.extend(Path(test_label_folder).glob(ext.upper()))
            
            if not image_paths:
                return None
                
            # Convert to strings
            image_paths = [str(p) for p in image_paths]
            
            # Create a temporary combined folder for processing
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy all images to temp directory with unique names to avoid conflicts
                copied_files = {}  # Track original path -> temp path mapping
                for img_path in image_paths:
                    filename = os.path.basename(img_path)
                    temp_path = os.path.join(temp_dir, filename)
                    
                    # Handle naming conflicts by adding counter
                    counter = 1
                    original_temp_path = temp_path
                    while os.path.exists(temp_path):
                        name, ext = os.path.splitext(filename)
                        temp_path = os.path.join(temp_dir, f"{name}_temp{counter}{ext}")
                        counter += 1
                    
                    shutil.copy2(img_path, temp_path)
                    copied_files[img_path] = temp_path
                
                # Process with the ML splitter
                try:
                    splitter.run_complete_pipeline(
                        temp_dir,
                        temp_dir,  # Output to same temp dir
                        evaluate_models=self.settings.get('evaluate_models', False)
                    )
                    
                    # Now reorganize the results back to train/test folders
                    train_output = os.path.join(temp_dir, 'train')
                    test_output = os.path.join(temp_dir, 'test')
                    
                    if os.path.exists(train_output) and os.path.exists(test_output):
                        # Clear original folders and move results with original names
                        self.reorganize_from_ml_output(train_output, test_output, train_label_folder, test_label_folder, copied_files)
                        
                        # Calculate results
                        train_count = len([f for f in os.listdir(train_label_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]) if os.path.exists(train_label_folder) else 0
                        test_count = len([f for f in os.listdir(test_label_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]) if os.path.exists(test_label_folder) else 0
                        
                        return {
                            'total_images': len(image_paths),
                            'duplicates_removed': len(splitter.exact_duplicates) + len(splitter.duplicate_groups) if hasattr(splitter, 'exact_duplicates') else 0,
                            'train_images': train_count,
                            'test_images': test_count
                        }
                    else:
                        return None
                        
                except Exception as e:
                    print(f"ML processing error: {e}")
                    return None
                
        except Exception as e:
            return None
    
    def reorganize_from_ml_output(self, train_output, test_output, train_label_folder, test_label_folder, copied_files=None):
        """Move ML-processed results back to original train/test folders with original names"""
        import tempfile
        
        # Create reverse mapping from temp filename to original path
        temp_to_original = {}
        if copied_files:
            for original_path, temp_path in copied_files.items():
                temp_filename = os.path.basename(temp_path)
                temp_to_original[temp_filename] = original_path
        
        # Ensure target folders exist
        os.makedirs(train_label_folder, exist_ok=True)
        os.makedirs(test_label_folder, exist_ok=True)
        
        # Create temporary backup of existing files
        with tempfile.TemporaryDirectory() as backup_dir:
            # Backup existing files
            for folder in [train_label_folder, test_label_folder]:
                if os.path.exists(folder):
                    for file in os.listdir(folder):
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            backup_path = os.path.join(backup_dir, file)
                            shutil.move(file_path, backup_path)
            
            # Move ML results to target folders with original names
            if os.path.exists(train_output):
                for file in os.listdir(train_output):
                    src_path = os.path.join(train_output, file)
                    if os.path.isfile(src_path):
                        # Get original filename if available
                        if file in temp_to_original:
                            original_filename = os.path.basename(temp_to_original[file])
                        else:
                            original_filename = file
                        
                        dst_path = os.path.join(train_label_folder, original_filename)
                        shutil.move(src_path, dst_path)
            
            if os.path.exists(test_output):
                for file in os.listdir(test_output):
                    src_path = os.path.join(test_output, file)
                    if os.path.isfile(src_path):
                        # Get original filename if available
                        if file in temp_to_original:
                            original_filename = os.path.basename(temp_to_original[file])
                        else:
                            original_filename = file
                        
                        dst_path = os.path.join(test_label_folder, original_filename)
                        shutil.move(src_path, dst_path)


# The SplintezImageSplitter class has been removed.
# We now use the full SmartImageSplitter from ML/sorter.py for better results.
        """Process images from both train and test folders, then re-split them"""
        try:
            if not ML_AVAILABLE:
                return False
                
            # Collect all images from both train and test folders
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            image_paths = []
            
            # Collect from train folder
            if os.path.exists(train_label_folder):
                print(f"DEBUG: Train folder exists: {train_label_folder}")
                for ext in image_extensions:
                    found_images = list(Path(train_label_folder).glob(f'*{ext}'))
                    print(f"DEBUG: Found {len(found_images)} images with extension {ext} in train folder")
                    image_paths.extend(found_images)
            else:
                print(f"DEBUG: Train folder does not exist: {train_label_folder}")
                    
            # Collect from test folder
            if os.path.exists(test_label_folder):
                print(f"DEBUG: Test folder exists: {test_label_folder}")
                for ext in image_extensions:
                    found_images = list(Path(test_label_folder).glob(f'*{ext}'))
                    print(f"DEBUG: Found {len(found_images)} images with extension {ext} in test folder")
                    image_paths.extend(found_images)
            else:
                print(f"DEBUG: Test folder does not exist: {test_label_folder}")
                
            print(f"DEBUG: Total image paths found: {len(image_paths)}")
            if not image_paths:
                print(f"No images found in {train_label_folder} or {test_label_folder}")
                return True  # Not an error, just empty folders
                
            print(f"Processing {len(image_paths)} images from both train and test folders")
            
            # Analyze images
            print(f"DEBUG: Starting analyze_images_from_paths")
            self.analyze_images_from_paths(image_paths)
            print(f"DEBUG: After analysis, image_data length: {len(self.image_data)}")
            
            if not self.image_data:
                print("No valid images found after analysis")
                return True
                
            # Detect duplicates
            print(f"DEBUG: Starting detect_duplicates_advanced")
            self.detect_duplicates_advanced()
            print(f"DEBUG: After duplicate detection")
            
            # Create unique dataset
            self.create_unique_dataset()
            
            if not self.unique_images:
                return True
                
            # Split the images
            splits = self.smart_split_with_duplicate_handling()
            
            # Move files to appropriate folders
            self.reorganize_files(splits, train_label_folder, test_label_folder)
            
            return True
            
        except Exception as e:
            return False
            
    def process_label_folder(self, train_label_folder, test_label_folder, evaluate_models=False):
        """Legacy method - redirects to combined processing"""
        return self.process_combined_label_folders(train_label_folder, test_label_folder, evaluate_models)
            
    # analyze_images_from_paths method removed - using SmartImageSplitter from sorter.py
                
    # detect_duplicates_advanced method removed - using SmartImageSplitter from sorter.py
        
    # create_unique_dataset method removed - using SmartImageSplitter from sorter.py
                
    # smart_split_with_duplicate_handling method removed - using SmartImageSplitter from sorter.py
        
    # reorganize_files method removed - using reorganize_from_ml_output for ML pipeline