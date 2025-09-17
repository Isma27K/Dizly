from PyQt5 import QtGui, QtCore
import os
import sys
from contextlib import contextmanager
from io import StringIO


class ImageLoader:
    """Utility class for loading images with proper error handling and ICC profile warning suppression."""
    
    @staticmethod
    @contextmanager
    def suppress_qt_warnings():
        """Context manager to suppress Qt ICC profile warnings."""
        # Store original stderr
        original_stderr = sys.stderr
        
        # Create a custom stderr that filters out ICC profile warnings
        class FilteredStderr:
            def __init__(self, original):
                self.original = original
                self.buffer = StringIO()
                
            def write(self, text):
                # Filter out Qt ICC profile warnings
                if 'qt.gui.icc' not in text.lower() and 'fromiccprofile' not in text.lower():
                    self.original.write(text)
                    
            def flush(self):
                self.original.flush()
                
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        try:
            # Replace stderr with filtered version
            sys.stderr = FilteredStderr(original_stderr)
            yield
        finally:
            # Restore original stderr
            sys.stderr = original_stderr
    
    @staticmethod
    def load_pixmap(image_path: str) -> QtGui.QPixmap:
        """Load a QPixmap with ICC profile warning suppression.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            QPixmap object, or null QPixmap if loading fails
        """
        if not image_path or not os.path.exists(image_path):
            return QtGui.QPixmap()
            
        try:
            with ImageLoader.suppress_qt_warnings():
                pixmap = QtGui.QPixmap(image_path)
                return pixmap
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            return QtGui.QPixmap()
    
    @staticmethod
    def load_scaled_pixmap(image_path: str, width: int, height: int, 
                          aspect_ratio_mode=QtCore.Qt.KeepAspectRatio,
                          transformation_mode=QtCore.Qt.SmoothTransformation) -> QtGui.QPixmap:
        """Load and scale a QPixmap with ICC profile warning suppression.
        
        Args:
            image_path: Path to the image file
            width: Target width
            height: Target height
            aspect_ratio_mode: Qt aspect ratio mode
            transformation_mode: Qt transformation mode
            
        Returns:
            Scaled QPixmap object, or null QPixmap if loading fails
        """
        pixmap = ImageLoader.load_pixmap(image_path)
        if not pixmap.isNull():
            return pixmap.scaled(width, height, aspect_ratio_mode, transformation_mode)
        return QtGui.QPixmap()
    
    @staticmethod
    def is_valid_image(image_path: str) -> bool:
        """Check if an image file can be loaded without warnings.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image can be loaded successfully, False otherwise
        """
        if not image_path or not os.path.exists(image_path):
            return False
            
        try:
            with ImageLoader.suppress_qt_warnings():
                pixmap = QtGui.QPixmap(image_path)
                return not pixmap.isNull()
        except Exception:
            return False