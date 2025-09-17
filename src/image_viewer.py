from PyQt5 import QtWidgets, QtGui, QtCore
import os
from .image_utils import ImageLoader


class ImageViewer(QtWidgets.QLabel):
    """Enhanced image viewer with zoom, pan, and better scaling for the new layout.

    Features:
    - Click to toggle between fit-to-widget and actual size
    - Mouse wheel zoom
    - Better scaling for different image sizes
    - Improved visual feedback
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setText("🖼️ Select an image to preview")
        self.setStyleSheet("QLabel { color: #666; font-size: 14px; }")
        self._pixmap = None
        self._scale = 1.0
        self._fit_mode = True

    def set_image(self, path):
        self._scale = 1.0
        self._fit_mode = True
        if path and os.path.exists(path):
            pix = ImageLoader.load_pixmap(path)
            if not pix.isNull():
                self._pixmap = pix
                self._apply_pixmap()
                # Show image info in tooltip
                self.setToolTip(f"Image: {os.path.basename(path)}\nSize: {pix.width()}x{pix.height()}\nClick to zoom, wheel to scale")
                return
        # fallback
        self._pixmap = None
        self.setText("🖼️ Select an image to preview")
        self.setToolTip("")
        self.setPixmap(QtGui.QPixmap())

    def _apply_pixmap(self):
        if not self._pixmap:
            return
            
        if self._fit_mode:
            # Fit to widget size while maintaining aspect ratio
            widget_size = self.size()
            scaled = self._pixmap.scaled(
                widget_size.width() - 20,  # Leave some margin
                widget_size.height() - 20,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        else:
            # Use manual scale
            w = int(self._pixmap.width() * self._scale)
            h = int(self._pixmap.height() * self._scale)
            if w <= 0 or h <= 0:
                return
            scaled = self._pixmap.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            
        self.setPixmap(scaled)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton and self._pixmap:
            # Toggle between fit-to-widget and actual size modes
            self._fit_mode = not self._fit_mode
            if not self._fit_mode:
                self._scale = 1.0  # Start at actual size when switching to manual mode
            self._apply_pixmap()
            
            # Update tooltip to show current mode
            mode_text = "fit-to-window" if self._fit_mode else "manual zoom"
            current_tooltip = self.toolTip()
            if "Mode:" in current_tooltip:
                lines = current_tooltip.split('\n')
                lines = [line for line in lines if not line.startswith("Mode:")]
                current_tooltip = '\n'.join(lines)
            self.setToolTip(f"{current_tooltip}\nMode: {mode_text}")

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if not self._pixmap:
            return
            
        # Switch to manual mode when using wheel
        if self._fit_mode:
            self._fit_mode = False
            self._scale = 1.0
            
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 0.85
        self._scale = max(0.1, min(10.0, self._scale * factor))
        self._apply_pixmap()
        
        # Update tooltip with current scale
        current_tooltip = self.toolTip()
        if "Scale:" in current_tooltip:
            lines = current_tooltip.split('\n')
            lines = [line for line in lines if not line.startswith("Scale:")]
            current_tooltip = '\n'.join(lines)
        self.setToolTip(f"{current_tooltip}\nScale: {self._scale:.1f}x")
    
    def resizeEvent(self, event):
        """Handle widget resize to maintain fit-to-widget mode."""
        super().resizeEvent(event)
        if self._pixmap and self._fit_mode:
            self._apply_pixmap()


