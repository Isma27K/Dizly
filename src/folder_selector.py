
from PyQt5 import QtWidgets, QtGui, QtCore
import os
import shutil
from pathlib import Path
from .image_viewer import ImageViewer
from .highlighter_settings import HighlighterSettings, HighlighterSettingsWidget
from .auto_sort_dialog import AutoSortSettingsDialog
from .auto_sorter import AutoSortWorker
from .auto_sort_progress import AutoSortProgressDialog, AutoSortResultDialog


class ImageGridWidget(QtWidgets.QWidget):
    """Grid widget for displaying images with drag-and-drop and zoom functionality."""
    
    def __init__(self, list_type, parent=None):
        super().__init__(parent)
        self.list_type = list_type  # 'train' or 'test'
        self.main_widget = None
        self.grid_size = 120  # Default thumbnail size
        self.images = []  # List of image paths
        
        # Zoom performance optimization
        self.zoom_timer = QtCore.QTimer()
        self.zoom_timer.setSingleShot(True)
        self.zoom_timer.timeout.connect(self._apply_zoom_resize)
        self.pending_zoom_size = None
        
        # Setup layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Create scroll area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # Create grid container
        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(5)
        
        self.scroll_area.setWidget(self.grid_container)
        self.layout.addWidget(self.scroll_area)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
    def set_main_widget(self, main_widget):
        """Set reference to the main widget."""
        self.main_widget = main_widget
        
    def wheelEvent(self, event):
        """Handle Ctrl+scroll for zooming grid with debouncing for better performance."""
        if event.modifiers() & QtCore.Qt.ControlModifier:
            delta = event.angleDelta().y()
            # Improved sensitivity - smaller increments, more responsive
            zoom_step = 8 if abs(delta) >= 120 else 4
            
            if delta > 0:
                new_size = min(400, self.grid_size + zoom_step)
            else:
                new_size = max(60, self.grid_size - zoom_step)
            
            # Only update if size actually changed
            if new_size != self.grid_size:
                self.pending_zoom_size = new_size
                # Debounce zoom operations - wait 100ms before applying
                self.zoom_timer.start(100)
            
            event.accept()
        else:
            super().wheelEvent(event)
            
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
            
    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        if event.mimeData().hasText() and self.main_widget:
            file_path = event.mimeData().text()
            # Extract source type from the data
            source_type = "train" if "[train]" in file_path else "test"
            
            # Clean the file path
            clean_path = file_path.replace("[train] ", "").replace("[test] ", "")
            
            # Only move if dropping to different section
            if source_type != self.list_type:
                # Immediate visual feedback - remove from source grid instantly
                source_grid = self.main_widget.train_grid if source_type == "train" else self.main_widget.test_grid
                source_grid.remove_image_instantly(clean_path)
                
                # Add to target grid instantly
                self.add_image_instantly(clean_path)
                
                # Perform actual file move in background
                self.main_widget.move_image_async(clean_path, self.list_type)
                event.acceptProposedAction()
            else:
                # Same section - just ignore, don't remove image
                event.ignore()
        else:
            event.ignore()
            
    def set_images(self, image_paths):
        """Set the list of images to display."""
        self.images = image_paths
        self.refresh_grid()
        
    def refresh_grid(self):
        """Refresh the grid layout with current images in sorted order."""
        # Sort images by filename
        self.images.sort(key=lambda x: os.path.basename(x).lower())
        
        # Clear existing items
        for i in reversed(range(self.grid_layout.count())):
            child = self.grid_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
                
        # Calculate columns based on container width
        container_width = self.scroll_area.viewport().width() - 20
        cols = max(1, container_width // (self.grid_size + 10))
        
        # Add images to grid in sorted order
        for i, image_path in enumerate(self.images):
            row = i // cols
            col = i % cols
            
            # Create image thumbnail
            thumbnail = self.create_thumbnail(image_path)
            self.grid_layout.addWidget(thumbnail, row, col)
            
    def remove_image_instantly(self, image_path):
        """Remove image from grid instantly for immediate visual feedback."""
        # Remove from images list
        if image_path in self.images:
            self.images.remove(image_path)
        
        # Find and remove the thumbnail widget
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'image_path') and widget.image_path == image_path:
                    widget.setParent(None)
                    break
                    
        # Reorganize remaining thumbnails
        self._reorganize_grid()
        
    def add_image_instantly(self, image_path):
        """Add image to grid instantly for immediate visual feedback without rebuilding."""
        # Calculate target path but use original path for thumbnail creation
        source_path = Path(image_path)
        if self.list_type == "train":
            # Moving to train folder
            target_path = str(self.main_widget.train_root / self.main_widget.current_label / source_path.name)
        else:
            # Moving to test folder  
            target_path = str(self.main_widget.test_root / self.main_widget.current_label / source_path.name)
            
        if target_path not in self.images:
            # Simply add to images list without sorting or rebuilding
            self.images.append(target_path)
            
            # Add thumbnail at the end of the grid for immediate feedback
            container_width = self.scroll_area.viewport().width() - 20
            cols = max(1, container_width // (self.grid_size + 10))
            current_count = len(self.images) - 1  # Index of the new image
            row = current_count // cols
            col = current_count % cols
            
            # Create thumbnail with original path and highlight
            thumbnail = self.create_thumbnail(image_path)
            thumbnail.image_path = target_path
            thumbnail.set_highlighted(True)
            
            self.grid_layout.addWidget(thumbnail, row, col)
                
    def _clear_grid(self):
        """Clear all widgets from the grid layout."""
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
    def _apply_zoom_resize(self):
        """Apply the pending zoom resize operation."""
        if self.pending_zoom_size is not None:
            self.grid_size = self.pending_zoom_size
            self.pending_zoom_size = None
            self._resize_thumbnails()
    
    def _resize_thumbnails(self):
        """Efficiently resize existing thumbnails without rebuilding grid."""
        # Update thumbnail sizes for all existing widgets
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and item.widget():
                thumbnail = item.widget()
                if hasattr(thumbnail, 'thumbnail_size'):
                    # Update thumbnail properties
                    thumbnail.thumbnail_size = self.grid_size
                    thumbnail.setFixedSize(self.grid_size, self.grid_size + 4)
                    
                    # Update image label size
                    if hasattr(thumbnail, 'image_label'):
                        thumbnail.image_label.setFixedSize(
                            self.grid_size - 4, 
                            self.grid_size - 20
                        )
                    
                    # Reload image with new size
                    if hasattr(thumbnail, 'load_image'):
                        thumbnail.load_image()
        
        # Reorganize grid layout with new spacing
        self._reorganize_grid()
    
    def _reorganize_grid(self):
        """Reorganize grid layout after removing items."""
        # Get all remaining widgets
        widgets = []
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item and item.widget():
                widgets.append(item.widget())
                
        # Clear layout
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.itemAt(i)
            if item:
                self.grid_layout.removeItem(item)
                
        # Re-add widgets in proper positions
        container_width = self.scroll_area.viewport().width() - 20
        cols = max(1, container_width // (self.grid_size + 10))
        
        for i, widget in enumerate(widgets):
            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(widget, row, col)
            
    def create_thumbnail(self, image_path):
        """Create a thumbnail widget for an image."""
        thumbnail = ImageThumbnail(image_path, self.list_type, self.grid_size)
        thumbnail.setParent(self)  # Set proper parent for hierarchy traversal
        thumbnail.clicked.connect(lambda path=image_path: self.show_popup_preview(path))
        return thumbnail
        
    def show_popup_preview(self, image_path):
        """Show popup preview of the image."""
        if self.main_widget:
            self.main_widget.show_popup_preview(image_path)


class ImageThumbnail(QtWidgets.QWidget):
    """Thumbnail widget for individual images with drag support and filename display."""
    
    clicked = QtCore.pyqtSignal()
    
    def __init__(self, image_path, list_type, size=120):
        super().__init__()
        
        # Store metadata for drag operations and identification (FIRST)
        self.image_path = image_path
        self.list_type = list_type
        self.thumbnail_size = size
        self.is_highlighted = False
        
        # Create layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Create image label
        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(size - 4, size - 20)  # Leave space for filename
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Create filename label
        self.filename_label = QtWidgets.QLabel()
        self.filename_label.setFixedHeight(16)
        self.filename_label.setAlignment(QtCore.Qt.AlignCenter)
        self.filename_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #333;
                background: transparent;
                border: none;
            }
        """)
        
        # Add to layout
        layout.addWidget(self.image_label)
        layout.addWidget(self.filename_label)
        self.setLayout(layout)
        
        # Load and set image
        self.load_image()
        
        # Setup appearance
        self.setFixedSize(size, size + 4)  # Slightly taller for filename
        self.update_style()
        
        # Enable drag
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
    def load_image(self):
        """Load and scale the image for thumbnail display."""
        try:
            pixmap = QtGui.QPixmap(self.image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.thumbnail_size - 14, 
                    self.thumbnail_size - 30,  # More space for filename
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Error")
        except Exception:
            self.image_label.setText("Error")
            
        # Set filename with truncation
        filename = os.path.basename(self.image_path)
        if len(filename) > 15:
            display_name = filename[:12] + "..."
        else:
            display_name = filename
        self.filename_label.setText(display_name)
            
        # Set tooltip
        self.setToolTip(f"{filename}\nClick to preview\nDrag to move")
        
    def update_style(self):
        """Update widget styling based on highlight state."""
        if self.is_highlighted:
            style = """
                QWidget {
                    border: 3px solid #ff6b35;
                    border-radius: 8px;
                    background-color: #fff8f0;
                }
            """
        else:
            style = """
                QWidget {
                    border: 2px solid #ddd;
                    border-radius: 5px;
                    background-color: white;
                }
                QWidget:hover {
                    border-color: #0078d4;
                    background-color: #f0f8ff;
                }
            """
        self.setStyleSheet(style)
        
    def pixmap(self):
        """Return the current pixmap for drag operations."""
        return self.image_label.pixmap()
        
    def set_highlighted(self, highlighted=True):
        """Set highlight state with auto-removal after delay."""
        # Get main widget to access settings
        main_widget = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'highlighter_settings'):
                main_widget = parent
                break
            parent = parent.parent()
        
        if highlighted and main_widget:
            # Check if highlighting is enabled
            if not main_widget.highlighter_settings.enabled:
                return
            
            # Handle "Latest Only" mode - clear others first
            main_widget.highlighter_settings.clear_others_if_latest_only(self)
        
        # Update visual state
        self.is_highlighted = highlighted
        self.update_style()
        
        if highlighted and main_widget:
            # Add to settings manager's tracking
            main_widget.highlighter_settings.add_highlighted(self)
            
            # Auto-remove highlight after delay
            if hasattr(self, 'highlight_timer'):
                self.highlight_timer.stop()
            
            delay_ms = main_widget.highlighter_settings.get_delay_ms()
            self.highlight_timer = QtCore.QTimer()
            self.highlight_timer.setSingleShot(True)
            self.highlight_timer.timeout.connect(lambda: self._remove_highlight_with_cleanup())
            self.highlight_timer.start(delay_ms)
        else:
            # Remove from tracking and stop timer
            if main_widget:
                main_widget.highlighter_settings.remove_highlighted(self)
            if hasattr(self, 'highlight_timer'):
                self.highlight_timer.stop()
    
    def _remove_highlight_with_cleanup(self):
        """Remove highlight and clean up from settings manager."""
        self.is_highlighted = False
        self.update_style()
        
        # Remove from settings manager tracking
        main_widget = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'highlighter_settings'):
                main_widget = parent
                break
            parent = parent.parent()
        
        if main_widget:
            main_widget.highlighter_settings.remove_highlighted(self)
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_start_position = event.pos()
            
    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            # Check if we've moved far enough to start a drag
            if hasattr(self, 'drag_start_position'):
                distance = (event.pos() - self.drag_start_position).manhattanLength()
                if distance >= QtWidgets.QApplication.startDragDistance():
                    # Start drag operation
                    drag = QtGui.QDrag(self)
                    mimeData = QtCore.QMimeData()
                    # Include list type in the data for proper handling
                    mimeData.setText(f"[{self.list_type}] {self.image_path}")
                    drag.setMimeData(mimeData)
                    
                    # Set drag pixmap (with safety check)
                    current_pixmap = self.pixmap()
                    if current_pixmap and not current_pixmap.isNull():
                        drag.setPixmap(current_pixmap.scaled(64, 64, QtCore.Qt.KeepAspectRatio))
                        drag.setHotSpot(QtCore.QPoint(32, 32))
                    else:
                        # Create a default drag pixmap if image failed to load
                        default_pixmap = QtGui.QPixmap(64, 64)
                        default_pixmap.fill(QtCore.Qt.lightGray)
                        drag.setPixmap(default_pixmap)
                        drag.setHotSpot(QtCore.QPoint(32, 32))
                    
                    # Execute drag
                    result = drag.exec_(QtCore.Qt.MoveAction)
                    
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if hasattr(self, 'drag_start_position'):
                distance = (event.pos() - self.drag_start_position).manhattanLength()
                if distance < QtWidgets.QApplication.startDragDistance():
                    # This was a click, not a drag
                    self.clicked.emit()
                delattr(self, 'drag_start_position')


class PopupPreview(QtWidgets.QDialog):
    """Popup dialog for image preview."""
    
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Preview - {os.path.basename(image_path)}")
        self.setModal(False)
        self.resize(800, 600)
        
        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.set_image(image_path)
        layout.addWidget(self.image_viewer)
        
        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        # Center on parent
        if parent:
            self.move(parent.geometry().center() - self.rect().center())


class FolderSelectorWidget(QtWidgets.QWidget):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dizly")
        self.resize(1400, 900)
        
        # Initialize highlighter settings manager
        self.highlighter_settings = HighlighterSettings(self)
        
        # Initialize variables
        self.train_root = None
        self.test_root = None
        self.current_label = None
        self.popup_previews = []  # Track open popups
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create the main splitter
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # Left sidebar panel
        self.create_left_panel()
        
        # Right panel with train/test grids
        self.create_right_panel()
        
        # Add splitter to main layout (takes most space)
        main_layout.addWidget(self.main_splitter, 1)  # Stretch factor 1
        
        # Status label with minimal height
        status_container = QtWidgets.QWidget()
        status_container.setFixedHeight(30)  # Fixed minimal height
        status_layout = QtWidgets.QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addStretch()  # Push status to right
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { padding: 3px 8px; background-color: #f0f0f0; border-radius: 3px; font-size: 11px; }")
        self.status_label.setMaximumWidth(250)
        status_layout.addWidget(self.status_label)
        main_layout.addWidget(status_container, 0)  # No stretch - minimal space
        
    def create_left_panel(self):
        """Create the left sidebar with folder selection."""
        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setMinimumWidth(300)
        self.left_panel.setMaximumWidth(400)
        
        layout = QtWidgets.QVBoxLayout(self.left_panel)
        
        # Note: Toggle button moved to right panel so it's always visible
        
        # Folder selection
        folder_group = QtWidgets.QGroupBox("Dataset Folders")
        folder_layout = QtWidgets.QFormLayout(folder_group)
        
        # Train folder
        self.train_edit = QtWidgets.QLineEdit()
        self.train_edit.setReadOnly(True)
        train_btn = QtWidgets.QPushButton("Browse...")
        train_btn.clicked.connect(lambda: self.choose_folder(self.train_edit))
        train_layout = QtWidgets.QHBoxLayout()
        train_layout.addWidget(self.train_edit)
        train_layout.addWidget(train_btn)
        folder_layout.addRow("Train Folder:", train_layout)
        
        # Test folder
        self.test_edit = QtWidgets.QLineEdit()
        self.test_edit.setReadOnly(True)
        test_btn = QtWidgets.QPushButton("Browse...")
        test_btn.clicked.connect(lambda: self.choose_folder(self.test_edit))
        test_layout = QtWidgets.QHBoxLayout()
        test_layout.addWidget(self.test_edit)
        test_layout.addWidget(test_btn)
        folder_layout.addRow("Test Folder:", test_layout)
        
        # Scan button
        scan_btn = QtWidgets.QPushButton("Scan Folders")
        scan_btn.clicked.connect(self.scan_folders)
        folder_layout.addRow(scan_btn)
        
        layout.addWidget(folder_group)
        
        # Labels list
        labels_group = QtWidgets.QGroupBox("Labels")
        labels_layout = QtWidgets.QVBoxLayout(labels_group)
        
        self.labels_list = QtWidgets.QListWidget()
        self.labels_list.itemClicked.connect(self.show_label_details)
        labels_layout.addWidget(self.labels_list)
        
        # Auto-sort button
        auto_sort_btn = QtWidgets.QPushButton("🤖 Auto Sort")
        auto_sort_btn.clicked.connect(self.show_auto_sort_dialog)
        auto_sort_btn.setToolTip("Automatically sort images using AI")
        labels_layout.addWidget(auto_sort_btn)
        
        layout.addWidget(labels_group)
        
        # Highlighter settings (using modular widget)
        self.highlighter_widget = HighlighterSettingsWidget(self.highlighter_settings)
        layout.addWidget(self.highlighter_widget)
        
        self.main_splitter.addWidget(self.left_panel)
        
    def clear_all_highlights(self):
        """Clear all highlights from both train and test grids."""
        self.highlighter_settings.clear_all_highlighted()
        
    def create_right_panel(self):
        """Create the right panel with train/test grids."""
        self.right_panel = QtWidgets.QWidget()
        
        # Main layout for right panel
        right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        
        # Toggle button (always visible)
        self.toggle_btn = QtWidgets.QPushButton("◀ Hide Panel")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(False)  # Start in shown state
        self.toggle_btn.clicked.connect(self.toggle_left_panel)
        self.toggle_btn.setMaximumWidth(120)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        right_layout.addWidget(self.toggle_btn)
        
        # Create splitter for train/test sections
        self.train_test_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # Test section (now on left)
        test_widget = QtWidgets.QWidget()
        test_layout = QtWidgets.QVBoxLayout(test_widget)
        test_layout.setContentsMargins(5, 5, 5, 5)
        
        # Test header with refresh button
        test_header_widget = QtWidgets.QWidget()
        test_header_layout = QtWidgets.QHBoxLayout(test_header_widget)
        test_header_layout.setContentsMargins(0, 0, 0, 0)
        
        test_header = QtWidgets.QLabel("🧪 TEST")
        test_header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #388e3c;
                padding: 8px;
                background-color: #e8f5e8;
                border-radius: 4px;
            }
        """)
        test_header.setAlignment(QtCore.Qt.AlignCenter)
        
        test_refresh_btn = QtWidgets.QPushButton("🔄 Sort")
        test_refresh_btn.setMaximumWidth(60)
        test_refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        test_refresh_btn.clicked.connect(lambda: self.test_grid.refresh_grid())
        
        test_header_layout.addWidget(test_header)
        test_header_layout.addWidget(test_refresh_btn)
        test_layout.addWidget(test_header_widget)
        
        self.test_grid = ImageGridWidget("test")
        self.test_grid.set_main_widget(self)
        self.test_grid.setStyleSheet("QWidget { border: 2px dashed #4caf50; background-color: #f8fff8; }")
        test_layout.addWidget(self.test_grid)
        
        # Train section (now on right)
        train_widget = QtWidgets.QWidget()
        train_layout = QtWidgets.QVBoxLayout(train_widget)
        train_layout.setContentsMargins(5, 5, 5, 5)
        
        # Train header with refresh button
        train_header_widget = QtWidgets.QWidget()
        train_header_layout = QtWidgets.QHBoxLayout(train_header_widget)
        train_header_layout.setContentsMargins(0, 0, 0, 0)
        
        train_header = QtWidgets.QLabel("🚂 TRAIN")
        train_header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #1976d2;
                padding: 8px;
                background-color: #e3f2fd;
                border-radius: 4px;
            }
        """)
        train_header.setAlignment(QtCore.Qt.AlignCenter)
        
        train_refresh_btn = QtWidgets.QPushButton("🔄 Sort")
        train_refresh_btn.setMaximumWidth(60)
        train_refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)
        train_refresh_btn.clicked.connect(lambda: self.train_grid.refresh_grid())
        
        train_header_layout.addWidget(train_header)
        train_header_layout.addWidget(train_refresh_btn)
        train_layout.addWidget(train_header_widget)
        
        self.train_grid = ImageGridWidget("train")
        self.train_grid.set_main_widget(self)
        self.train_grid.setStyleSheet("QWidget { border: 2px dashed #2196f3; background-color: #f8f9ff; }")
        train_layout.addWidget(self.train_grid)
        
        # Add to splitter (test first, then train)
        self.train_test_splitter.addWidget(test_widget)
        self.train_test_splitter.addWidget(train_widget)
        self.train_test_splitter.setSizes([700, 700])  # Equal sizes
        
        right_layout.addWidget(self.train_test_splitter)
        
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([350, 1050])  # Show left panel by default
        
    def toggle_left_panel(self, checked):
        """Toggle the visibility of the left panel."""
        if checked:
            # Hide panel
            self.main_splitter.setSizes([0, 1400])
            self.toggle_btn.setText("▶ Show Panel")
        else:
            # Show panel
            self.main_splitter.setSizes([350, 1050])
            self.toggle_btn.setText("◀ Hide Panel")
            
    def choose_folder(self, target_edit: QtWidgets.QLineEdit):
        """Open folder selection dialog."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            target_edit.setText(folder)
            
    def scan_folders(self):
        """Scan the selected folders for labels."""
        train_path = self.train_edit.text()
        test_path = self.test_edit.text()
        
        if not train_path or not test_path:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select both train and test folders.")
            return
            
        self.train_root = Path(train_path)
        self.test_root = Path(test_path)
        
        # Get all subdirectories (labels)
        try:
            train_labels = {d.name for d in self.train_root.iterdir() if d.is_dir()}
        except Exception as e:
            train_labels = set()
            
        try:
            test_labels = {d.name for d in self.test_root.iterdir() if d.is_dir()}
        except Exception as e:
            test_labels = set()
            
        all_labels = sorted(train_labels | test_labels)
        
        # Populate labels list
        self.labels_list.clear()
        for label in all_labels:
            train_count = self._count_images_in_dir(self.train_root / label)
            test_count = self._count_images_in_dir(self.test_root / label)
            
            item = QtWidgets.QListWidgetItem(f"{label} (Train: {train_count}, Test: {test_count})")
            item.setData(QtCore.Qt.UserRole, label)
            self.labels_list.addItem(item)
            
        self.status_label.setText(f"Found {len(all_labels)} labels")
        
    def _count_images_in_dir(self, path: Path) -> int:
        """Count images in a directory."""
        if not path.exists():
            return 0
        try:
            return len([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS])
        except Exception:
            return 0
            
    def show_label_details(self):
        """Show images for the selected label."""
        current_item = self.labels_list.currentItem()
        if not current_item:
            return
            
        self.current_label = current_item.data(QtCore.Qt.UserRole)
        
        # Get image paths
        train_images = self._list_images(self.train_root / self.current_label)
        test_images = self._list_images(self.test_root / self.current_label)
        
        # Update grids
        self.train_grid.set_images([str(p) for p in train_images])
        self.test_grid.set_images([str(p) for p in test_images])
        
        self.status_label.setText(f"Showing {self.current_label}: {len(train_images)} train, {len(test_images)} test")
        
    def _list_images(self, path: Path):
        """List all images in a directory."""
        if not path or not path.exists():
            return []
        try:
            return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS]
        except Exception:
            return []
            
    def move_image_async(self, file_path, target_type):
        """Move image between train and test folders asynchronously."""
        # Use QTimer to defer file operation to avoid blocking UI
        QtCore.QTimer.singleShot(0, lambda: self._perform_file_move(file_path, target_type))
        
    def _perform_file_move(self, file_path, target_type):
        """Perform the actual file move operation."""
        try:
            source_path = Path(file_path)
            
            # If file doesn't exist at given path, try to find it in current directories
            if not source_path.exists():
                # Try to find the file in train or test directories
                filename = source_path.name
                train_path = self.train_root / self.current_label / filename
                test_path = self.test_root / self.current_label / filename
                
                if train_path.exists():
                    source_path = train_path
                elif test_path.exists():
                    source_path = test_path
                else:
                    QtWidgets.QMessageBox.warning(self, "Error", f"Source file not found: {filename}")
                    return
                
            # Determine source and target directories
            if "train" in str(source_path):
                source_type = "train"
                target_dir = self.test_root / self.current_label
            else:
                source_type = "test"
                target_dir = self.train_root / self.current_label
                
            # Create target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name
            
            # Move the file
            shutil.move(str(source_path), str(target_path))
            
            self.status_label.setText(f"Moved {source_path.name} from {source_type} to {target_type}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to move image: {str(e)}")
            # On error, we should refresh grids to restore correct state
            self.refresh_current_label()
            
    def move_image(self, file_path, target_type):
        """Legacy synchronous move method - kept for compatibility."""
        self._perform_file_move(file_path, target_type)
            
    # Removed refresh_single_grid_after_move - now using instant visual updates
    
    def show_popup_preview(self, image_path):
        """Show popup preview of an image."""
        popup = PopupPreview(image_path, self)
        popup.show()
        self.popup_previews.append(popup)
        
    def show_auto_sort_dialog(self):
        """Show the auto-sort settings dialog."""
        if not self.train_root or not self.test_root:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select train and test folders first.")
            return
            
        if self.labels_list.count() == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No labels found. Please scan folders first.")
            return
            
        # Get all available labels
        available_labels = [self.labels_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.labels_list.count())]
        
        dialog = AutoSortSettingsDialog(self)
        # Set available labels for test mode selection
        dialog.set_available_labels(available_labels)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            settings = dialog.get_settings()
            self.start_auto_sort(settings)
            
    def start_auto_sort(self, settings):
        """Start the auto-sorting process."""
        try:
            # Get all labels or just selected one for testing
            if settings.get('test_mode', False):
                if self.current_label:
                    labels_to_sort = [self.current_label]
                else:
                    QtWidgets.QMessageBox.warning(self, "Warning", "Please select a label for test mode.")
                    return
            else:
                labels_to_sort = [self.labels_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.labels_list.count())]
                
            if not labels_to_sort:
                QtWidgets.QMessageBox.warning(self, "Warning", "No labels to sort.")
                return
                
            # Show progress dialog
            self.progress_dialog = AutoSortProgressDialog(self)
            self.progress_dialog.show()
            
            # Create and start worker thread
            self.sort_worker = AutoSortWorker(
                train_folder=str(self.train_root),
                test_folder=str(self.test_root),
                labels=labels_to_sort,
                settings=settings
            )
            
            # Connect signals
            self.sort_worker.progress_updated.connect(self.progress_dialog.update_progress)
            self.sort_worker.status_updated.connect(self.progress_dialog.update_status)
            self.sort_worker.finished.connect(self.on_auto_sort_finished)
            self.sort_worker.error_occurred.connect(self.on_auto_sort_error)
            
            # Connect cancel button
            self.progress_dialog.cancelled.connect(self.sort_worker.cancel)
            
            # Start the worker
            self.sort_worker.start()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to start auto-sort: {str(e)}")
            
    def on_auto_sort_finished(self, results):
        """Handle auto-sort completion."""
        self.progress_dialog.close()
        
        # Show results dialog
        result_dialog = AutoSortResultDialog(results, self)
        result_dialog.exec_()
        
        # Refresh the current view
        if self.current_label:
            self.refresh_current_label()
            
        self.status_label.setText("Auto-sort completed successfully!")
        
    def on_auto_sort_error(self, error_message):
        """Handle auto-sort error."""
        self.progress_dialog.close()
        QtWidgets.QMessageBox.critical(self, "Auto-Sort Error", f"Auto-sort failed: {error_message}")
        self.status_label.setText("Auto-sort failed.")
        
        # Clean up closed popups
        self.popup_previews = [p for p in self.popup_previews if p.isVisible()]
        
    def refresh_current_label(self):
        """Refresh the current label display."""
        if self.current_label:
            self.show_label_details()


