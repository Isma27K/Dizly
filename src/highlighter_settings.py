from PyQt5 import QtWidgets, QtCore

class HighlighterSettings(QtCore.QObject):
    """Manages highlighter settings and provides a centralized interface."""
    
    settings_changed = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled = True
        self._show_mode = "Latest Only"  # "Latest Only" or "All Moved"
        self._delay = 3.0
        self._currently_highlighted = []  # Track highlighted thumbnails
    
    @property
    def enabled(self):
        return self._enabled
    
    @enabled.setter
    def enabled(self, value):
        if self._enabled != value:
            self._enabled = value
            self.settings_changed.emit()
    
    @property
    def show_mode(self):
        return self._show_mode
    
    @show_mode.setter
    def show_mode(self, value):
        if self._show_mode != value:
            self._show_mode = value
            self.settings_changed.emit()
    
    @property
    def delay(self):
        return self._delay
    
    @delay.setter
    def delay(self, value):
        if self._delay != value:
            self._delay = value
            self.settings_changed.emit()
    
    def add_highlighted(self, thumbnail):
        """Add a thumbnail to the highlighted list."""
        if thumbnail not in self._currently_highlighted:
            self._currently_highlighted.append(thumbnail)
    
    def remove_highlighted(self, thumbnail):
        """Remove a thumbnail from the highlighted list."""
        if thumbnail in self._currently_highlighted:
            self._currently_highlighted.remove(thumbnail)
    
    def clear_all_highlighted(self):
        """Clear all highlighted thumbnails."""
        for thumbnail in self._currently_highlighted[:]:
            if hasattr(thumbnail, 'is_highlighted'):
                thumbnail.is_highlighted = False
                thumbnail.update_style()
                if hasattr(thumbnail, 'highlight_timer'):
                    thumbnail.highlight_timer.stop()
        self._currently_highlighted.clear()
    
    def clear_others_if_latest_only(self, new_thumbnail):
        """Clear other highlights if in 'Latest Only' mode."""
        if self._show_mode == "Latest Only":
            for thumbnail in self._currently_highlighted[:]:
                if thumbnail != new_thumbnail:
                    if hasattr(thumbnail, 'is_highlighted'):
                        thumbnail.is_highlighted = False
                        thumbnail.update_style()
                        if hasattr(thumbnail, 'highlight_timer'):
                            thumbnail.highlight_timer.stop()
                    self._currently_highlighted.remove(thumbnail)
    
    def get_delay_ms(self):
        """Get delay in milliseconds for QTimer."""
        return int(self._delay * 1000)

class HighlighterSettingsWidget(QtWidgets.QGroupBox):
    """Widget for highlighter settings UI."""
    
    def __init__(self, settings_manager, parent=None):
        super().__init__("Highlighter Settings", parent)
        self.settings = settings_manager
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Show mode setting
        show_mode_layout = QtWidgets.QHBoxLayout()
        show_mode_label = QtWidgets.QLabel("Show Mode:")
        self.show_mode_combo = QtWidgets.QComboBox()
        self.show_mode_combo.addItems(["Latest Only", "All Moved"])
        self.show_mode_combo.setCurrentText(self.settings.show_mode)
        show_mode_layout.addWidget(show_mode_label)
        show_mode_layout.addWidget(self.show_mode_combo)
        layout.addLayout(show_mode_layout)
        
        # Delay duration setting
        delay_layout = QtWidgets.QHBoxLayout()
        delay_label = QtWidgets.QLabel("Delay (seconds):")
        self.delay_spinbox = QtWidgets.QDoubleSpinBox()
        self.delay_spinbox.setRange(0.5, 10.0)
        self.delay_spinbox.setSingleStep(0.5)
        self.delay_spinbox.setValue(self.settings.delay)
        self.delay_spinbox.setDecimals(1)
        delay_layout.addWidget(delay_label)
        delay_layout.addWidget(self.delay_spinbox)
        layout.addLayout(delay_layout)
        
        # Enable/disable highlighting
        self.highlight_enabled = QtWidgets.QCheckBox("Enable Highlighting")
        self.highlight_enabled.setChecked(self.settings.enabled)
        layout.addWidget(self.highlight_enabled)
        
        # Clear all highlights button
        clear_highlights_btn = QtWidgets.QPushButton("Clear All Highlights")
        clear_highlights_btn.clicked.connect(self.settings.clear_all_highlighted)
        layout.addWidget(clear_highlights_btn)
    
    def _connect_signals(self):
        """Connect UI signals to settings updates."""
        self.show_mode_combo.currentTextChanged.connect(
            lambda text: setattr(self.settings, 'show_mode', text)
        )
        self.delay_spinbox.valueChanged.connect(
            lambda value: setattr(self.settings, 'delay', value)
        )
        self.highlight_enabled.toggled.connect(
            lambda checked: setattr(self.settings, 'enabled', checked)
        )