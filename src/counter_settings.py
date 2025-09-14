from PyQt5 import QtWidgets, QtCore, QtGui
import json
from pathlib import Path

class CounterSettings(QtCore.QObject):
    """Manages counter display settings and provides a centralized interface."""
    
    settings_changed = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled = True
        self._target_train_ratio = 0.8  # Default 80% train, 20% test
        self._show_percentages = True
        
        # Load settings from file if exists
        self.load_settings()
    
    @property
    def enabled(self):
        return self._enabled
    
    @enabled.setter
    def enabled(self, value):
        if self._enabled != value:
            self._enabled = value
            self.settings_changed.emit()
            self.save_settings()
    
    @property
    def target_train_ratio(self):
        return self._target_train_ratio
    
    @target_train_ratio.setter
    def target_train_ratio(self, value):
        if self._target_train_ratio != value:
            self._target_train_ratio = value
            self.settings_changed.emit()
            self.save_settings()
    
    @property
    def target_test_ratio(self):
        return 1.0 - self._target_train_ratio
    
    @property
    def show_percentages(self):
        return self._show_percentages
    
    @show_percentages.setter
    def show_percentages(self, value):
        if self._show_percentages != value:
            self._show_percentages = value
            self.settings_changed.emit()
            self.save_settings()
    

    
    def get_settings_file_path(self):
        """Get the path to the settings file."""
        return Path(__file__).parent.parent / "counter_settings.json"
    
    def save_settings(self):
        """Save settings to file."""
        try:
            settings_data = {
                'enabled': self._enabled,
                'target_train_ratio': self._target_train_ratio,
                'show_percentages': self._show_percentages
            }
            
            with open(self.get_settings_file_path(), 'w') as f:
                json.dump(settings_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save counter settings: {e}")
    
    def load_settings(self):
        """Load settings from file."""
        try:
            settings_file = self.get_settings_file_path()
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings_data = json.load(f)
                
                self._enabled = settings_data.get('enabled', True)
                self._target_train_ratio = settings_data.get('target_train_ratio', 0.8)
                self._show_percentages = settings_data.get('show_percentages', True)
        except Exception as e:
            print(f"Failed to load counter settings: {e}")


class CounterWidget(QtWidgets.QWidget):
    """Widget for displaying train/test counters in compact mode only."""
    
    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)
        self.settings = settings_manager
        self.train_count = 0
        self.test_count = 0
        self.total_count = 0
        
        self._setup_ui()
        self._connect_signals()
        self.update_display()
    
    def _setup_ui(self):
        """Setup the compact UI components."""
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(8, 4, 8, 4)
        self.main_layout.setSpacing(15)
        
        # Train counter section
        self.train_section = self._create_compact_counter_section("🚂 TRAIN", "#1976d2", "#e3f2fd")
        self.main_layout.addWidget(self.train_section)
        
        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        separator.setStyleSheet("color: #ccc;")
        self.main_layout.addWidget(separator)
        
        # Test counter section
        self.test_section = self._create_compact_counter_section("🧪 TEST", "#388e3c", "#e8f5e8")
        self.main_layout.addWidget(self.test_section)
        
        # Overall styling
        self.setStyleSheet("""
            CounterWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        
        self.setMaximumHeight(60)
    
    def _create_compact_counter_section(self, title, color, bg_color):
        """Create a compact counter section with title, count, and percentage."""
        section = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)
        
        # Title
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 11px;
                font-weight: bold;
                color: {color};
            }}
        """)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Count and percentage in horizontal layout
        count_layout = QtWidgets.QHBoxLayout()
        count_layout.setContentsMargins(0, 0, 0, 0)
        count_layout.setSpacing(4)
        
        count_label = QtWidgets.QLabel("0")
        count_label.setStyleSheet(f"""
            QLabel {{
                font-weight: bold;
                font-size: 16px;
                color: {color};
            }}
        """)
        count_label.setAlignment(QtCore.Qt.AlignCenter)
        
        percentage_label = QtWidgets.QLabel("(0%)")
        percentage_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
            }
        """)
        percentage_label.setAlignment(QtCore.Qt.AlignCenter)
        
        count_layout.addWidget(count_label)
        count_layout.addWidget(percentage_label)
        layout.addLayout(count_layout)
        
        # Store references
        if "TRAIN" in title:
            self.train_count_label = count_label
            self.train_percentage_label = percentage_label
        else:
            self.test_count_label = count_label
            self.test_percentage_label = percentage_label
        
        section.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color};
                border-radius: 4px;
                padding: 4px;
                min-width: 80px;
            }}
        """)
        
        return section
    
    def _connect_signals(self):
        """Connect settings signals."""
        self.settings.settings_changed.connect(self.update_display)
    
    def update_counts(self, train_count, test_count):
        """Update the counter with new counts."""
        self.train_count = train_count
        self.test_count = test_count
        self.total_count = train_count + test_count
        self.update_display()
    
    def update_display(self):
        """Update the display based on current counts and settings."""
        if not self.settings.enabled:
            self.hide()
            return
        
        self.show()
        
        # Update counts
        self.train_count_label.setText(str(self.train_count))
        self.test_count_label.setText(str(self.test_count))
        
        # Update percentages
        if self.settings.show_percentages and self.total_count > 0:
            train_pct = (self.train_count / self.total_count) * 100
            test_pct = (self.test_count / self.total_count) * 100
            self.train_percentage_label.setText(f"({train_pct:.1f}%)")
            self.test_percentage_label.setText(f"({test_pct:.1f}%)")
            self.train_percentage_label.show()
            self.test_percentage_label.show()
        else:
            self.train_percentage_label.hide()
            self.test_percentage_label.hide()
        
        # Update tooltips
        self.train_section.setToolTip(f"Train images: {self.train_count}")
        self.test_section.setToolTip(f"Test images: {self.test_count}")
    



class CounterSettingsWidget(QtWidgets.QGroupBox):
    """Widget for counter settings UI."""
    
    settings_changed = QtCore.pyqtSignal()
    
    def __init__(self, settings_manager, parent=None):
        super().__init__("Counter Settings", parent)
        self.settings = settings_manager
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the settings UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Enable counter checkbox
        self.enabled_checkbox = QtWidgets.QCheckBox("Enable Counter")
        self.enabled_checkbox.setChecked(self.settings.enabled)
        layout.addWidget(self.enabled_checkbox)
        
        # Show percentages checkbox
        self.percentages_checkbox = QtWidgets.QCheckBox("Show Percentages")
        self.percentages_checkbox.setChecked(self.settings.show_percentages)
        layout.addWidget(self.percentages_checkbox)
        
        # Add info label about compact mode
        # info_label = QtWidgets.QLabel("Counter is now always displayed in compact mode")
        # info_label.setStyleSheet("""
        #     QLabel {
        #         color: #666;
        #         font-size: 11px;
        #         font-style: italic;
        #         padding: 5px;
        #     }
        # """)
        # layout.addWidget(info_label)
    
    def _connect_signals(self):
        """Connect UI signals to update settings."""
        self.enabled_checkbox.toggled.connect(self._update_settings)
        self.percentages_checkbox.toggled.connect(self._update_settings)
    
    def _update_settings(self):
        """Update settings based on UI state."""
        self.settings.enabled = self.enabled_checkbox.isChecked()
        self.settings.show_percentages = self.percentages_checkbox.isChecked()
        self.settings_changed.emit()