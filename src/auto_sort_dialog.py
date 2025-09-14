from PyQt5 import QtWidgets, QtCore, QtGui
import json
import os
from pathlib import Path


class AutoSortSettingsDialog(QtWidgets.QDialog):
    """Dialog for configuring auto-sort parameters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Sort Settings")
        self.setModal(True)
        self.resize(500, 600)
        
        # Default settings - optimized for better duplicate detection
        self.settings = {
            'train_ratio': 0.75,  # Slightly more balanced split
            'similarity_threshold': 0.92,  # More strict semantic similarity
            'hash_threshold': 8,  # More lenient perceptual hash for better duplicate detection
            'remove_exact_duplicates': True,
            'keep_near_duplicates': False,  # Remove near-duplicates for cleaner dataset
            'rotation_similarity_threshold': 0.88,  # More strict rotation detection
            'foreground_similarity_threshold': 0.90,  # More strict foreground similarity
            'cross_validation_folds': 5,
            'test_mode': False,
            'test_label': '',
            'evaluate_models': True,  # Enable model evaluation by default
            # Content-based filtering settings
            'enable_content_filtering': False,
            'complexity_threshold': 0.6,  # Threshold for determining complex vs simple content
            'prefer_complex_in_train': True,  # Put complex images (full plants) in training
            'prefer_simple_in_test': True,  # Put simple images (leaves) in test for evaluation
            'content_balance_ratio': 0.7  # Ratio of complex images to prefer in training
        }
        
        self.load_settings()
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create scroll area for settings
        scroll = QtWidgets.QScrollArea()
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        # Basic Settings Group
        basic_group = QtWidgets.QGroupBox("Basic Settings")
        basic_layout = QtWidgets.QFormLayout(basic_group)
        
        # Train/Test Ratio
        self.train_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.9)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setDecimals(2)
        self.train_ratio_spin.setValue(self.settings['train_ratio'])
        basic_layout.addRow("Train Ratio:", self.train_ratio_spin)
        
        # Test Ratio (calculated automatically)
        self.test_ratio_label = QtWidgets.QLabel()
        self.update_test_ratio_label()
        basic_layout.addRow("Test Ratio:", self.test_ratio_label)
        
        scroll_layout.addWidget(basic_group)
        
        # Duplicate Detection Group
        duplicate_group = QtWidgets.QGroupBox("Duplicate Detection")
        duplicate_layout = QtWidgets.QFormLayout(duplicate_group)
        
        # Similarity Threshold
        self.similarity_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.similarity_threshold_spin.setRange(0.5, 1.0)
        self.similarity_threshold_spin.setSingleStep(0.01)
        self.similarity_threshold_spin.setDecimals(3)
        self.similarity_threshold_spin.setValue(self.settings['similarity_threshold'])
        duplicate_layout.addRow("Semantic Similarity Threshold:", self.similarity_threshold_spin)
        
        # Hash Threshold
        self.hash_threshold_spin = QtWidgets.QSpinBox()
        self.hash_threshold_spin.setRange(1, 20)
        self.hash_threshold_spin.setValue(self.settings['hash_threshold'])
        duplicate_layout.addRow("Perceptual Hash Threshold:", self.hash_threshold_spin)
        
        # Rotation Similarity
        self.rotation_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.rotation_threshold_spin.setRange(0.5, 1.0)
        self.rotation_threshold_spin.setSingleStep(0.01)
        self.rotation_threshold_spin.setDecimals(3)
        self.rotation_threshold_spin.setValue(self.settings['rotation_similarity_threshold'])
        duplicate_layout.addRow("Rotation Similarity Threshold:", self.rotation_threshold_spin)
        
        # Foreground Similarity
        self.foreground_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.foreground_threshold_spin.setRange(0.5, 1.0)
        self.foreground_threshold_spin.setSingleStep(0.01)
        self.foreground_threshold_spin.setDecimals(3)
        self.foreground_threshold_spin.setValue(self.settings['foreground_similarity_threshold'])
        duplicate_layout.addRow("Foreground Similarity Threshold:", self.foreground_threshold_spin)
        
        # Remove Exact Duplicates
        self.remove_exact_cb = QtWidgets.QCheckBox("Remove exact duplicates (keep best quality)")
        self.remove_exact_cb.setChecked(self.settings['remove_exact_duplicates'])
        duplicate_layout.addRow(self.remove_exact_cb)
        
        # Keep Near Duplicates
        self.keep_near_cb = QtWidgets.QCheckBox("Keep near-duplicates in dataset")
        self.keep_near_cb.setChecked(self.settings['keep_near_duplicates'])
        duplicate_layout.addRow(self.keep_near_cb)
        
        scroll_layout.addWidget(duplicate_group)
        
        # Content-Based Filtering Group
        content_group = QtWidgets.QGroupBox("Content-Based Filtering")
        content_layout = QtWidgets.QFormLayout(content_group)
        
        # Enable Content Filtering
        self.enable_content_cb = QtWidgets.QCheckBox("Enable intelligent content-based splitting")
        self.enable_content_cb.setChecked(self.settings['enable_content_filtering'])
        content_layout.addRow(self.enable_content_cb)
        
        # Complexity Threshold
        self.complexity_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.complexity_threshold_spin.setRange(0.1, 1.0)
        self.complexity_threshold_spin.setSingleStep(0.05)
        self.complexity_threshold_spin.setDecimals(2)
        self.complexity_threshold_spin.setValue(self.settings['complexity_threshold'])
        self.complexity_threshold_spin.setEnabled(self.settings['enable_content_filtering'])
        content_layout.addRow("Complexity Threshold:", self.complexity_threshold_spin)
        
        # Content Balance Ratio
        self.content_balance_spin = QtWidgets.QDoubleSpinBox()
        self.content_balance_spin.setRange(0.1, 0.9)
        self.content_balance_spin.setSingleStep(0.05)
        self.content_balance_spin.setDecimals(2)
        self.content_balance_spin.setValue(self.settings['content_balance_ratio'])
        self.content_balance_spin.setEnabled(self.settings['enable_content_filtering'])
        content_layout.addRow("Complex Content in Train Ratio:", self.content_balance_spin)
        
        # Prefer Complex in Train
        self.prefer_complex_train_cb = QtWidgets.QCheckBox("Prefer complex content (full plants) in training")
        self.prefer_complex_train_cb.setChecked(self.settings['prefer_complex_in_train'])
        self.prefer_complex_train_cb.setEnabled(self.settings['enable_content_filtering'])
        content_layout.addRow(self.prefer_complex_train_cb)
        
        # Prefer Simple in Test
        self.prefer_simple_test_cb = QtWidgets.QCheckBox("Prefer simple content (leaves, parts) in testing")
        self.prefer_simple_test_cb.setChecked(self.settings['prefer_simple_in_test'])
        self.prefer_simple_test_cb.setEnabled(self.settings['enable_content_filtering'])
        content_layout.addRow(self.prefer_simple_test_cb)
        
        # Content filtering help text
        content_help = QtWidgets.QLabel()
        content_help.setWordWrap(True)
        content_help.setText("Content filtering analyzes image complexity to intelligently distribute full plants vs. plant parts between train/test sets for better model learning.")
        content_help.setStyleSheet("color: #666; font-size: 11px; margin-top: 5px;")
        content_layout.addRow(content_help)
        
        scroll_layout.addWidget(content_group)
        
        # Advanced Settings Group
        advanced_group = QtWidgets.QGroupBox("Advanced Settings")
        advanced_layout = QtWidgets.QFormLayout(advanced_group)
        
        # Cross Validation Folds
        self.cv_folds_spin = QtWidgets.QSpinBox()
        self.cv_folds_spin.setRange(3, 10)
        self.cv_folds_spin.setValue(self.settings['cross_validation_folds'])
        advanced_layout.addRow("Cross-Validation Folds:", self.cv_folds_spin)
        
        # Evaluate Models
        self.evaluate_models_cb = QtWidgets.QCheckBox("Evaluate different model configurations")
        self.evaluate_models_cb.setChecked(self.settings['evaluate_models'])
        advanced_layout.addRow(self.evaluate_models_cb)
        
        scroll_layout.addWidget(advanced_group)
        
        # Test Mode Group
        test_group = QtWidgets.QGroupBox("Test Mode")
        test_layout = QtWidgets.QFormLayout(test_group)
        
        # Test Mode Checkbox
        self.test_mode_cb = QtWidgets.QCheckBox("Test mode (sort only one label)")
        self.test_mode_cb.setChecked(self.settings['test_mode'])
        test_layout.addRow(self.test_mode_cb)
        
        # Test Label Selection
        self.test_label_combo = QtWidgets.QComboBox()
        self.test_label_combo.setEnabled(self.settings['test_mode'])
        test_layout.addRow("Test Label:", self.test_label_combo)
        
        scroll_layout.addWidget(test_group)
        
        # Help Text
        help_text = QtWidgets.QTextEdit()
        help_text.setMaximumHeight(120)
        help_text.setReadOnly(True)
        help_text.setHtml("""
        <b>Settings Guide:</b><br>
        • <b>Similarity Thresholds:</b> Higher values = stricter duplicate detection<br>
        • <b>Train Ratio:</b> Proportion of images for training (0.8 = 80% train, 20% test)<br>
        • <b>Content Filtering:</b> Intelligently distributes complex vs simple content between train/test<br>
        • <b>Test Mode:</b> Process only one label to test settings before full execution<br>
        • <b>Model Evaluation:</b> Tests different configurations (slower but more accurate)
        """)
        scroll_layout.addWidget(help_text)
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        # Reset to Defaults
        reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        # Cancel and OK buttons
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
    def connect_signals(self):
        """Connect UI signals"""
        self.train_ratio_spin.valueChanged.connect(self.update_test_ratio_label)
        self.test_mode_cb.toggled.connect(self.test_label_combo.setEnabled)
        self.enable_content_cb.toggled.connect(self.toggle_content_filtering_controls)
        
    def update_test_ratio_label(self):
        """Update the test ratio label when train ratio changes"""
        test_ratio = 1.0 - self.train_ratio_spin.value()
        self.test_ratio_label.setText(f"{test_ratio:.2f}")
        
    def toggle_content_filtering_controls(self, enabled: bool):
        """Enable/disable content filtering controls based on checkbox state"""
        self.complexity_threshold_spin.setEnabled(enabled)
        self.content_balance_spin.setEnabled(enabled)
        self.prefer_complex_train_cb.setEnabled(enabled)
        self.prefer_simple_test_cb.setEnabled(enabled)
        
    def set_available_labels(self, labels):
        """Set available labels for test mode selection"""
        self.test_label_combo.clear()
        self.test_label_combo.addItems(labels)
        if self.settings['test_label'] in labels:
            self.test_label_combo.setCurrentText(self.settings['test_label'])
            
    def get_settings(self):
        """Get current settings from UI"""
        return {
            'train_ratio': self.train_ratio_spin.value(),
            'similarity_threshold': self.similarity_threshold_spin.value(),
            'hash_threshold': self.hash_threshold_spin.value(),
            'remove_exact_duplicates': self.remove_exact_cb.isChecked(),
            'keep_near_duplicates': self.keep_near_cb.isChecked(),
            'rotation_similarity_threshold': self.rotation_threshold_spin.value(),
            'foreground_similarity_threshold': self.foreground_threshold_spin.value(),
            'cross_validation_folds': self.cv_folds_spin.value(),
            'test_mode': self.test_mode_cb.isChecked(),
            'test_label': self.test_label_combo.currentText(),
            'evaluate_models': self.evaluate_models_cb.isChecked(),
            # Content-based filtering settings
            'enable_content_filtering': self.enable_content_cb.isChecked(),
            'complexity_threshold': self.complexity_threshold_spin.value(),
            'prefer_complex_in_train': self.prefer_complex_train_cb.isChecked(),
            'prefer_simple_in_test': self.prefer_simple_test_cb.isChecked(),
            'content_balance_ratio': self.content_balance_spin.value()
        }
        
    def accept(self):
        """Accept dialog and save settings"""
        self.settings = self.get_settings()
        self.save_settings()
        super().accept()
        
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        defaults = {
            'train_ratio': 0.8,
            'similarity_threshold': 0.87,
            'hash_threshold': 5,
            'remove_exact_duplicates': True,
            'keep_near_duplicates': False,
            'rotation_similarity_threshold': 0.82,
            'foreground_similarity_threshold': 0.85,
            'cross_validation_folds': 5,
            'test_mode': False,
            'test_label': '',
            'evaluate_models': False,
            # Content-based filtering defaults
            'enable_content_filtering': False,
            'complexity_threshold': 0.6,
            'prefer_complex_in_train': True,
            'prefer_simple_in_test': True,
            'content_balance_ratio': 0.7
        }
        
        self.train_ratio_spin.setValue(defaults['train_ratio'])
        self.similarity_threshold_spin.setValue(defaults['similarity_threshold'])
        self.hash_threshold_spin.setValue(defaults['hash_threshold'])
        self.remove_exact_cb.setChecked(defaults['remove_exact_duplicates'])
        self.keep_near_cb.setChecked(defaults['keep_near_duplicates'])
        self.rotation_threshold_spin.setValue(defaults['rotation_similarity_threshold'])
        self.foreground_threshold_spin.setValue(defaults['foreground_similarity_threshold'])
        self.cv_folds_spin.setValue(defaults['cross_validation_folds'])
        self.test_mode_cb.setChecked(defaults['test_mode'])
        self.evaluate_models_cb.setChecked(defaults['evaluate_models'])
        # Reset content filtering settings
        self.enable_content_cb.setChecked(defaults['enable_content_filtering'])
        self.complexity_threshold_spin.setValue(defaults['complexity_threshold'])
        self.prefer_complex_train_cb.setChecked(defaults['prefer_complex_in_train'])
        self.prefer_simple_test_cb.setChecked(defaults['prefer_simple_in_test'])
        self.content_balance_spin.setValue(defaults['content_balance_ratio'])
        
    def get_settings_file_path(self):
        """Get path to settings file"""
        app_dir = Path(__file__).parent.parent
        return app_dir / "auto_sort_settings.json"
        
    def save_settings(self):
        """Save settings to file"""
        try:
            settings_file = self.get_settings_file_path()
            with open(settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Failed to save auto-sort settings: {e}")
            
    def load_settings(self):
        """Load settings from file"""
        try:
            settings_file = self.get_settings_file_path()
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
        except Exception as e:
            print(f"Failed to load auto-sort settings: {e}")