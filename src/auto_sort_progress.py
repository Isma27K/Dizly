from PyQt5 import QtWidgets, QtCore, QtGui


class AutoSortProgressDialog(QtWidgets.QDialog):
    """Progress dialog for auto-sorting operations"""
    
    cancelled = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Sort Progress")
        self.setModal(True)
        self.resize(500, 300)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        
        self.worker = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Auto-Sorting Dataset")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("margin: 10px; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Details text area
        details_label = QtWidgets.QLabel("Details:")
        details_label.setStyleSheet("font-weight: bold; margin-top: 15px; margin-bottom: 5px;")
        layout.addWidget(details_label)
        
        self.details_text = QtWidgets.QTextEdit()
        self.details_text.setMaximumHeight(120)
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.details_text)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        # Info about AI processing
        info_label = QtWidgets.QLabel("⚡ Using AI for duplicate detection and quality assessment")
        info_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        button_layout.addWidget(info_label)
        
        button_layout.addStretch()
        
        # Cancel button
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_operation)
        button_layout.addWidget(self.cancel_btn)
        
        # Close button (initially hidden)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setVisible(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def start_sorting(self, worker):
        """Start the sorting operation with the given worker"""
        self.worker = worker
        
        # Connect worker signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.finished.connect(self.sorting_finished)
        self.worker.error_occurred.connect(self.handle_error)
        
        # Start the worker
        self.worker.start()
        
        # Update UI
        self.status_label.setText("Starting auto-sort process...")
        self.add_detail("Auto-sort process initiated")
        
    def update_progress(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
        self.add_detail(f"Progress: {progress}%")
        
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
        self.add_detail(status)
        
    def handle_error(self, error_message):
        """Handle error from worker"""
        self.status_label.setText("❌ Auto-sorting failed")
        self.status_label.setStyleSheet("margin: 10px; font-size: 12px; color: red; font-weight: bold;")
        self.add_detail(f"❌ ERROR: {error_message}")
        
        # Update buttons
        self.cancel_btn.setVisible(False)
        self.close_btn.setVisible(True)
        self.close_btn.setDefault(True)
        
    def sorting_finished(self, results):
        """Handle sorting completion"""
        if results:
            self.progress_bar.setValue(100)
            self.status_label.setText("✅ Auto-sorting completed successfully!")
            self.status_label.setStyleSheet("margin: 10px; font-size: 12px; color: green; font-weight: bold;")
            
            # Display results summary
            total_processed = sum(label_results.get('total_images', 0) for label_results in results.values())
            self.add_detail(f"✅ SUCCESS: Processed {total_processed} images across {len(results)} labels")
            
            for label, label_results in results.items():
                if label_results:
                    self.add_detail(f"  {label}: {label_results.get('total_images', 0)} images, "
                                  f"{label_results.get('duplicates_removed', 0)} duplicates removed")
        else:
            self.status_label.setText("❌ Auto-sorting completed with errors")
            self.status_label.setStyleSheet("margin: 10px; font-size: 12px; color: orange; font-weight: bold;")
            self.add_detail("⚠️ WARNING: Some labels could not be processed")
            
        # Update buttons
        self.cancel_btn.setVisible(False)
        self.close_btn.setVisible(True)
        self.close_btn.setDefault(True)
        
    def cancel_operation(self):
        """Cancel the sorting operation"""
        if self.worker and self.worker.isRunning():
            self.add_detail("Cancelling operation...")
            self.worker.cancel()
            self.worker.wait(3000)  # Wait up to 3 seconds
            
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(1000)
                
        self.reject()
        
    def add_detail(self, message):
        """Add a detail message to the text area"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.details_text.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.details_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def closeEvent(self, event):
        """Handle dialog close event"""
        if self.worker and self.worker.isRunning():
            reply = QtWidgets.QMessageBox.question(
                self, 
                "Cancel Operation",
                "Auto-sorting is still in progress. Do you want to cancel it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.cancel_operation()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


class AutoSortResultDialog(QtWidgets.QDialog):
    """Dialog to show auto-sort results and statistics"""
    
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setWindowTitle("Auto-Sort Results")
        self.setModal(True)
        self.resize(600, 400)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Auto-Sort Results")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 15px;")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Results summary
        summary_group = QtWidgets.QGroupBox("Summary")
        summary_layout = QtWidgets.QFormLayout(summary_group)
        
        # Add summary statistics
        if self.results:
            summary_layout.addRow("Labels Processed:", QtWidgets.QLabel(str(self.results.get('labels_processed', 0))))
            summary_layout.addRow("Total Images:", QtWidgets.QLabel(str(self.results.get('total_images', 0))))
            summary_layout.addRow("Duplicates Found:", QtWidgets.QLabel(str(self.results.get('duplicates_found', 0))))
            summary_layout.addRow("Images Moved to Test:", QtWidgets.QLabel(str(self.results.get('moved_to_test', 0))))
            summary_layout.addRow("Processing Time:", QtWidgets.QLabel(self.results.get('processing_time', 'N/A')))
        
        layout.addWidget(summary_group)
        
        # Detailed results
        details_group = QtWidgets.QGroupBox("Details")
        details_layout = QtWidgets.QVBoxLayout(details_group)
        
        self.details_text = QtWidgets.QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        
        # Populate details
        if self.results and 'details' in self.results:
            self.details_text.setPlainText(self.results['details'])
        else:
            self.details_text.setPlainText("No detailed results available.")
            
        details_layout.addWidget(self.details_text)
        layout.addWidget(details_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)