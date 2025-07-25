"""
Fast Protein Liquid Chromatography (FPLC) Control Software
A comprehensive software suite for protein purification workflows

Author: Sagnik Mitra
License: IIT INDORE

This version integrates:
1. Detailed (simulated) hardware communication protocols.
2. A conceptual Model Predictive Control (MPC)-like logic for pressure and gradient control.
"""
import sys
import numpy as np
from datetime import datetime
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTextEdit,
    QTableWidgetItem, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QLabel, QWidget, QDialog, QFormLayout, QTabWidget, QRadioButton,
    QMenuBar, QStatusBar, QComboBox, QMessageBox, QFileDialog, QCheckBox,
    QApplication, QInputDialog, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap, QAction
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import serial
import json
import csv
import logging
import hashlib
import os
import subprocess
import shutil
import random
from enum import Enum
import threading
import time
import collections # For deque

from fplc_system_state import SystemState 
from fplc_method import FPLCMethod, MethodEditor
from fplc_serial_manager import FPLCSerialManager
from fplc_controller import FPLCController
# --- User Management ---
from fplc_user_management import UserManager, PermissionDeniedError
from fplc_sample_manager import SampleManager 
from fplc_maintenance_scheduler import SystemDiagnostics, MaintenanceScheduler 
from fplc_validation_rules import MethodValidationEngine
from fplc_method_validator import MethodStep, MethodValidator
from fplc_data_processor import DataProcessor, DataAnalysis
from fplc_flow_diagram import FPLCSystemView, FPLCWindow
from fplc_column_qualification import ColumnQualification, ColumnQualificationUI
from fplc_peak_analysis import PeakAnalysis
from fplc_error_handler import ErrorSeverity, FPLCError, ErrorHandler, DataIntegrityManager, ConfigManager, SystemConfig, SystemConfig2, ComplianceManager

# --- Logging Configuration ---
# Set up a basic logger for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fplc_log.log"), # Log to file
        logging.StreamHandler(sys.stdout)     # Log to console
    ]
)
logger = logging.getLogger('FPLC_App')

# --- Data Classes and Enums ---
@dataclass
class Step:
    """Represents a single step in an FPLC method."""
    step_type: str
    duration: float  # in minutes
    flow_rate: float # in mL/min
    buffer_a: float  # percentage
    buffer_b: float  # percentage
    collection: bool = False
    notes: str = ""

class FractionCollector(QGroupBox):
    """Enhanced fraction collector control interface."""
    def __init__(self, parent=None):
        super().__init__("Fraction Collector", parent)
        self.collection_mode = "Peak"
        self.current_fraction = 0
        self.fraction_size = 1.0 # mL
        self.max_fractions = 96
        self.collection_threshold = 50 # mAU
        self.is_collecting = False
        self.fractions = []
        self.serial_manager = None # Use the serial manager directly
        self.current_volume = 0.0 # Accumulated volume for volume-based collection
        self.last_uv = 0.0 # Last UV value for peak detection
        self.logger = logging.getLogger('FractionCollector')

        self.setup_ui()
        self.logger.info("Fraction Collector UI initialized.")

    def setup_ui(self):
        """Sets up the UI for the fraction collector."""
        layout = QVBoxLayout()

        # Collection mode selection
        mode_group = QGroupBox("Collection Mode")
        mode_layout = QHBoxLayout()
        self.peak_mode = QRadioButton("Peak-Based")
        self.time_mode = QRadioButton("Time-Based")
        self.volume_mode = QRadioButton("Volume-Based")
        self.peak_mode.setChecked(True)

        mode_layout.addWidget(self.peak_mode)
        mode_layout.addWidget(self.time_mode)
        mode_layout.addWidget(self.volume_mode)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Collection parameters
        params_group = QGroupBox("Collection Parameters")
        params_layout = QFormLayout()

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1000)
        self.threshold_spin.setValue(50)
        params_layout.addRow("Peak Threshold (mAU):", self.threshold_spin)

        self.fraction_size_spin = QDoubleSpinBox()
        self.fraction_size_spin.setRange(0.1, 50)
        self.fraction_size_spin.setValue(1.0)
        params_layout.addRow("Fraction Size (mL):", self.fraction_size_spin)

        self.max_fractions_spin = QSpinBox()
        self.max_fractions_spin.setRange(1, 384)
        self.max_fractions_spin.setValue(96)
        params_layout.addRow("Max Fractions:", self.max_fractions_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Fraction table
        self.fraction_table = QTableWidget()
        self.fraction_table.setColumnCount(5)
        self.fraction_table.setHorizontalHeaderLabels([
            "Fraction #", "Start Time", "Volume (mL)",
            "UV (mAU)", "Collection Status"
        ])
        layout.addWidget(self.fraction_table)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_collection = QPushButton("Start Collection")
        self.stop_collection = QPushButton("Stop Collection")
        self.reset_collection = QPushButton("Reset")

        button_layout.addWidget(self.start_collection)
        button_layout.addWidget(self.stop_collection)
        button_layout.addWidget(self.reset_collection)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        self.start_collection.clicked.connect(self.start_collecting)
        self.stop_collection.clicked.connect(self.stop_collecting)
        self.reset_collection.clicked.connect(self.reset)

        # Connect radio buttons to update collection mode
        self.peak_mode.toggled.connect(lambda: self._set_collection_mode("Peak"))
        self.time_mode.toggled.connect(lambda: self._set_collection_mode("Time"))
        self.volume_mode.toggled.connect(lambda: self._set_collection_mode("Volume"))

    def _set_collection_mode(self, mode):
        """Sets the collection mode and updates UI accordingly."""
        self.collection_mode = mode
        self.logger.info(f"Fraction collection mode set to: {mode}")
        # Enable/disable relevant controls based on mode
        self.threshold_spin.setEnabled(mode == "Peak")
        self.fraction_size_spin.setEnabled(mode in ["Time", "Volume"])


    def set_serial_manager(self, manager: FPLCSerialManager):
        """Sets the serial communication manager."""
        self.serial_manager = manager

    def start_collecting(self):
        """Starts fraction collection."""
        try:
            if self.serial_manager:
                self.serial_manager.send_command("FC_START")

            self.is_collecting = True
            self.collection_threshold = self.threshold_spin.value()
            self.fraction_size = self.fraction_size_spin.value()
            self.max_fractions = self.max_fractions_spin.value()
            self.current_fraction = 0
            self.fractions.clear()
            self.fraction_table.setRowCount(0)
            self.current_volume = 0.0
            self.last_uv = 0.0

            self.start_collection.setEnabled(False)
            self.stop_collection.setEnabled(True)

            self.logger.info("Started fraction collection.")

        except Exception as e:
            self.logger.error(f"Error starting collection: {str(e)}")
            QMessageBox.critical(
                self,
                "Collection Error",
                f"Failed to start collection: {str(e)}"
            )

    def stop_collecting(self):
        """Stops fraction collection."""
        try:
            if self.serial_manager:
                self.serial_manager.send_command("FC_STOP")

            self.is_collecting = False
            self.start_collection.setEnabled(True)
            self.stop_collection.setEnabled(False)

            self.logger.info("Stopped fraction collection.")

        except Exception as e:
            self.logger.error(f"Error stopping collection: {str(e)}")

    def reset(self):
        """Resets the fraction collector."""
        try:
            if self.serial_manager:
                self.serial_manager.send_command("FC_RESET")

            self.current_fraction = 0
            self.fractions.clear()
            self.fraction_table.setRowCount(0)
            self.is_collecting = False
            self.current_volume = 0.0
            self.last_uv = 0.0

            self.start_collection.setEnabled(True)
            self.stop_collection.setEnabled(False)

            self.logger.info("Reset fraction collector.")

        except Exception as e:
            self.logger.error(f"Error resetting collector: {str(e)}")

    def update(self, time_point, uv_value, volume_increment):
        """Updates the fraction collector state based on incoming data."""
        if not self.is_collecting:
            return

        if self.current_fraction >= self.max_fractions:
            self.stop_collecting()
            self.logger.warning("Max fractions reached. Collection stopped.")
            return

        collect_now = False
        if self.collection_mode == "Peak":
            # Simple peak detection logic for collection
            if uv_value > self.collection_threshold and self.last_uv <= self.collection_threshold:
                # Start of a peak (UV crosses threshold going up)
                collect_now = True
            elif uv_value <= self.collection_threshold and self.last_uv > self.collection_threshold:
                # End of a peak (UV drops below threshold) - could trigger end of current fraction
                pass # For simplicity, we just start a new fraction at peak start for now
        elif self.collection_mode == "Volume":
            self.current_volume += volume_increment
            if self.current_volume >= self.fraction_size:
                collect_now = True
                self.current_volume -= self.fraction_size # Keep remainder for next fraction
        # Time-based collection would be handled by a separate timer in the main system or controller

        if collect_now:
            self.current_fraction += 1
            fraction_data = {
                'fraction_num': self.current_fraction,
                'start_time': f"{time_point:.2f} min",
                'volume': f"{self.fraction_size:.2f} mL" if self.collection_mode == "Volume" else "N/A",
                'uv_at_start': f"{uv_value:.2f} mAU",
                'status': "Collecting"
            }
            self.fractions.append(fraction_data)
            self._update_fraction_table(fraction_data)
            self.logger.info(f"Collecting fraction #{self.current_fraction} at {time_point:.2f} min (UV: {uv_value:.2f}).")

        self.last_uv = uv_value


    def _update_fraction_table(self, fraction_data):
        """Updates the fraction table with new fraction data."""
        row = self.fraction_table.rowCount()
        self.fraction_table.insertRow(row)

        self.fraction_table.setItem(row, 0, QTableWidgetItem(str(fraction_data['fraction_num'])))
        self.fraction_table.setItem(row, 1, QTableWidgetItem(fraction_data['start_time']))
        self.fraction_table.setItem(row, 2, QTableWidgetItem(fraction_data['volume']))
        self.fraction_table.setItem(row, 3, QTableWidgetItem(fraction_data['uv_at_start']))
        self.fraction_table.setItem(row, 4, QTableWidgetItem(fraction_data['status']))


class BufferBlending:
    """Handles buffer preparation and blending calculations."""
    def __init__(self):
        self.buffer_recipes = {}
        self.current_blend = {'A': 100, 'B': 0}
        self.logger = logging.getLogger('BufferBlending')
        self.logger.info("Buffer Blending initialized.")

    def add_buffer_recipe(self, name, components):
        """Adds a new buffer recipe."""
        self.buffer_recipes[name] = components
        self.logger.info(f"Buffer recipe '{name}' added.")

    def calculate_blend(self, target_ph, target_conductivity):
        """Simulates buffer blending calculation."""
        buffer_a_ratio = random.uniform(0, 100)
        buffer_b_ratio = 100 - buffer_a_ratio

        return {
            'A': buffer_a_ratio,
            'B': buffer_b_ratio,
            'expected_ph': target_ph + random.uniform(-0.1, 0.1),
            'expected_conductivity': target_conductivity + random.uniform(-1, 1)
        }

# --- UI Components ---
class ChromatogramPlot(QWidget):
    """Advanced chromatogram visualization with peak detection and real-time data display."""
    def __init__(self, data_analysis):
        super().__init__()
        self.data_analysis = data_analysis
        self.logger = logging.getLogger('ChromatogramPlot')
        self.setup_ui()
        self.logger.info("Chromatogram Plot UI initialized.")

    def setup_ui(self):
        """Sets up the UI for the chromatogram plot."""
        main_layout = QVBoxLayout()
        plot_and_controls_layout = QHBoxLayout()

        # Create matplotlib figure with professional styling
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor='#f8f9fa')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax1 = self.figure.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        # Style the plot
        self.ax1.set_xlabel('Time (min)', fontsize=10)
        self.ax1.set_ylabel('UV Absorbance (mAU)', fontsize=10, color='blue')
        self.ax2.set_ylabel('Conductivity (mS/cm)', fontsize=10, color='red')
        self.ax1.grid(True, linestyle='--', alpha=0.3)
        self.figure.tight_layout()

        plot_and_controls_layout.addWidget(self.canvas)

        # Add controls for visualization
        controls_layout = QVBoxLayout()
        self.auto_scale = QPushButton("Auto Scale")
        self.show_peaks = QPushButton("Show Peaks")
        self.export_data_btn = QPushButton("Export Data") # Renamed to avoid conflict

        self.auto_scale.clicked.connect(self.auto_scale_plot)
        self.show_peaks.clicked.connect(self.toggle_peak_display)
        self.export_data_btn.clicked.connect(self.export_plot_data)

        controls_layout.addWidget(self.auto_scale)
        controls_layout.addWidget(self.show_peaks)
        controls_layout.addWidget(self.export_data_btn)

        # Real-time data displays
        data_display_group = QGroupBox("Current Data")
        data_display_layout = QFormLayout()
        self.current_uv_label = QLabel("UV: 0.000 mAU")
        self.current_pressure_label = QLabel("Pressure: 0.000 MPa")
        self.current_conductivity_label = QLabel("Conductivity: 0.00 mS/cm")
        self.current_flow_label = QLabel("Flow: 0.00 mL/min")
        self.current_buffer_a_label = QLabel("Buffer A: 0.0%")
        self.current_buffer_b_label = QLabel("Buffer B: 0.0%")
        self.current_temp_label = QLabel("Temp: 0.0 °C")
        self.current_ph_label = QLabel("pH: 0.00")

        data_display_layout.addRow("UV:", self.current_uv_label)
        data_display_layout.addRow("Pressure:", self.current_pressure_label)
        data_display_layout.addRow("Conductivity:", self.current_conductivity_label)
        data_display_layout.addRow("Flow Rate:", self.current_flow_label)
        data_display_layout.addRow("Buffer A:", self.current_buffer_a_label)
        data_display_layout.addRow("Buffer B:", self.current_buffer_b_label)
        data_display_layout.addRow("Temperature:", self.current_temp_label)
        data_display_layout.addRow("pH:", self.current_ph_label)

        data_display_group.setLayout(data_display_layout)
        controls_layout.addWidget(data_display_group)


        plot_and_controls_layout.addLayout(controls_layout)
        main_layout.addLayout(plot_and_controls_layout)

        # Add peak information display
        self.peak_info = QTableWidget()
        self.peak_info.setColumnCount(6)
        self.peak_info.setHorizontalHeaderLabels([
            "Peak #", "Time (min)", "Height (mAU)",
            "Area (mAU*min)", "Width (min)", "Standard"
        ])
        self.peak_info.setMaximumHeight(150)
        main_layout.addWidget(self.peak_info)

        self.setLayout(main_layout)

        # Initialize plot data
        self.peak_markers = []
        self.showing_peaks = False

    def update_plot(self, time_data, uv_data, conductivity_data):
        """Updates the chromatogram plot with new data."""
        self.ax1.clear()
        self.ax2.clear()

        # Plot UV and conductivity data
        self.ax1.plot(time_data, uv_data, 'b-', label='UV')
        self.ax2.plot(time_data, conductivity_data, 'r-', label='Conductivity')

        self.ax1.set_xlabel('Time (min)', fontsize=10)
        self.ax1.set_ylabel('UV Absorbance (mAU)', fontsize=10, color='blue')
        self.ax2.set_ylabel('Conductivity (mS/cm)', fontsize=10, color='red')
        self.ax1.grid(True, linestyle='--', alpha=0.3)
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper right')

        # Update peak detection if enabled
        if self.showing_peaks:
            self.display_peaks()

        self.canvas.draw()

    def update_realtime_data_labels(self, uv, pressure, conductivity, flow_rate, buffer_a, buffer_b, temperature, ph):
        """Updates the real-time data display labels."""
        self.current_uv_label.setText(f"UV: {uv:.3f} mAU")
        self.current_pressure_label.setText(f"Pressure: {pressure:.3f} MPa")
        self.current_conductivity_label.setText(f"Conductivity: {conductivity:.2f} mS/cm")
        self.current_flow_label.setText(f"Flow: {flow_rate:.2f} mL/min")
        self.current_buffer_a_label.setText(f"Buffer A: {buffer_a:.1f}%")
        self.current_buffer_b_label.setText(f"Buffer B: {buffer_b:.1f}%")
        self.current_temp_label.setText(f"Temp: {temperature:.1f} °C")
        self.current_ph_label.setText(f"pH: {ph:.2f}")


    def display_peaks(self):
        """Detects and displays peaks on the plot and in the table."""
        self.data_analysis.peak_analyzer.detect_peaks(self.data_analysis.time_data, self.data_analysis.uv_data)
        peaks = self.data_analysis.peak_analyzer.peaks

        # Clear previous peak markers
        for marker in self.peak_markers:
            if marker in self.ax1.lines: # Check if marker still exists in plot
                marker.remove()
        self.peak_markers.clear()

        # Plot peak markers and update peak info table
        self.peak_info.setRowCount(len(peaks))
        for i, peak in enumerate(peaks):
            # Add peak marker to plot
            marker = self.ax1.plot(peak['time'], peak['height'], 'r^', markersize=8, label=f'Peak {i+1}')[0]
            self.peak_markers.append(marker)

            # Update peak info table
            self.peak_info.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.peak_info.setItem(i, 1, QTableWidgetItem(f"{peak['time']:.2f}"))
            self.peak_info.setItem(i, 2, QTableWidgetItem(f"{peak['height']:.1f}"))
            self.peak_info.setItem(i, 3, QTableWidgetItem(f"{peak['area']:.1f}"))
            self.peak_info.setItem(i, 4, QTableWidgetItem(f"{peak['width']:.2f}"))
            self.peak_info.setItem(i, 5, QTableWidgetItem(peak['standard_match'] if peak['standard_match'] else "Unknown"))
        self.canvas.draw()
        self.logger.info(f"Displayed {len(peaks)} peaks.")

    def auto_scale_plot(self):
        """Auto-scales the plot axes."""
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()
        self.logger.info("Chromatogram plot auto-scaled.")

    def toggle_peak_display(self):
        """Toggles the display of peaks on the chromatogram."""
        self.showing_peaks = not self.showing_peaks
        if self.showing_peaks:
            self.display_peaks()
            self.show_peaks.setText("Hide Peaks")
            self.logger.info("Peak display enabled.")
        else:
            # Clear peak markers
            for marker in self.peak_markers:
                if marker in self.ax1.lines:
                    marker.remove()
            self.peak_markers.clear()
            self.canvas.draw()
            self.show_peaks.setText("Show Peaks")
            self.logger.info("Peak display disabled.")

    def export_plot_data(self):
        """Prompts the user for export options and exports data."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Data")
        layout = QVBoxLayout()

        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout()

        raw_data_cb = QCheckBox("Raw Chromatogram Data (CSV)")
        raw_data_cb.setChecked(True)
        peak_data_cb = QCheckBox("Peak Analysis Data (CSV)")
        peak_data_cb.setChecked(True)
        report_cb = QCheckBox("Generate HTML Report")
        report_cb.setChecked(True)

        options_layout.addWidget(raw_data_cb)
        options_layout.addWidget(peak_data_cb)
        options_layout.addWidget(report_cb)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        export_btn = QPushButton("Export")
        layout.addWidget(export_btn)

        dialog.setLayout(layout)

        def do_export():
            base_path = QFileDialog.getExistingDirectory(
                self, "Select Export Directory"
            )
            if not base_path:
                self.logger.info("Export cancelled by user (no directory selected).")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            success = True

            # Access the main window to get method and run info
            main_window = self.parent().parent() # Assumes Chromatogram is in a tab in main window
            current_method = main_window.method_editor.method
            current_run_data = main_window.current_run_data

            if raw_data_cb.isChecked():
                filename = os.path.join(base_path, f"chromatogram_data_{timestamp}.csv")
                if not self.data_analysis.export_data(filename):
                    success = False
                    self.logger.error(f"Failed to export raw data to {filename}.")

            if peak_data_cb.isChecked():
                filename = os.path.join(base_path, f"peak_data_{timestamp}.csv")
                if not self.data_analysis.export_peaks(filename):
                    success = False
                    self.logger.error(f"Failed to export peak data to {filename}.")

            if report_cb.isChecked():
                filename = os.path.join(base_path, f"run_report_{timestamp}.html")
                if not self.data_analysis.generate_report(
                    current_method,
                    current_run_data,
                    filename
                ):
                    success = False
                    self.logger.error(f"Failed to generate HTML report to {filename}.")

            if success:
                QMessageBox.information(
                    dialog,
                    "Export Complete",
                    f"Data exported successfully to:\n{base_path}"
                )
                self.logger.info(f"Data exported successfully to {base_path}.")
            else:
                QMessageBox.warning(
                    dialog,
                    "Export Warning",
                    "Some export operations failed. Check the log for details."
                )
                self.logger.warning("Some export operations failed.")

            dialog.accept()

        export_btn.clicked.connect(do_export)
        dialog.exec()

class LoginDialog(QDialog):
    # This __init__ method does NOT take 'user_manager' as an argument.
    # It will handle credential validation internally using hardcoded values.
    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.setWindowTitle("Login")
        self.setFixedSize(300, 180) # Adjust size as needed for your UI

        layout = QVBoxLayout()

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter Username")
        layout.addWidget(QLabel("Username:"))
        layout.addWidget(self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter Password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password) # Masks the password input
        layout.addWidget(QLabel("Password:"))
        layout.addWidget(self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.check_credentials)
        layout.addWidget(self.login_button)

        self.setLayout(layout)

        # --- IMPORTANT: Securely Store Credentials ---
        # These are the hardcoded credentials that will be used for validation.
        self.valid_username = "Sagnik"
        self.valid_password_hash = "25d5089f33c5d34643c097cc7068a772b68593318b4a52d999fa84a4830a3f9a"

    def check_credentials(self):
        entered_username = self.username_input.text()
        # Hash the entered password for comparison with the stored hash
        entered_password_hash = hashlib.sha256(self.password_input.text().encode()).hexdigest()

        overall_match = (entered_username == self.valid_username and
                         entered_password_hash == self.valid_password_hash)

        if overall_match:
            self.accept() # Signals successful login
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")
            self.password_input.clear() # Clear password field for retry
            self.username_input.clear() # Clear username field for retry

# --- Alarm Management ---
class AlarmManager(QObject):
    """Alarm management system."""
    alarm_triggered = pyqtSignal(str, str) # severity, message

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('AlarmManager')
        self.logger.info("Alarm Manager initialized.")

    def add_alarm(self, severity, message):
        """Adds a new alarm and emits a signal."""
        self.logger.warning(f"ALARM: {severity} - {message}")
        self.alarm_triggered.emit(severity, message)

# --- Main Application Window ---
class IndustrialFPLCSystem(QMainWindow):
    """Main FPLC control system with industrial features."""
    def __init__(self, user_manager, parent=None):
        super().__init__()
        self.user_manager = user_manager # Store the instance
        self.setWindowTitle("Industrial FPLC Control System")
        self.setGeometry(100, 100, 1200, 800) # Set initial window size

        # Initialize core components
        self.user_manager = UserManager()
        self.sample_manager = SampleManager()
        self.data_processor = DataProcessor() # Instantiate the data processor
        self.method_validator_engine = MethodValidationEngine() # Instantiate the validation engine
        self.error_severity = ErrorSeverity.INFO
        self.fplc_error = FPLCError("Default system error message", ErrorSeverity.INFO)
        self.error_handler = ErrorHandler()
        self.data_integrity_manager = DataIntegrityManager()
        self.config_manager = ConfigManager()
        self.system_config = SystemConfig()
        self.system_config_chrom = SystemConfig2()
        self.compliance_manager = ComplianceManager(self.config_manager)
        self.current_user = None # Will be set after login
        self.system_state = SystemState.IDLE
        self.serial_manager = FPLCSerialManager() # Central communication hub
        self.data_analysis = DataAnalysis()
        self.fraction_collector = FractionCollector()
        self.fraction_collector.set_serial_manager(self.serial_manager) # Pass the serial manager
        self.diagnostics = SystemDiagnostics(self.serial_manager) # Pass the serial manager
        self.maintenance = MaintenanceScheduler()
        self.fplc_view = FPLCSystemView()
        self.fplc_view_window = FPLCWindow()
        self.buffer_blending = BufferBlending()
        self.column_qualification = ColumnQualification()
        self.alarm_manager = AlarmManager()

        # FPLC Controller (manages run logic and hardware interaction)
        self.method_editor = MethodEditor() # Needs to be initialized before setting controller's method
        self.fplc_controller = FPLCController(self.serial_manager, self.method_editor.method) # Initialize with method from editor

        # Data management for current run
        self.current_run_data = {
            'run_id': None,
            'start_time': None,
            'method_name': None
        }

        self.setup_ui()
        self.setup_menu()
        self.setup_timers()
        self.setup_signals()

        self.system_max_pressure = 25.0  # Example value in MPa or bar
        self.system_max_flow_rate = 5.0  # Example value in ml/min
        system_config = SystemConfig2(max_pressure_psi=self.system_max_pressure, max_flow_rate_ml_min=self.system_max_flow_rate)

        # Then, instantiate MethodValidator with this config object
        self.method_validator = MethodValidator(config=system_config)
        # Initial system state and login prompt
        self.update_system_state(SystemState.IDLE)
        # The main function handles the initial login dialog.
        # After successful login, initialize_system will be called.

        logger.info("Industrial FPLC System application started.")

    def setup_ui(self):
        """Sets up the main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Add logos to the main layout
        main_layout.addLayout(self.add_logos())

        # Control panel
        control_panel = QHBoxLayout()

        # System controls
        control_group = QGroupBox("System Controls")
        control_layout = QVBoxLayout()

        self.start_button = QPushButton("Start Run")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.emergency_stop = QPushButton("EMERGENCY STOP")
        self.load_standards_button = QPushButton("Load Standards")
        self.emergency_stop.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.ph_cond_button = QPushButton("Launch pH & Conductivity Analyzer")
        self.ph_cond_button.clicked.connect(self.launch_ph_conductivity_app)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.emergency_stop)
        control_layout.addWidget(self.load_standards_button)
        control_group.setLayout(control_layout)
        control_panel.addWidget(control_group)

        # Status displays
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()

        self.system_status = QLabel("System Status: IDLE")
        self.logged_in_user_label = QLabel("Logged In: None")
        self.current_step_label = QLabel("Current Step: N/A")
        self.step_time_label = QLabel("Step Time: 0.00 min")
        self.run_progress_bar = QProgressBar(self)
        self.run_progress_bar.setTextVisible(True)
        self.run_progress_bar.setFormat("Run Progress: %p%")

        status_layout.addWidget(self.system_status)
        status_layout.addWidget(self.logged_in_user_label)
        status_layout.addWidget(self.current_step_label)
        status_layout.addWidget(self.step_time_label)
        status_layout.addWidget(self.run_progress_bar)
        status_group.setLayout(status_layout)
        control_panel.addWidget(status_group)

        # MPC Control Status Display
        mpc_status_group = QGroupBox("Control Status (MPC)")
        mpc_status_layout = QFormLayout()
        self.set_flow_label = QLabel("Set Flow: 0.00 mL/min")
        self.actual_flow_label = QLabel("Actual Flow: 0.00 mL/min")
        self.set_buffer_a_label = QLabel("Set Buffer A: 0.0%")
        self.actual_buffer_a_label = QLabel("Actual Buffer A: 0.0%")

        mpc_status_layout.addRow("Target Flow:", self.set_flow_label)
        mpc_status_layout.addRow("Actual Flow:", self.actual_flow_label)
        mpc_status_layout.addRow("Target Buff A:", self.set_buffer_a_label)
        mpc_status_layout.addRow("Actual Buff A:", self.actual_buffer_a_label)
        mpc_status_group.setLayout(mpc_status_layout)
        control_panel.addWidget(mpc_status_group)

        main_layout.addLayout(control_panel)

        # Create tab widget for different views
        self.tab_widget = QTabWidget()

        # Add tabs in the desired order
        # self.method_editor = MethodEditor() # Already initialized as instance variable
        self.tab_widget.addTab(self.method_editor, "Method Editor")

        self.chromatogram = ChromatogramPlot(self.data_analysis)
        self.tab_widget.addTab(self.chromatogram, "Chromatogram")

        self.qualification_ui = ColumnQualificationUI(self.column_qualification)
        self.tab_widget.addTab(self.qualification_ui, "Column Qualification")

        fraction_tab = QWidget()
        fraction_layout = QVBoxLayout()
        fraction_layout.addWidget(self.fraction_collector)
        fraction_tab.setLayout(fraction_layout)
        self.tab_widget.addTab(fraction_tab, "Fraction Collector")

        main_layout.addWidget(self.tab_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready")
        logger.info("Main UI components set up.")

    def setup_menu(self):
        """Sets up the application menu bar with functionality."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        self.new_method_action = QAction("New Method", self)
        self.new_method_action.setShortcut("Ctrl+N")
        self.new_method_action.triggered.connect(self.new_method)
        file_menu.addAction(self.new_method_action)

        self.open_method_action = QAction("Open Method...", self)
        self.open_method_action.setShortcut("Ctrl+O")
        self.open_method_action.triggered.connect(self.open_method)
        file_menu.addAction(self.open_method_action)

        self.save_method_action = QAction("Save Method...", self)
        self.save_method_action.setShortcut("Ctrl+S")
        self.save_method_action.triggered.connect(self.save_method)
        file_menu.addAction(self.save_method_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # User menu
        user_menu = menubar.addMenu("User")
        self.login_action = QAction("Login", self)
        # Login is handled by the main function's dialog, this action is just for visibility
        user_menu.addAction(self.login_action)

        self.logout_action = QAction("Logout", self)
        self.logout_action.triggered.connect(self.logout_user)
        user_menu.addAction(self.logout_action)

        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction(QAction("Method Editor", self, triggered=lambda: self.tab_widget.setCurrentWidget(self.method_editor)))
        view_menu.addAction(QAction("Chromatogram", self, triggered=lambda: self.tab_widget.setCurrentWidget(self.chromatogram)))
        view_menu.addAction(QAction("Column Qualification", self, triggered=lambda: self.tab_widget.setCurrentWidget(self.qualification_ui)))
        view_menu.addAction(QAction("Fraction Collector", self, triggered=lambda: self.tab_widget.setCurrentWidget(self.fraction_collector.parent()))) # Parent is the QWidget holding FC

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        self.run_diagnostics_action = QAction("Run Diagnostics", self)
        self.run_diagnostics_action.triggered.connect(self.run_diagnostics)
        tools_menu.addAction(self.run_diagnostics_action)

        ph_cond_action = QAction("pH & Conductivity Analyzer", self)
        ph_cond_action.triggered.connect(self.launch_ph_conductivity_app)
        tools_menu.addAction(ph_cond_action)

        self.export_logs_action = QAction("Export Logs", self)
        self.export_logs_action.triggered.connect(self.export_logs)
        tools_menu.addAction(self.export_logs_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        documentation = QAction("Documentation", self)
        documentation.triggered.connect(self.open_documentation)
        help_menu.addAction(documentation)

        about = QAction("About", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)
        logger.info("Menu bar set up.")

        self.update_menu_visibility() # Set initial visibility

    def launch_ph_conductivity_app(self):
        # Get the directory of the current FPLC_15.py script
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        ph_cond_script_path = os.path.join(current_script_dir, "fplc_pH&Cond.py")

        if not os.path.exists(ph_cond_script_path):
            QMessageBox.critical(self, "File Not Found",
                                 f"Error: fplc_pH&Cond.py not found at {ph_cond_script_path}")
            logging.error(f"fplc_pH&Cond.py not found at {ph_cond_script_path}")
            return

        try:
            # Launch the pH&Cond.py script using subprocess
            # This runs it as a completely separate process.
            # Use sys.executable to ensure the same Python interpreter is used.
            subprocess.Popen([sys.executable, ph_cond_script_path])
            logging.info(f"Launched pH & Conductivity Analyzer: {ph_cond_script_path}")
        except Exception as e:
            QMessageBox.critical(self, "Launch Error",
                                 f"Could not launch pH & Conductivity Analyzer: {e}")
            logging.error(f"Failed to launch pH & Conductivity Analyzer: {e}")

    def update_menu_visibility(self):
        """Updates menu item visibility based on login status."""
        is_logged_in = (self.user_manager.current_user is not None)
        is_admin = False # Initialize to False, will be updated if a user is logged in and is admin

        try:
            # Attempt to check for 'administrator' permission.
            # This will raise PermissionDeniedError if no user is logged in.
            is_admin = self.user_manager.check_permission('administrator')
        except PermissionDeniedError:
            # If no user is logged in, or if the current user doesn't have the permission
            # (which check_permission would also handle internally with a False return
            # if the current_user is not None but lacks permission),
            # this specific exception means no user is logged in.
            # In this case, is_admin remains False, which is the desired behavior.
            pass # No action needed, is_admin is already False

        self.login_action.setEnabled(not is_logged_in)
        self.logout_action.setEnabled(is_logged_in)

        # Directly use the instance attributes for menu actions
        if hasattr(self, 'new_method_action'):
            self.new_method_action.setEnabled(is_logged_in)
        if hasattr(self, 'open_method_action'):
            self.open_method_action.setEnabled(is_logged_in)
        if hasattr(self, 'save_method_action'):
            self.save_method_action.setEnabled(is_logged_in)
        if hasattr(self, 'run_diagnostics_action'):
            self.run_diagnostics_action.setEnabled(is_admin) # Only admin can run diagnostics
        if hasattr(self, 'export_logs_action'):
            self.export_logs_action.setEnabled(is_admin) # Only admin can export logs

        self.start_button.setEnabled(is_logged_in and self.system_state == SystemState.IDLE)
        self.pause_button.setEnabled(is_logged_in and self.system_state == SystemState.RUNNING)
        self.stop_button.setEnabled(is_logged_in and self.system_state in [SystemState.RUNNING, SystemState.PAUSED])
        self.emergency_stop.setEnabled(is_logged_in) # Always enable if logged in
        self.load_standards_button.setEnabled(is_logged_in)

        # Qualification UI actions
        self.qualification_ui.start_qual.setEnabled(is_logged_in)
        self.qualification_ui.export_results.setEnabled(is_logged_in)
        self.qualification_ui.view_history.setEnabled(is_logged_in)

        # Fraction collector actions
        self.fraction_collector.start_collection.setEnabled(is_logged_in)
        self.fraction_collector.stop_collection.setEnabled(is_logged_in)
        self.fraction_collector.reset_collection.setEnabled(is_logged_in)

        # Update logged in user label
        self.logged_in_user_label.setText(f"Logged In: {self.user_manager.current_user if self.user_manager.current_user else 'None'}")

    def set_logged_in_user(self, username):
        """Sets the current logged-in user and updates UI visibility."""
        self.current_user = username
        self.user_manager.current_user = username # Also update user manager's current user
        self.update_menu_visibility()
        self.initialize_system() # Initialize system after successful login

    def logout_user(self):
        """Logs out the current user."""
        self.user_manager.logout()
        self.current_user = None
        self.status_bar.showMessage("Logged out.")
        self.update_menu_visibility()
        logger.info("User logged out.")
        # Reset system state and clear data on logout
        self.update_system_state(SystemState.IDLE)
        self.data_analysis = DataAnalysis() # Clear old run data
        self.chromatogram.data_analysis = self.data_analysis # Ensure plot uses new data analysis instance
        self.chromatogram.update_plot([], [], []) # Clear plot
        self.chromatogram.update_realtime_data_labels(0, 0, 0, 0, 0, 0, 0, 0) # Reset labels
        self.fplc_controller.stop_run() # Ensure controller is stopped

    def new_method(self):
        """Resets the method editor to a new, empty method."""
        if not self.user_manager.check_permission('operator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to create a new method.")
            logger.warning(f"Permission denied for {self.current_user} to create new method.")
            return

        reply = QMessageBox.question(self, "New Method",
                                     "Are you sure you want to create a new method? Unsaved changes will be lost.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            new_method_obj = FPLCMethod()
            self.method_editor.load_method_into_ui(new_method_obj)
            self.fplc_controller.method = new_method_obj # Update controller's method
            self.status_bar.showMessage("New method created.")
            logger.info("New method created.")

    def open_method(self):
        """Opens an existing method file and populates the UI."""
        if not self.user_manager.check_permission('operator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to open methods.")
            logger.warning(f"Permission denied for {self.current_user} to open method.")
            return

        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Method", "", "FPLC Method Files (*.json);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    method_json = f.read()
                method = FPLCMethod.from_json(method_json)
                self.method_editor.load_method_into_ui(method)
                self.fplc_controller.method = method # Update controller's method
                self.status_bar.showMessage(f"Method '{method.name}' loaded.")
                logger.info(f"Method '{method.name}' loaded from {filename}.")
            except FileNotFoundError:
                QMessageBox.critical(self, "File Error", "Selected file not found.")
                logger.error(f"File not found during open method: {filename}")
            except json.JSONDecodeError:
                QMessageBox.critical(self, "File Error", "Invalid method file format (not a valid JSON).")
                logger.error(f"Invalid JSON format for method file: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while opening the method: {str(e)}")
                logger.error(f"Error opening method {filename}: {str(e)}")


    def save_method(self):
        """Saves the current method to a file."""
        if not self.user_manager.check_permission('operator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to save methods.")
            logger.warning(f"Permission denied for {self.current_user} to save method.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Method", self.method_editor.method.name + ".json", "FPLC Method Files (*.json)"
        )
        if filename:
            try:
                method_json = self.method_editor.method.to_json()
                with open(filename, 'w') as f:
                    f.write(method_json)
                self.status_bar.showMessage(f"Method saved to: {filename}")
                logger.info(f"Method '{self.method_editor.method.name}' saved to {filename}.")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save method: {str(e)}")
                logger.error(f"Error saving method {filename}: {str(e)}")

    def load_standards(self):
        """Loads chromatography standards and integrates with data analysis."""
        if not self.user_manager.check_permission('operator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to load standards.")
            logger.warning(f"Permission denied for {self.current_user} to load standards.")
            return

        filename, _ = QFileDialog.getOpenFileName(self, "Open Standards File", "", "JSON Files (*.json)")
        if filename:
            self.data_analysis.load_standards(filename)
            # The peak_analyzer's standards are set within data_analysis.load_standards
            QMessageBox.information(self, "Standards Loaded", "Chromatography standards loaded successfully.")
            logger.info(f"Standards loaded from {filename}.")

    def run_diagnostics(self):
        """Runs system diagnostics."""
        if not self.user_manager.check_permission('administrator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to run diagnostics.")
            logger.warning(f"Permission denied for {self.current_user} to run diagnostics.")
            return

        self.status_bar.showMessage("Running system diagnostics...")
        logger.info("Running system diagnostics.")
        try:
            result = self.diagnostics.run_diagnostics()
            if result:
                QMessageBox.information(self, "Diagnostics", "All diagnostics passed.")
                self.status_bar.showMessage("Diagnostics passed.")
                logger.info("All diagnostics passed.")
            else:
                QMessageBox.warning(self, "Diagnostics", "Some diagnostics failed. Check logs for details.")
                self.status_bar.showMessage("Diagnostics failed.")
                logger.warning("Some diagnostics failed.")
        except ConnectionError as ce:
            QMessageBox.critical(self, "Diagnostics Error", f"Hardware not connected: {str(ce)}")
            self.status_bar.showMessage("Diagnostics failed: Hardware not connected.")
            logger.error(f"Diagnostics failed due to hardware connection: {str(ce)}")
        except Exception as e:
            QMessageBox.critical(self, "Diagnostics Error", f"An error occurred during diagnostics: {str(e)}")
            self.status_bar.showMessage("Diagnostics failed due to error.")
            logger.error(f"Error during diagnostics: {str(e)}")


    def export_logs(self):
        """Exports system logs to a file."""
        if not self.user_manager.check_permission('administrator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to export logs.")
            logger.warning(f"Permission denied for {self.current_user} to export logs.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "fplc_log.txt", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                shutil.copy2("fplc_log.log", filename)
                QMessageBox.information(self, "Export Logs", f"Logs exported to: {filename}")
                logger.info(f"Logs exported to {filename}.")
            except FileNotFoundError:
                QMessageBox.critical(self, "Export Error", "Log file not found.")
                logger.error("Log file 'fplc_log.log' not found for export.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export logs: {str(e)}")
                logger.error(f"Error exporting logs to {filename}: {str(e)}")

    def open_documentation(self):
        try:
            from docs.FPLC_USER_GUIDE import DocumentationContent
            # Create documentation instance
            fplc_docs = DocumentationContent()
        
            # Generate sample documentation
            sections = {
                "User Guide": fplc_docs.get_user_guide(),
                "API Documentation": fplc_docs.get_api_docs(),
                "Maintenance Manual": fplc_docs.get_maintenance_manual(),
                "Method Development": fplc_docs.get_method_development_guide(),
                "Quality Control": fplc_docs.get_quality_control_guide(),
                "Regulatory Compliance": fplc_docs.get_regulatory_compliance_guide()
            }
        
            # Create a scrollable dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("FPLC Documentation")
            dialog.setModal(True)
            dialog.resize(1000, 700)
        
            layout = QVBoxLayout()
        
            # Create tab widget
            tab_widget = QTabWidget()
        
            # Add each section as a separate tab
            for section_name, content in sections.items():
                text_widget = QTextEdit()
                text_widget.setPlainText(content)
                text_widget.setReadOnly(True)
                text_widget.setFont(QFont("Consolas", 10))  # Monospace font for better readability
                tab_widget.addTab(text_widget, section_name)
        
            layout.addWidget(tab_widget)
        
            # Button layout
            button_layout = QHBoxLayout()
        
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.close)
        
            # Optional: Add export button
            export_button = QPushButton("Export All")
            export_button.clicked.connect(lambda: self.export_documentation(sections))
        
            button_layout.addWidget(export_button)
            button_layout.addStretch()
            button_layout.addWidget(close_button)
        
            layout.addLayout(button_layout)
        
            dialog.setLayout(layout)
            dialog.exec()
        
            logger.info("Documentation opened successfully.")
        
        except ImportError:
            QMessageBox.information(self, "Documentation", "Documentation files not found.")
            logger.warning("Documentation files missing.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open documentation: {str(e)}")
            logger.error(f"Error opening documentation: {str(e)}")

    def export_documentation(self, sections):
        """Exports all documentation sections to a selected directory."""
        export_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Export Documentation")
        if not export_dir:
            self.logger.info("Documentation export cancelled.")
            return

        try:
            for section_name, content in sections.items():
                # Sanitize section name for filename
                filename = "".join(c for c in section_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
                file_path = os.path.join(export_dir, f"{filename.replace(' ', '_')}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Exported documentation section '{section_name}' to {file_path}")
            QMessageBox.information(self, "Export Complete", f"All documentation exported to:\n{export_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export documentation: {str(e)}")
            self.logger.error(f"Error exporting documentation: {str(e)}")

    def show_about(self):
        """Shows information about the application."""
        QMessageBox.about(
            self,
            "About",
            "Industrial FPLC Control System\nVersion 1.0\nDeveloped by Sagnik Mitra @ IIT Indore"
        )
        logger.info("About dialog shown.")

    def setup_timers(self):
        """Initializes system timers."""
        # Data timer (for receiving data from serial_manager)
        # The serial_manager's internal thread now emits data_received directly.
        # So, this timer is no longer needed for data acquisition.
        # It's now handled by serial_manager.data_received signal.

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_system_status)
        self.status_timer.setInterval(1000) # 1s interval for system checks
        self.status_timer.start()
        logger.info("System timers set up.")

    def setup_signals(self):
        """Connects system signals."""
        # Connect signals from FPLCSerialManager
        self.serial_manager.data_received.connect(self.on_data_received)
        self.serial_manager.connection_status_changed.connect(self.on_connection_status_changed)
        self.serial_manager.error_occurred.connect(lambda msg: self.alarm_manager.add_alarm("Communication Error", msg))

        # Connect signals from FPLCController
        self.fplc_controller.current_step_changed.connect(self.update_current_step_display)
        self.fplc_controller.run_progress_updated.connect(self.run_progress_bar.setValue)
        self.fplc_controller.run_finished.connect(self.handle_run_finished)
        self.fplc_controller.control_data_updated.connect(self.update_mpc_status_display)
        self.fplc_controller.error_occurred.connect(lambda msg: self.alarm_manager.add_alarm("Control Error", msg))

        # Connect UI buttons to FPLCController methods
        self.start_button.clicked.connect(self.start_run)
        self.pause_button.clicked.connect(self.pause_run)
        self.stop_button.clicked.connect(self.stop_run)
        self.emergency_stop.clicked.connect(self.emergency_stop_run)
        self.load_standards_button.clicked.connect(self.load_standards)

        # Connect alarm manager to main window's alarm handler
        self.alarm_manager.alarm_triggered.connect(self.handle_alarm)
        logger.info("System signals connected.")

    def check_system_status(self):
        """Periodic system status check."""
        try:
            if not self.serial_manager.connected:
                if self.system_state != SystemState.ERROR: # Avoid re-logging if already in error
                    self.update_system_state(SystemState.ERROR)
                    self.status_bar.showMessage("Hardware connection lost!")
                    self.alarm_manager.add_alarm("Critical", "Hardware connection lost. Please check connections.")
                return

            # Check system pressure against method's max pressure
            if self.system_state == SystemState.RUNNING:
                max_pressure_limit = self.method_editor.method.max_pressure
                current_pressure = self.serial_manager._sim_pressure # Get current simulated pressure
                if max_pressure_limit > 0 and current_pressure > max_pressure_limit:
                    self.alarm_manager.add_alarm(
                        "Critical",
                        f"Pressure exceeded maximum limit ({max_pressure_limit:.2f} MPa): {current_pressure:.2f} MPa"
                    )
                    self.emergency_stop_run() # Trigger emergency stop on critical pressure

        except Exception as e:
            logger.error(f"System status check error: {str(e)}")
            self.update_system_state(SystemState.ERROR)
            self.alarm_manager.add_alarm("Error", f"System status check failed: {str(e)}")


    def initialize_system(self):
        """Initializes system components and performs startup checks."""
        self.status_bar.showMessage("Initializing system...")
        logger.info("Initializing system components.")
        try:
            # Initialize hardware first
            if not self.serial_manager.connect():
                QMessageBox.critical(
                    self,
                    "Hardware Error",
                    "Failed to connect to FPLC hardware. Please check connections and restart."
                )
                self.update_system_state(SystemState.ERROR)
                logger.critical("Hardware initialization failed. System in ERROR state.")
                return

            # Run system diagnostics
            if not self.diagnostics.run_diagnostics():
                QMessageBox.warning(
                    self,
                    "System Warning",
                    "System diagnostics detected issues. Check logs for details."
                )
                logger.warning("System diagnostics reported issues.")

            # Load default buffer recipes
            self.load_default_buffer_recipes()

            # Set initial system state
            self.update_system_state(SystemState.IDLE)

            self.status_bar.showMessage("System initialized successfully.")
            logger.info("System initialized successfully.")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Failed to initialize system: {str(e)}"
            )
            logger.critical(f"Initialization Error: {str(e)}")
            self.update_system_state(SystemState.ERROR)

    def update_system_state(self, new_state):
        """Updates system state and UI elements."""
        self.system_state = new_state
        self.system_status.setText(f"System Status: {new_state.value}")

        # Update UI elements based on state and user permissions
        is_logged_in = (self.current_user is not None)
        can_start = is_logged_in and (new_state == SystemState.IDLE)
        can_pause = is_logged_in and (new_state == SystemState.RUNNING)
        can_stop = is_logged_in and (new_state in [SystemState.RUNNING, SystemState.PAUSED])

        self.start_button.setEnabled(can_start)
        self.pause_button.setEnabled(can_pause)
        self.stop_button.setEnabled(can_stop)
        self.emergency_stop.setEnabled(is_logged_in) # Always enable if logged in

        logger.info(f"System state changed to: {new_state.value}")
        self.update_menu_visibility() # Re-evaluate button states based on new system state

    def on_data_received(self, time_point, uv, pressure, conductivity, flow_rate, buffer_a, buffer_b, temperature, ph):
        """Handles received data from hardware, updates UI and analysis."""
        try:
            # Add data to analysis
            self.data_analysis.add_data_point(
                time_point, uv, pressure, conductivity,
                flow_rate, buffer_a, buffer_b
            )

            # Update real-time data displays on chromatogram tab
            self.chromatogram.update_realtime_data_labels(uv, pressure, conductivity, flow_rate, buffer_a, buffer_b, temperature, ph)

            # Update chromatogram plot
            self.chromatogram.update_plot(
                self.data_analysis.time_data,
                self.data_analysis.uv_data,
                self.data_analysis.conductivity_data
            )

            # Check for fraction collection
            if self.fraction_collector.is_collecting:
                # Assuming 0.1 min per data point, flow_rate is in mL/min
                volume_increment = flow_rate * (self.serial_manager.control_timer.interval() / 1000.0 / 60.0) # mL per interval
                self.fraction_collector.update(
                    time_point,
                    uv,
                    volume_increment
                )

        except Exception as e:
            logger.error(f"Error processing data in on_data_received: {str(e)}")
            self.fplc_controller.stop_run() # Stop run on data processing error
            self.update_system_state(SystemState.ERROR)
            self.alarm_manager.add_alarm("Error", f"Data processing error: {str(e)}")

    def start_run(self):
        """Starts a new chromatography run."""
        if not self.user_manager.check_permission('operator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to start a run.")
            logger.warning(f"Permission denied for {self.current_user} to start run.")
            return

        if not self.validate_run_parameters():
            return
        
        try:
            current_method_steps = self.get_current_fplc_method_steps() # You need to implement this
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to retrieve method steps: {e}")
            return

        # 2. Validate the method using the MethodValidationEngine
        validation_results = self.method_validator_engine.validate_method(current_method_steps) 

        # 3. Handle validation results
        if validation_results:
            error_message = "Method validation failed:\n" + "\n".join(validation_results)
            QMessageBox.warning(self, "Validation Error", error_message)
            logging.warning(f"FPLC method validation failed: {validation_results}")
            return # Prevent the run if validation fails
        else:
            logging.info("FPLC method validated successfully. Starting run...")
            # Proceed with the FPLC run
            self.execute_fplc_method(current_method_steps) # You need to implement this

        try:
            # Generate unique run ID
            self.current_run_data['run_id'] = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_run_data['start_time'] = datetime.now()
            self.current_run_data['method_name'] = self.method_editor.method.name

            # Reset data analysis for a new run
            self.data_analysis = DataAnalysis()
            self.data_analysis.peak_analyzer.set_standards(self.data_analysis.standards) # Re-apply standards if loaded
            self.chromatogram.data_analysis = self.data_analysis # Ensure plot uses new data analysis instance
            self.chromatogram.update_plot([], [], []) # Clear plot for new run
            self.chromatogram.update_realtime_data_labels(0, 0, 0, 0, 0, 0, 0, 0) # Reset labels
            self.run_progress_bar.setValue(0) # Reset progress bar

            # Pass the current method to the FPLCController
            self.fplc_controller.method = self.method_editor.method
            self.fplc_controller.start_run() # Start the controller's run logic

            # Update system state
            self.update_system_state(SystemState.RUNNING)

            logger.info(f"Started run: {self.current_run_data['run_id']}")
            self.status_bar.showMessage(f"Run '{self.current_run_data['run_id']}' started successfully.")

        except ConnectionError as ce:
            QMessageBox.critical(self, "Run Error", f"Hardware not connected: {str(ce)}")
            logger.error(f"Run start error (hardware not connected): {str(ce)}")
            self.update_system_state(SystemState.ERROR)
        except Exception as e:
            QMessageBox.critical(self, "Run Error", f"Failed to start run: {str(e)}")
            logger.error(f"Run start error: {str(e)}")
            self.update_system_state(SystemState.ERROR)


    def pause_run(self):
        """Pauses the current run."""
        if not self.user_manager.check_permission('operator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to pause a run.")
            logger.warning(f"Permission denied for {self.current_user} to pause run.")
            return

        try:
            self.fplc_controller.pause_run()
            self.update_system_state(SystemState.PAUSED)
            logger.info("Run paused.")
            self.status_bar.showMessage("Run paused.")
        except Exception as e:
            QMessageBox.critical(self, "Pause Error", f"Failed to pause run: {str(e)}")
            logger.error(f"Run pause error: {str(e)}")

    def stop_run(self):
        """Stops the current run."""
        if not self.user_manager.check_permission('operator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to stop a run.")
            logger.warning(f"Permission denied for {self.current_user} to stop run.")
            return

        try:
            self.fplc_controller.stop_run() # This will emit run_finished

        except Exception as e:
            QMessageBox.critical(self, "Stop Error", f"Failed to stop run: {str(e)}")
            logger.error(f"Run stop error: {str(e)}")

    def emergency_stop_run(self):
        """Performs emergency stop."""
        if not self.user_manager.check_permission('operator'): # Emergency stop should be accessible to operators
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to perform an emergency stop.")
            logger.warning(f"Permission denied for {self.current_user} to emergency stop.")
            return

        try:
            self.fplc_controller.emergency_stop() # This will stop everything and emit error
            self.update_system_state(SystemState.ERROR)

            msg = "EMERGENCY STOP ACTIVATED! System requires immediate attention."
            logger.critical(msg)
            self.status_bar.showMessage(msg)
            self.alarm_manager.add_alarm("Critical", msg)

            QMessageBox.critical(
                self,
                "EMERGENCY STOP",
                "Emergency stop activated. System must be checked by an administrator before resuming operation."
            )
            # Clear current run data
            self.current_run_data = {
                'run_id': None,
                'start_time': None,
                'method_name': None
            }
            self.data_analysis = DataAnalysis() # Reset data analysis
            self.chromatogram.data_analysis = self.data_analysis
            self.chromatogram.update_plot([], [], [])
            self.chromatogram.update_realtime_data_labels(0, 0, 0, 0, 0, 0, 0, 0)
            self.run_progress_bar.setValue(0)

        except Exception as e:
            logger.critical(f"Emergency stop error: {str(e)}")
            QMessageBox.critical(self, "Emergency Stop Error", f"Error during emergency stop: {str(e)}")

    def handle_run_finished(self):
        """Called when FPLCController signals run completion."""
        self.update_system_state(SystemState.IDLE)
        logger.info("Run finished normally.")
        self.status_bar.showMessage("Run completed.")

        # Save run data if a run was active
        if self.current_run_data['run_id']:
            self.save_run_data()
            self.prompt_data_export() # Prompt for export after saving

        # Clear current run data
        self.current_run_data = {
            'run_id': None,
            'start_time': None,
            'method_name': None
        }
        self.data_analysis = DataAnalysis() # Reset data analysis for next run
        self.chromatogram.data_analysis = self.data_analysis
        self.chromatogram.update_plot([], [], []) # Clear plot
        self.chromatogram.update_realtime_data_labels(0, 0, 0, 0, 0, 0, 0, 0) # Reset labels
        self.run_progress_bar.setValue(0)

    def update_current_step_display(self, step_type, step_index, elapsed_time_in_step):
        """Updates the current step display labels."""
        self.current_step_label.setText(f"Current Step: {step_type} (Step {step_index})")
        self.step_time_label.setText(f"Step Time: {elapsed_time_in_step:.2f} min")

    def update_mpc_status_display(self, set_flow, actual_flow, set_buffer_a, actual_buffer_a):
        """Updates the MPC control status display."""
        self.set_flow_label.setText(f"Set Flow: {set_flow:.2f} mL/min")
        self.actual_flow_label.setText(f"Actual Flow: {actual_flow:.2f} mL/min")
        self.set_buffer_a_label.setText(f"Set Buffer A: {set_buffer_a:.1f}%")
        self.actual_buffer_a_label.setText(f"Actual Buffer A: {actual_buffer_a:.1f}%")

    def handle_alarm(self, severity, message):
        """Handles system alarms by displaying a message box."""
        if severity == "Critical":
            # Critical alarms might trigger emergency stop, already handled in check_system_status or fplc_controller
            pass
        QMessageBox.warning(
            self,
            f"{severity} Alarm",
            message
        )
        logger.warning(f"Alarm handled: {severity} - {message}")

    def on_connection_status_changed(self, is_connected: bool):
        """Handles changes in hardware connection status."""
        if is_connected:
            self.status_bar.showMessage("Hardware connected.")
            logger.info("Hardware connection established.")
            if self.system_state == SystemState.ERROR: # If previously in error due to connection, go to idle
                self.update_system_state(SystemState.IDLE)
        else:
            self.status_bar.showMessage("Hardware disconnected.")
            logger.warning("Hardware connection lost.")
            self.update_system_state(SystemState.ERROR)
            self.alarm_manager.add_alarm("Critical", "Hardware connection lost!")

    def save_run_data(self):
        """Saves run data to a dedicated run directory."""
        if not self.current_run_data['run_id']:
            logger.warning("Attempted to save run data but no active run ID found.")
            return

        try:
            run_dir = os.path.join(
                "run_data",
                self.current_run_data['run_id']
            )
            os.makedirs(run_dir, exist_ok=True)
            logger.info(f"Created run data directory: {run_dir}")

            method_file = os.path.join(run_dir, "method.json")
            with open(method_file, 'w') as f:
                f.write(self.method_editor.method.to_json())
            logger.info(f"Method saved to {method_file}.")

            data_file = os.path.join(run_dir, "raw_data.csv")
            self.data_analysis.export_data(data_file)
            logger.info(f"Raw data saved to {data_file}.")

            peaks_file = os.path.join(run_dir, "peaks.csv")
            self.data_analysis.export_peaks(peaks_file)
            logger.info(f"Peak data saved to {peaks_file}.")

            report_file = os.path.join(run_dir, "report.html")
            self.data_analysis.generate_report(
                self.method_editor.method,
                self.current_run_data,
                report_file
            )
            logger.info(f"Run report generated to {report_file}.")

            self.status_bar.showMessage(f"Run data saved to: {run_dir}")
            logger.info(f"All run data for {self.current_run_data['run_id']} saved successfully.")

        except Exception as e:
            logger.error(f"Error saving run data for {self.current_run_data['run_id']}: {str(e)}")
            QMessageBox.warning(
                self,
                "Save Error",
                f"Error saving run data: {str(e)}\nCheck logs for details."
            )

    def prompt_data_export(self):
        """Prompts user for additional data export options after a run."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Run Data")
        layout = QVBoxLayout()

        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout()

        export_options = [
            ("Raw Data (CSV)", "raw_data"),
            ("Peak Analysis (CSV)", "peaks"),
            ("Method (JSON)", "method"),
            ("Report (HTML)", "report"),
            ("System Log (Copy)", "log")
        ]

        checkboxes = {}
        for label, key in export_options:
            cb = QCheckBox(label)
            cb.setChecked(True)
            checkboxes[key] = cb
            options_layout.addWidget(cb)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        location_group = QGroupBox("Export Location")
        location_layout = QHBoxLayout()
        location_edit = QLineEdit()
        # Set default export location to the current run's data directory
        if self.current_run_data['run_id']:
            location_edit.setText(os.path.join(os.getcwd(), "run_data", self.current_run_data['run_id']))
        else:
            location_edit.setText(os.path.join(os.getcwd(), "exports")) # Default if no run ID

        browse_btn = QPushButton("Browse")

        def browse_location():
            dir_path = QFileDialog.getExistingDirectory(
                dialog, "Select Export Directory", location_edit.text()
            )
            if dir_path:
                location_edit.setText(dir_path)

        browse_btn.clicked.connect(browse_location)
        location_layout.addWidget(location_edit)
        location_layout.addWidget(browse_btn)
        location_group.setLayout(location_layout)
        layout.addWidget(location_group)

        export_btn = QPushButton("Export")
        layout.addWidget(export_btn)

        def do_export():
            export_dir = location_edit.text()
            if not export_dir:
                QMessageBox.warning(dialog, "Export Error", "Please select an export location.")
                logger.warning("Export cancelled: No export directory selected.")
                return

            try:
                os.makedirs(export_dir, exist_ok=True)
                logger.info(f"Ensured export directory exists: {export_dir}")

                success_count = 0
                total_exports = 0

                if checkboxes["raw_data"].isChecked():
                    total_exports += 1
                    if self.data_analysis.export_data(os.path.join(export_dir, "raw_data.csv")):
                        success_count += 1

                if checkboxes["peaks"].isChecked():
                    total_exports += 1
                    if self.data_analysis.export_peaks(os.path.join(export_dir, "peaks.csv")):
                        success_count += 1

                if checkboxes["method"].isChecked():
                    total_exports += 1
                    try:
                        with open(os.path.join(export_dir, "method.json"), 'w') as f:
                            f.write(self.method_editor.method.to_json())
                        success_count += 1
                        logger.info(f"Method exported to {os.path.join(export_dir, 'method.json')}.")
                    except Exception as e:
                        logger.error(f"Error exporting method: {str(e)}")

                if checkboxes["report"].isChecked():
                    total_exports += 1
                    if self.data_analysis.generate_report(
                        self.method_editor.method,
                        self.current_run_data,
                        os.path.join(export_dir, "report.html")
                    ):
                        success_count += 1

                if checkboxes["log"].isChecked():
                    total_exports += 1
                    try:
                        shutil.copy2("fplc_log.log", os.path.join(export_dir, "system.log"))
                        success_count += 1
                        logger.info(f"System log copied to {os.path.join(export_dir, 'system.log')}.")
                    except FileNotFoundError:
                        logger.error("Log file 'fplc_log.log' not found for export.")
                    except Exception as e:
                        logger.error(f"Error copying log file: {str(e)}")

                if success_count == total_exports:
                    QMessageBox.information(
                        dialog,
                        "Export Complete",
                        f"All selected data exported successfully to:\n{export_dir}"
                    )
                    logger.info(f"All selected data exported successfully to {export_dir}.")
                elif success_count > 0:
                    QMessageBox.warning(
                        dialog,
                        "Export Warning",
                        f"{success_count} of {total_exports} items exported. Some exports failed. Check the log for details."
                    )
                    logger.warning(f"Partial export success: {success_count}/{total_exports} items exported.")
                else:
                    QMessageBox.critical(
                        dialog,
                        "Export Failed",
                        "No data was exported. Check the log for details."
                    )
                    logger.error("No data exported during prompt_data_export.")

                dialog.accept()

            except Exception as e:
                QMessageBox.critical(
                    dialog,
                    "Export Error",
                    f"An unexpected error occurred during export: {str(e)}"
                )
                logger.critical(f"Unexpected error during prompt_data_export: {str(e)}")

        export_btn.clicked.connect(do_export)
        dialog.setLayout(layout)
        dialog.exec()

    def validate_run_parameters(self):
        """Validates all run parameters before starting."""
        if not self.current_user:
            QMessageBox.warning(self, "Validation Error", "User not logged in.")
            logger.warning("Run validation failed: User not logged in.")
            return False

        if not self.serial_manager.connected:
            QMessageBox.warning(self, "Validation Error", "Hardware not connected. Please connect hardware first.")
            logger.warning("Run validation failed: Hardware not connected.")
            return False

        if not self.method_editor.method.steps:
            QMessageBox.warning(self, "Validation Error", "No method steps defined. Please add steps to the method.")
            logger.warning("Run validation failed: No method steps defined.")
            return False

        if not self.method_editor.method.name:
            QMessageBox.warning(self, "Validation Error", "Method Name is empty. Please provide a method name.")
            logger.warning("Run validation failed: Method name is empty.")
            return False

        # Add more comprehensive validation here (e.g., buffer concentrations, flow rates within limits)
        for i, step in enumerate(self.method_editor.method.steps):
            if step['duration'] <= 0:
                QMessageBox.warning(self, "Validation Error", f"Step {i+1}: Duration must be greater than 0.")
                logger.warning(f"Run validation failed: Step {i+1} has invalid duration.")
                return False
            if step['flow_rate'] <= 0:
                QMessageBox.warning(self, "Validation Error", f"Step {i+1}: Flow Rate must be greater than 0.")
                logger.warning(f"Run validation failed: Step {i+1} has invalid flow rate.")
                return False
            if step['buffer_a'] + step['buffer_b'] != 100:
                 QMessageBox.warning(self, "Validation Error", f"Step {i+1}: Buffer A and Buffer B percentages must sum to 100%.")
                 logger.warning(f"Run validation failed: Step {i+1} buffer percentages do not sum to 100%.")
                 return False

        logger.info("Run parameters validated successfully.")
        return True

    def load_default_buffer_recipes(self):
        """Loads default buffer recipes into the buffer blending manager."""
        default_recipes = {
            "PBS": {
                "components": {
                    "NaCl": "137 mM",
                    "KCl": "2.7 mM",
                    "Na2HPO4": "10 mM",
                    "KH2PO4": "1.8 mM"
                }
            },
            "Tris": {
                "components": {
                    "Tris": "50 mM",
                    "NaCl": "150 mM",
                    "pH": "7.5"
                }
            }
        }
        for name, recipe in default_recipes.items():
            self.buffer_blending.add_buffer_recipe(name, recipe["components"])
        logger.info("Default buffer recipes loaded.")

    def add_logos(self):
        """Adds two placeholder logos to the UI."""
        # Logo 1
        logo1_label = QLabel(self)
        # Replace 'path/to/your/logo1.png' with the actual path to your first logo image
        # Make sure the image file is accessible from where the script is run
        logo1_pixmap = QPixmap(r"C:\Users\sagni\OneDrive - IIT Indore\Current and all folders\Course Registration_files\iiti.png")
        if not logo1_pixmap.isNull():
            logo1_label.setPixmap(logo1_pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            logo1_label.setText("Logo 1 Placeholder")
            logo1_label.setStyleSheet("border: 1px solid gray;")
        logo1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Logo 2
        logo2_label = QLabel(self)
        # Replace 'path/to/your/logo2.png' with the actual path to your second logo image
        # Make sure the image file is accessible from where the script is run
        logo2_pixmap = QPixmap(r"C:\Users\sagni\OneDrive - IIT Indore\Current and all folders\Personal Details\Sag95.jpg")
        if not logo2_pixmap.isNull():
            logo2_label.setPixmap(logo2_pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            logo2_label.setText("Logo 2 Placeholder")
            logo2_label.setStyleSheet("border: 1px solid gray;")
        logo2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create a layout for the logos
        logo_layout = QHBoxLayout()
        logo_layout.addStretch(1) # Pushes logos to center/right
        logo_layout.addWidget(logo1_label)
        logo_layout.addSpacing(20) # Space between logos
        logo_layout.addWidget(logo2_label)
        logo_layout.addStretch(1) # Pushes logos to center/left

        return logo_layout

def main():
    """Main entry point for the FPLC application."""
    app = QApplication(sys.argv)
    user_manager = UserManager()
    # --- Add Login Check Here ---
    login_dialog = LoginDialog(user_manager)
    # Show the login dialog and wait for user interaction
    if login_dialog.exec() == QDialog.DialogCode.Accepted:
        # If login is successful, proceed to launch the main application
        logging.info("User successfully logged in.")
        window = IndustrialFPLCSystem(user_manager) # Assuming IndustrialFPLCSystem is your main window class
        window.showMaximized() # Or window.show()
        sys.exit(app.exec())
    else:
        # If login failed or was cancelled (e.g., user closed the dialog)
        logging.warning("Login failed or cancelled. Exiting application.")
        sys.exit(1) # Exit the application

if __name__ == "__main__":
    main()
        