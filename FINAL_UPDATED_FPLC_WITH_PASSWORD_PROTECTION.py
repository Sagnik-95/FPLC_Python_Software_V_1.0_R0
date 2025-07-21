"""
Fast Protein Liquid Chromatography (FPLC) Control Software
A comprehensive software suite for protein purification workflows

Author: Sagnik Mitra
License: IIT INDORE
"""
import sys
import numpy as np
from datetime import datetime
from scipy.signal import find_peaks
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTextEdit,
    QTableWidgetItem, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QLabel, QWidget, QDialog, QFormLayout, QTabWidget, QRadioButton, 
    QMenuBar, QStatusBar, QComboBox, QMessageBox, QFileDialog, QCheckBox,
    QApplication, QInputDialog
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
import shutil
import hashlib # For hashing passwords

from collections import deque
import random
from enum import Enum
import threading
import time

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
    duration: float
    flow_rate: float
    buffer_a: float
    buffer_b: float
    collection: bool = False
    notes: str = ""

class SystemState(Enum):
    """Enumeration of possible system states."""
    IDLE = "Idle"
    RUNNING = "Running"
    PAUSED = "Paused"
    ERROR = "Error"
    MAINTENANCE = "Maintenance"
    CALIBRATING = "Calibrating"
    WASHING = "Washing"
    EQUILIBRATING = "Equilibrating"

# --- FPLC Core Logic Classes ---
class FPLCMethod:
    """Represents a complete FPLC method with all parameters and steps."""
    def __init__(self):
        self.name = ""
        self.description = ""
        self.author = ""
        self.created_date = datetime.now()
        self.modified_date = datetime.now()
        self.column_type = ""
        self.column_volume = 0.0
        self.max_pressure = 0.0
        self.steps = []
        self.sample_info = {}
        self.buffer_info = {
            'A': {'name': '', 'composition': ''},
            'B': {'name': '', 'composition': ''}
        }

    def add_step(self, step_type, duration, flow_rate, buffer_a, buffer_b,
                 collection=False, notes=""):
        """Adds a new step to the method."""
        step = {
            'type': step_type,
            'duration': duration,
            'flow_rate': flow_rate,
            'buffer_a': buffer_a,
            'buffer_b': buffer_b,
            'collection': collection,
            'notes': notes
        }
        self.steps.append(step)
        self.modified_date = datetime.now()
        logger.info(f"Added step '{step_type}' to method '{self.name}'.")

    def to_json(self):
        """Converts the method object to a JSON string."""
        return json.dumps({
            'name': self.name,
            'description': self.description,
            'author': self.author,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'column_type': self.column_type,
            'column_volume': self.column_volume,
            'max_pressure': self.max_pressure,
            'steps': self.steps,
            'sample_info': self.sample_info,
            'buffer_info': self.buffer_info
        }, indent=4)

    @classmethod
    def from_json(cls, json_str):
        """Creates an FPLCMethod object from a JSON string."""
        data = json.loads(json_str)
        method = cls()
        method.name = data.get('name', '')
        method.description = data.get('description', '')
        method.author = data.get('author', '')
        method.created_date = datetime.fromisoformat(data.get('created_date', datetime.now().isoformat()))
        method.modified_date = datetime.fromisoformat(data.get('modified_date', datetime.now().isoformat()))
        method.column_type = data.get('column_type', '')
        method.column_volume = data.get('column_volume', 0.0)
        method.max_pressure = data.get('max_pressure', 0.0)
        method.steps = data.get('steps', [])
        method.sample_info = data.get('sample_info', {})
        method.buffer_info = data.get('buffer_info', {'A': {'name': '', 'composition': ''}, 'B': {'name': '', 'composition': ''}})
        logger.info(f"Loaded method '{method.name}' from JSON.")
        return method

class HardwareCommunication:
    """Handles simulated serial communication with FPLC hardware."""
    def __init__(self, port='COM3', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False

    def connect(self):
        """Establishes a simulated serial connection."""
        try:
            # Simulate connection success/failure
            if random.random() > 0.1: # 90% chance of success
                self.connected = True
                logger.info(f"Simulated hardware connected on {self.port}.")
                return True
            else:
                raise IOError("Simulated connection failed.")
        except Exception as e:
            logger.error(f"Serial connection error: {str(e)}")
            self.connected = False
            return False

    def disconnect(self):
        """Closes the simulated serial connection."""
        if self.connected:
            self.connected = False
            logger.info("Simulated hardware disconnected.")

    def send_command(self, command):
        """Sends a simulated command to hardware and receives a simulated response."""
        if not self.connected:
            raise ConnectionError("Not connected to hardware.")

        logger.debug(f"Sending command: {command}")
        # Simulate hardware response
        if "FC_START" in command:
            return "FC_STARTED"
        elif "FC_STOP" in command:
            return "FC_STOPPED"
        elif "FC_RESET" in command:
            return "FC_RESET_OK"
        return "OK"

class ColumnQualification:
    """Handles column qualification testing and performance monitoring."""
    def __init__(self):
        self.qualification_data = {}
        self.acceptance_criteria = {
            'theoretical_plates': 2000,  # Minimum number of theoretical plates
            'asymmetry': 1.5,           # Maximum peak asymmetry
            'peak_capacity': 40,        # Minimum peak capacity
            'resolution': 1.5,          # Minimum resolution between peaks
            'pressure_drop': 2.0        # Maximum pressure drop (MPa)
        }

    def run_qualification(self, column_type, test_mixture="standard"):
        """Runs a simulated column qualification test."""
        try:
            test_data = {
                'date': datetime.now(),
                'column_type': column_type,
                'test_mixture': test_mixture,
                'results': {},
                'passed': False
            }

            # Simulate running specific tests based on column type
            if column_type == "SEC":
                test_data['results'] = self._run_sec_qualification()
            elif column_type == "IEX":
                test_data['results'] = self._run_iex_qualification()
            elif column_type == "HIC":
                test_data['results'] = self._run_hic_qualification()
            else:
                test_data['results'] = self._run_generic_qualification()

            test_data['passed'] = self._evaluate_results(test_data['results'])
            self.qualification_data[str(test_data['date'])] = test_data
            logger.info(f"Column qualification completed for {column_type}. Passed: {test_data['passed']}")
            return test_data

        except Exception as e:
            logger.error(f"Column qualification error: {str(e)}")
            raise

    def calculate_hetp(self, flow_rates):
        """Calculates simulated HETP values for given flow rates."""
        A = 0.5  # Example constant
        B = 0.1  # Example constant
        hetp_values = [(A + B / flow_rate) for flow_rate in flow_rates if flow_rate > 0]
        return hetp_values

    def calculate_symmetry(self, flow_rates):
        """Calculates simulated symmetry values for given flow rates."""
        # This is a simplified simulation. Real calculation is more complex.
        symmetry_values = [random.uniform(0.8, 1.2) for _ in flow_rates]
        return symmetry_values

    def _run_sec_qualification(self):
        """Simulates SEC-specific qualification tests."""
        return {
            'theoretical_plates': self._calculate_theoretical_plates(),
            'asymmetry': self._calculate_peak_asymmetry(),
            'resolution': self._calculate_resolution(),
            'pressure_drop': self._measure_pressure_drop(),
            'void_volume': self._determine_void_volume(),
            'calibration_curve': self._generate_calibration_curve(),
            'peak_capacity': self._measure_peak_capacity()
        }

    def _run_iex_qualification(self):
        """Simulates IEX-specific qualification tests."""
        return {
            'theoretical_plates': self._calculate_theoretical_plates(),
            'asymmetry': self._calculate_peak_asymmetry(),
            'resolution': self._calculate_resolution(),
            'pressure_drop': self._measure_pressure_drop(),
            'binding_capacity': self._measure_binding_capacity(),
            'salt_gradient': self._evaluate_salt_gradient(),
            'peak_capacity': self._measure_peak_capacity()
        }

    def _run_hic_qualification(self):
        """Simulates HIC-specific qualification tests."""
        return {
            'theoretical_plates': self._calculate_theoretical_plates(),
            'asymmetry': self._calculate_peak_asymmetry(),
            'resolution': self._calculate_resolution(),
            'pressure_drop': self._measure_pressure_drop(),
            'hydrophobicity': self._measure_hydrophobicity(),
            'salt_response': self._evaluate_salt_response(),
            'peak_capacity': self._measure_peak_capacity()
        }

    def _run_generic_qualification(self):
        """Simulates generic column qualification tests."""
        return {
            'theoretical_plates': self._calculate_theoretical_plates(),
            'asymmetry': self._calculate_peak_asymmetry(),
            'resolution': self._calculate_resolution(),
            'pressure_drop': self._measure_pressure_drop(),
            'peak_capacity': self._measure_peak_capacity()
        }

    def _measure_peak_capacity(self):
        """Measures simulated peak capacity with enhanced precision."""
        return random.uniform(40, 60)

    def _calculate_theoretical_plates(self, retention_time=10, peak_width=0.5):
        """Calculates simulated number of theoretical plates."""
        return 16 * (retention_time / peak_width) ** 2

    def _calculate_peak_asymmetry(self, front_width=0.4, back_width=0.6):
        """Calculates simulated peak asymmetry factor."""
        return back_width / front_width

    def _calculate_resolution(self, rt1=10, rt2=12, w1=0.5, w2=0.5):
        """Calculates simulated resolution between two peaks."""
        return 2 * (rt2 - rt1) / (w1 + w2)

    def _measure_pressure_drop(self, flow_rate=1.0):
        """Measures simulated pressure drop across column."""
        return random.uniform(0.5, 2.0)

    def _determine_void_volume(self):
        """Determines simulated column void volume."""
        return random.uniform(0.3, 0.5)

    def _generate_calibration_curve(self):
        """Generates a simulated SEC calibration curve."""
        molecular_weights = [1000, 10000, 100000, 1000000]
        retention_times = [15, 12, 9, 6]
        return {'mw': molecular_weights, 'rt': retention_times}

    def _measure_binding_capacity(self):
        """Measures simulated dynamic binding capacity."""
        return random.uniform(80, 120)

    def _evaluate_salt_gradient(self):
        """Evaluates simulated salt gradient performance."""
        return {
            'linearity': random.uniform(0.95, 1.0),
            'reproducibility': random.uniform(0.98, 1.0)
        }

    def _measure_hydrophobicity(self):
        """Measures simulated column hydrophobicity."""
        return random.uniform(0.7, 0.9)

    def _evaluate_salt_response(self):
        """Evaluates simulated HIC salt response."""
        return {
            'retention_factor': random.uniform(1.5, 2.5),
            'selectivity': random.uniform(1.2, 1.8)
        }

    def _evaluate_results(self, results):
        """Evaluates if simulated results meet acceptance criteria."""
        if 'theoretical_plates' in results and results['theoretical_plates'] < self.acceptance_criteria['theoretical_plates']:
            return False
        if 'asymmetry' in results and results['asymmetry'] > self.acceptance_criteria['asymmetry']:
            return False
        if 'resolution' in results and results['resolution'] < self.acceptance_criteria['resolution']:
            return False
        if 'pressure_drop' in results and results['pressure_drop'] > self.acceptance_criteria['pressure_drop']:
            return False
        if 'peak_capacity' in results and results['peak_capacity'] < self.acceptance_criteria['peak_capacity']:
            return False
        return True

    def generate_qualification_report(self, test_data):
        """Generates a detailed qualification report."""
        report = {
            'date': test_data['date'].strftime("%Y-%m-%d %H:%M:%S"),
            'column_type': test_data['column_type'],
            'test_mixture': test_data['test_mixture'],
            'results': test_data['results'],
            'passed': test_data['passed'],
            'acceptance_criteria': self.acceptance_criteria
        }
        return report

    def export_qualification_history(self, filename):
        """Exports qualification history to a CSV file."""
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Date', 'Column Type', 'Test Mixture',
                    'Theoretical Plates', 'Asymmetry', 'Resolution',
                    'Pressure Drop', 'Peak Capacity', 'Passed'
                ])

                for date, data in self.qualification_data.items():
                    results = data['results']
                    writer.writerow([
                        date,
                        data['column_type'],
                        data['test_mixture'],
                        results.get('theoretical_plates', 'N/A'),
                        results.get('asymmetry', 'N/A'),
                        results.get('resolution', 'N/A'),
                        results.get('pressure_drop', 'N/A'),
                        results.get('peak_capacity', 'N/A'),
                        data['passed']
                    ])
            logger.info(f"Qualification history exported to {filename}.")
            return True
        except Exception as e:
            logger.error(f"Error exporting qualification history: {str(e)}")
            return False

class SimulatedFPLCHardware:
    """Simulates FPLC hardware behavior for testing and development."""
    def __init__(self):
        self.flow_rate = 0.0
        self.pressure = 0.0
        self.uv = 0.0
        self.conductivity = 0.0
        self.buffer_a = 100.0
        self.buffer_b = 0.0
        self.temperature = 25.0
        self.ph = 7.0
        self.valve_positions = {}
        self.is_running = False
        self._simulation_thread = None
        self.data_callback = None
        self.hardware_comm = HardwareCommunication()
        self.connected = False # Initial state

    def initialize_hardware(self):
        """Initializes simulated hardware components."""
        try:
            # Attempt to connect via simulated communication
            self.connected = self.hardware_comm.connect()
            if self.connected:
                self.flow_rate = 0.0
                self.pressure = 0.0
                self.uv = 0.0
                self.conductivity = 0.0
                self.buffer_a = 100.0
                self.buffer_b = 0.0
                self.temperature = 25.0
                self.ph = 7.0
                self.valve_positions = {}
                self.is_running = False
                logger.info("Simulated hardware initialized.")
                return True
            else:
                logger.warning("Failed to connect simulated hardware during initialization.")
                return False
        except Exception as e:
            logger.error(f"Hardware initialization error: {str(e)}")
            self.connected = False
            return False

    def start_run(self):
        """Starts the simulated FPLC run."""
        if not self.connected:
            logger.error("Cannot start run: Hardware not connected.")
            raise ConnectionError("Hardware not connected.")
        self.is_running = True
        self._simulation_thread = threading.Thread(target=self._simulate_run)
        self._simulation_thread.daemon = True
        self._simulation_thread.start()
        logger.info("Simulated run started.")

    def stop_run(self):
        """Stops the simulated FPLC run."""
        self.is_running = False
        if self._simulation_thread and self._simulation_thread.is_alive():
            self._simulation_thread.join(timeout=1) # Wait for thread to finish
            if self._simulation_thread.is_alive():
                logger.warning("Simulation thread did not terminate gracefully.")
        logger.info("Simulated run stopped.")

    def pause_run(self):
        """Pauses the simulated FPLC run."""
        # For a simple simulation, pausing means stopping data generation temporarily
        self.is_running = False
        logger.info("Simulated run paused.")

    def resume_run(self):
        """Resumes the simulated FPLC run."""
        if not self.connected:
            logger.error("Cannot resume run: Hardware not connected.")
            raise ConnectionError("Hardware not connected.")
        if not self.is_running:
            self.is_running = True
            self._simulation_thread = threading.Thread(target=self._simulate_run)
            self._simulation_thread.daemon = True
            self._simulation_thread.start()
            logger.info("Simulated run resumed.")

    def emergency_stop(self):
        """Performs an emergency stop on the simulated hardware."""
        self.is_running = False
        if self._simulation_thread and self._simulation_thread.is_alive():
            self._simulation_thread.join(timeout=1)
        logger.critical("Simulated emergency stop activated.")
        self.flow_rate = 0.0
        self.pressure = 0.0
        # Reset other parameters if necessary

    def _simulate_run(self):
        """Simulates data generation for an FPLC run."""
        time_point = 0
        base_uv = 0
        while self.is_running:
            # Simulate chromatogram peaks
            if 10 <= time_point <= 15:  # First peak
                base_uv = 100 * np.exp(-(time_point - 12.5)**2 / 2)
            elif 20 <= time_point <= 25:  # Second peak
                base_uv = 150 * np.exp(-(time_point - 22.5)**2 / 1.5)
            else:
                base_uv = 0

            # Add noise and simulate other parameters
            self.uv = base_uv + random.gauss(0, 2)
            self.pressure = 0.5 + random.gauss(0, 0.02)
            self.conductivity = 20 + random.gauss(0, 0.1) # Conductivity simulation
            self.flow_rate = random.uniform(0.5, 1.5) # Simulate varying flow
            self.buffer_a = random.uniform(0, 100)
            self.buffer_b = 100 - self.buffer_a
            self.temperature = 25.0 + random.gauss(0, 0.1)
            self.ph = 7.0 + random.gauss(0, 0.05)


            if self.data_callback:
                self.data_callback(time_point, self.uv, self.pressure,
                                 self.conductivity, self.flow_rate,
                                 self.buffer_a, self.buffer_b,
                                 self.temperature, self.ph)

            time_point += 0.1
            time.sleep(0.1)

    def get_current_data(self):
        """Gets current simulated system data."""
        return {
            'time': time.time(), # This will be the absolute time, not run time
            'uv': self.uv,
            'pressure': self.pressure,
            'conductivity': self.conductivity,
            'flow_rate': self.flow_rate,
            'buffer_a': self.buffer_a,
            'buffer_b': self.buffer_b,
            'temperature': self.temperature,
            'ph': self.ph,
            'volume': self.flow_rate * 0.1  # Approximate volume based on flow rate
        }

class PeakAnalysis:
    """Handles peak detection and analysis."""
    def __init__(self):
        self.peaks = []
        self.baseline = None
        self.standards = []

    def set_standards(self, standards):
        """Sets the standards for peak comparison."""
        self.standards = standards
        logger.info(f"Peak analysis standards set: {len(standards)} standards loaded.")

    def detect_peaks(self, time_data, uv_data):
        """Detects peaks in UV data and calculates their properties."""
        if not time_data or not uv_data:
            self.peaks = []
            logger.warning("No data provided for peak detection.")
            return

        # Find peaks using scipy
        # Adjust height and distance based on typical chromatogram characteristics
        peaks, properties = find_peaks(uv_data, height=10, distance=5) # Reduced distance for more peaks
        logger.info(f"Detected {len(peaks)} peaks.")

        # Calculate peak properties
        self.peaks = []
        for i, peak_idx in enumerate(peaks):
            # Ensure peak_idx is within bounds
            if peak_idx >= len(time_data):
                logger.warning(f"Peak index {peak_idx} out of bounds for time_data.")
                continue

            peak = {
                'index': peak_idx,
                'time': time_data[peak_idx],
                'height': uv_data[peak_idx],
                'area': self._calculate_peak_area(uv_data, peak_idx),
                'width': self._calculate_peak_width(uv_data, peak_idx),
                'standard_match': self.match_peak_to_standard(time_data[peak_idx])
            }
            self.peaks.append(peak)

    def match_peak_to_standard(self, peak_time):
        """Matches detected peak to known standards based on retention time."""
        for standard in self.standards:
            if abs(standard['retention_time'] - peak_time) < 0.5:  # Allow a tolerance
                return standard['name']
        return None

    def _calculate_peak_area(self, data, peak_idx):
        """Calculates peak area using simple trapezoidal integration."""
        # Define a window around the peak for integration
        window_size = 20 # points on each side
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(data), peak_idx + window_size)
        return np.trapz(data[start_idx:end_idx])

    def _calculate_peak_width(self, data, peak_idx):
        """Calculates peak width at half maximum."""
        if peak_idx < 0 or peak_idx >= len(data):
            return 0.0

        peak_height = data[peak_idx]
        half_height = peak_height / 2.0

        # Find left and right points at half height
        left_idx = peak_idx
        while left_idx > 0 and data[left_idx - 1] > half_height:
            left_idx -= 1

        right_idx = peak_idx
        while right_idx < len(data) - 1 and data[right_idx + 1] > half_height:
            right_idx += 1

        return (right_idx - left_idx) * 0.1 # Assuming 0.1 min per data point

class EnhancedFractionCollector(QGroupBox):
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
        self.hardware_comm = None
        self.current_volume = 0.0 # Accumulated volume for volume-based collection
        self.last_uv = 0.0 # Last UV value for peak detection

        self.setup_ui()
        logger.info("Fraction Collector UI initialized.")

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
        logger.info(f"Fraction collection mode set to: {mode}")
        # Enable/disable relevant controls based on mode
        self.threshold_spin.setEnabled(mode == "Peak")
        self.fraction_size_spin.setEnabled(mode in ["Time", "Volume"])


    def set_hardware_comm(self, comm):
        """Sets the hardware communication interface."""
        self.hardware_comm = comm

    def start_collecting(self):
        """Starts fraction collection."""
        try:
            if self.hardware_comm:
                self.hardware_comm.send_command("FC_START")

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

            logger.info("Started fraction collection.")

        except Exception as e:
            logger.error(f"Error starting collection: {str(e)}")
            QMessageBox.critical(
                self,
                "Collection Error",
                f"Failed to start collection: {str(e)}"
            )

    def stop_collecting(self):
        """Stops fraction collection."""
        try:
            if self.hardware_comm:
                self.hardware_comm.send_command("FC_STOP")

            self.is_collecting = False
            self.start_collection.setEnabled(True)
            self.stop_collection.setEnabled(False)

            logger.info("Stopped fraction collection.")

        except Exception as e:
            logger.error(f"Error stopping collection: {str(e)}")

    def reset(self):
        """Resets the fraction collector."""
        try:
            if self.hardware_comm:
                self.hardware_comm.send_command("FC_RESET")

            self.current_fraction = 0
            self.fractions.clear()
            self.fraction_table.setRowCount(0)
            self.is_collecting = False
            self.current_volume = 0.0
            self.last_uv = 0.0

            self.start_collection.setEnabled(True)
            self.stop_collection.setEnabled(False)

            logger.info("Reset fraction collector.")

        except Exception as e:
            logger.error(f"Error resetting collector: {str(e)}")

    def update(self, time_point, uv_value, volume_increment):
        """Updates the fraction collector state based on incoming data."""
        if not self.is_collecting:
            return

        if self.current_fraction >= self.max_fractions:
            self.stop_collecting()
            logger.warning("Max fractions reached. Collection stopped.")
            return

        collect_now = False
        if self.collection_mode == "Peak":
            if uv_value > self.collection_threshold and self.last_uv <= self.collection_threshold:
                # Start of a peak
                collect_now = True
            elif uv_value <= self.collection_threshold and self.last_uv > self.collection_threshold:
                # End of a peak
                # This logic could be enhanced to collect until UV drops below threshold for a sustained period
                pass
        elif self.collection_mode == "Volume":
            self.current_volume += volume_increment
            if self.current_volume >= self.fraction_size:
                collect_now = True
                self.current_volume = 0.0 # Reset for next fraction
        # Time-based collection would be handled by a separate timer in the main system

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
            logger.info(f"Collecting fraction #{self.current_fraction} at {time_point:.2f} min (UV: {uv_value:.2f}).")

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


class SystemDiagnostics:
    """Handles system diagnostics and maintenance procedures."""
    def __init__(self, hardware):
        self.hardware = hardware
        self.test_results = {}
        self.maintenance_history = []
        logger.info("System Diagnostics initialized.")

    def run_diagnostics(self):
        """Runs a suite of simulated diagnostic tests."""
        tests = [
            self._test_pressure_sensor,
            self._test_uv_detector,
            self._test_conductivity_detector, # Added conductivity detector test
            self._test_pump,
            self._test_valves
        ]

        self.test_results = {}
        all_passed = True
        for test in tests:
            result = test()
            self.test_results[test.__name__] = result
            if not result:
                all_passed = False
            logger.info(f"Diagnostic test '{test.__name__}' result: {'Passed' if result else 'Failed'}")

        logger.info(f"All diagnostics completed. Overall result: {'Passed' if all_passed else 'Failed'}")
        return all_passed

    def _test_pressure_sensor(self):
        """Simulates a pressure sensor test."""
        return random.random() > 0.1 # 90% pass rate

    def _test_uv_detector(self):
        """Simulates a UV detector test."""
        return random.random() > 0.1

    def _test_conductivity_detector(self):
        """Simulates a conductivity detector test."""
        # Simulate a 90% pass rate for the conductivity detector
        return random.random() > 0.1

    def _test_pump(self):
        """Simulates a pump test."""
        return random.random() > 0.1

    def _test_valves(self):
        """Simulates a valve test."""
        return random.random() > 0.1

    def log_maintenance(self, activity, user):
        """Logs a maintenance activity."""
        entry = {
            'date': datetime.now().isoformat(),
            'activity': activity,
            'user': user
        }
        self.maintenance_history.append(entry)
        logger.info(f"Maintenance logged: '{activity}' by {user}.")

class BufferBlending:
    """Handles buffer preparation and blending calculations."""
    def __init__(self):
        self.buffer_recipes = {}
        self.current_blend = {'A': 100, 'B': 0}
        logger.info("Buffer Blending initialized.")

    def add_buffer_recipe(self, name, components):
        """Adds a new buffer recipe."""
        self.buffer_recipes[name] = components
        logger.info(f"Buffer recipe '{name}' added.")

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

class DataAnalysis:
    """Handles advanced data analysis and reporting."""
    def __init__(self):
        self.time_data = []
        self.uv_data = []
        self.pressure_data = []
        self.conductivity_data = []
        self.flow_data = []
        self.buffer_a_data = []
        self.buffer_b_data = []
        self.peak_analyzer = PeakAnalysis()
        self.standards = []
        logger.info("Data Analysis initialized.")

    def add_data_point(self, time_point, uv, pressure, conductivity,
                      flow_rate, buffer_a, buffer_b):
        """Adds a new data point to the analysis storage."""
        self.time_data.append(time_point)
        self.uv_data.append(uv)
        self.pressure_data.append(pressure)
        self.conductivity_data.append(conductivity)
        self.flow_data.append(flow_rate)
        self.buffer_a_data.append(buffer_a)
        self.buffer_b_data.append(buffer_b)

    def load_standards(self, filename):
        """Loads chromatography standards from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.standards = data.get('standards', [])
            self.peak_analyzer.set_standards(self.standards)
            logger.info(f"Standards loaded successfully from {filename}.")
        except FileNotFoundError:
            logger.error(f"Standards file not found: {filename}")
            QMessageBox.critical(None, "File Error", f"Standards file not found: {filename}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from standards file: {filename}")
            QMessageBox.critical(None, "File Error", f"Invalid JSON in standards file: {filename}")
        except Exception as e:
            logger.error(f"Error loading standards: {str(e)}")
            QMessageBox.critical(None, "Error", f"Error loading standards: {str(e)}")

    def export_data(self, filename):
        """Exports all collected raw data to a CSV file."""
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Time (min)', 'UV (mAU)', 'Pressure (MPa)',
                    'Conductivity (mS/cm)', 'Flow Rate (mL/min)',
                    'Buffer A (%)', 'Buffer B (%)'
                ])

                for i in range(len(self.time_data)):
                    writer.writerow([
                        f"{self.time_data[i]:.2f}",
                        f"{self.uv_data[i]:.2f}",
                        f"{self.pressure_data[i]:.3f}",
                        f"{self.conductivity_data[i]:.2f}",
                        f"{self.flow_data[i]:.2f}",
                        f"{self.buffer_a_data[i]:.1f}",
                        f"{self.buffer_b_data[i]:.1f}"
                    ])
            logger.info(f"Raw data exported to {filename}.")
            return True
        except Exception as e:
            logger.error(f"Error exporting raw data: {str(e)}")
            return False

    def export_peaks(self, filename):
        """Exports detected peaks information to a CSV file."""
        try:
            peaks = self.peak_analyzer.peaks
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Peak Number', 'Retention Time (min)', 'Height (mAU)',
                    'Area (mAU*min)', 'Width (min)', 'Standard Match'
                ])

                for i, peak in enumerate(peaks, 1):
                    writer.writerow([
                        i,
                        f"{peak['time']:.2f}",
                        f"{peak['height']:.1f}",
                        f"{peak['area']:.1f}",
                        f"{peak['width']:.2f}",
                        peak['standard_match'] if peak['standard_match'] else "Unknown"
                    ])
            logger.info(f"Peak data exported to {filename}.")
            return True
        except Exception as e:
            logger.error(f"Error exporting peak data: {str(e)}")
            return False

    def calculate_statistics(self):
        """Calculates basic run statistics."""
        stats = {
            'run_duration': self.time_data[-1] if self.time_data else 0,
            'max_uv': max(self.uv_data) if self.uv_data else 0,
            'max_pressure': max(self.pressure_data) if self.pressure_data else 0,
            'average_conductivity': np.mean(self.conductivity_data) if self.conductivity_data else 0
        }
        return stats

    def generate_report(self, method, run_info, filename):
        """Generates a comprehensive run report in HTML format."""
        try:
            stats = self.calculate_statistics()
            peaks = self.peak_analyzer.peaks

            with open(filename, 'w') as f:
                f.write(f"""
                <html>
                <head>
                    <title>Chromatography Run Report - {run_info.get('run_id', 'N/A')}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                        h1, h2 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        .section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
                        .pass {{ color: green; font-weight: bold; }}
                        .fail {{ color: red; font-weight: bold; }}
                    </style>
                </head>
                <body>
                    <h1>Chromatography Run Report</h1>

                    <div class="section">
                        <h2>Run Information</h2>
                        <table>
                            <tr><td>Run ID</td><td>{run_info.get('run_id', 'N/A')}</td></tr>
                            <tr><td>Run Date</td><td>{run_info.get('start_time', datetime.now()).strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
                            <tr><td>Method Name</td><td>{method.name}</td></tr>
                            <tr><td>Author</td><td>{method.author}</td></tr>
                        </table>
                    </div>

                    <div class="section">
                        <h2>Method Details</h2>
                        <table>
                            <tr><td>Description</td><td>{method.description if method.description else 'N/A'}</td></tr>
                            <tr><td>Column Type</td><td>{method.column_type}</td></tr>
                            <tr><td>Column Volume</td><td>{method.column_volume:.1f} mL</td></tr>
                            <tr><td>Max Pressure</td><td>{method.max_pressure:.2f} MPa</td></tr>
                            <tr><td>Buffer A</td><td>{method.buffer_info['A']['name']} ({method.buffer_info['A']['composition']})</td></tr>
                            <tr><td>Buffer B</td><td>{method.buffer_info['B']['name']} ({method.buffer_info['B']['composition']})</td></tr>
                        </table>
                    </div>

                    <div class="section">
                        <h2>Run Statistics</h2>
                        <table>
                            <tr><td>Duration</td><td>{stats['run_duration']:.1f} min</td></tr>
                            <tr><td>Max UV</td><td>{stats['max_uv']:.1f} mAU</td></tr>
                            <tr><td>Max Pressure</td><td>{stats['max_pressure']:.2f} MPa</td></tr>
                            <tr><td>Average Conductivity</td>
                                <td>{stats['average_conductivity']:.1f} mS/cm</td></tr>
                        </table>
                    </div>

                    <div class="section">
                        <h2>Peak Analysis</h2>
                        <table>
                            <tr>
                                <th>Peak #</th>
                                <th>Retention Time (min)</th>
                                <th>Height (mAU)</th>
                                <th>Area (mAU*min)</th>
                                <th>Width (min)</th>
                                <th>Standard Match</th>
                            </tr>
                """)

                if not peaks:
                    f.write("<tr><td colspan='6'>No peaks detected.</td></tr>")
                else:
                    for i, peak in enumerate(peaks, 1):
                        f.write(f"""
                                <tr>
                                    <td>{i}</td>
                                    <td>{peak['time']:.2f}</td>
                                    <td>{peak['height']:.1f}</td>
                                    <td>{peak['area']:.1f}</td>
                                    <td>{peak['width']:.2f}</td>
                                    <td>{peak['standard_match'] if peak['standard_match'] else "Unknown"}</td>
                                </tr>
                        """)

                f.write("""
                        </table>
                    </div>
                </body>
                </html>
                """)
            logger.info(f"Run report generated at {filename}.")
            return True
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return False

# --- UI Components ---
class ColumnQualificationUI(QGroupBox):
    """UI for column qualification testing."""
    def __init__(self, qualification_manager, parent=None):
        super().__init__("Column Qualification", parent)
        self.qualification_manager = qualification_manager
        self.setup_ui()
        logger.info("Column Qualification UI initialized.")

    def setup_ui(self):
        """Sets up the UI elements for column qualification."""
        layout = QHBoxLayout()

        # Column information
        info_group = QGroupBox("Column Information")
        info_layout = QFormLayout()

        self.column_type = QComboBox()
        self.column_type.addItems(["SEC", "IEX", "HIC", "Custom"])

        self.test_mixture = QComboBox()
        self.test_mixture.addItems(["standard", "custom"])

        info_layout.addRow("Column Type:", self.column_type)
        info_layout.addRow("Test Mixture:", self.test_mixture)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        h_layout = QVBoxLayout()

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Result", "Pass/Fail"])
        h_layout.addWidget(self.results_table)

        # Add plot area
        self.plot_area = FigureCanvasQTAgg(Figure(figsize=(8, 4)))

        # Create subplots with proper grid specification
        gs = GridSpec(1, 2, figure=self.plot_area.figure)
        self.ax1 = self.plot_area.figure.add_subplot(gs[0, 0])
        self.ax2 = self.plot_area.figure.add_subplot(gs[0, 1])

        # Configure Van Deemter Plot
        self.ax1.set_xlabel('Linear Velocity (cm/s)')
        self.ax1.set_ylabel('HETP (μm)')
        self.ax1.set_title('Van Deemter Plot')
        self.ax1.grid(True, linestyle='--', alpha=0.3)

        # Configure Peak Symmetry Plot
        self.ax2.set_xlabel('Flow Rate (mL/min)')
        self.ax2.set_ylabel('Peak Symmetry Factor')
        self.ax2.set_title('Peak Symmetry vs Flow Rate')
        self.ax2.grid(True, linestyle='--', alpha=0.3)
        self.ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)

        self.plot_area.figure.tight_layout()
        h_layout.addWidget(self.plot_area)
        layout.addLayout(h_layout)

        # Control buttons
        button_layout = QVBoxLayout()
        self.start_qual = QPushButton("Start Qualification")
        self.export_results = QPushButton("Export Results")
        self.view_history = QPushButton("View History")

        button_layout.addWidget(self.start_qual)
        button_layout.addWidget(self.export_results)
        button_layout.addWidget(self.view_history)
        layout.addLayout(button_layout)

        # Connect signals
        self.start_qual.clicked.connect(self.start_qualification_process)
        self.export_results.clicked.connect(self.export_qualification_results)
        self.view_history.clicked.connect(self.show_qualification_history)

        self.setLayout(layout)

    def start_qualification_process(self):
        """Initiates the column qualification test and updates UI."""
        column_type = self.column_type.currentText()
        test_mixture = self.test_mixture.currentText()
        try:
            # Get flow rate range for Van Deemter analysis
            flow_rates = np.linspace(0.1, 1.0, 10) # Example flow rates

            # Run the qualification test
            test_data = self.qualification_manager.run_qualification(column_type, test_mixture)

            # Calculate HETP and Symmetry for plotting
            hetp_values = self.qualification_manager.calculate_hetp(flow_rates)
            symmetry_values = self.qualification_manager.calculate_symmetry(flow_rates)

            # Prepare results for display and plot
            results_for_display = {
                'results': test_data['results'],
                'passed': test_data['passed'],
                'flow_rates': flow_rates,
                'hetp_values': hetp_values,
                'symmetry_values': symmetry_values
            }

            self.display_results(results_for_display)
            self.update_plot(results_for_display)

        except Exception as e:
            logger.error(f"Qualification process error: {str(e)}")
            QMessageBox.critical(self, "Qualification Error", str(e))

    def display_results(self, results):
        """Displays qualification results in the table."""
        self.results_table.setRowCount(0)
        for param, value in results['results'].items():
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            self.results_table.setItem(row, 0, QTableWidgetItem(param))
            if isinstance(value, (int, float)):
                self.results_table.setItem(row, 1, QTableWidgetItem(f"{value:.2f}"))
            else:
                self.results_table.setItem(row, 1, QTableWidgetItem(str(value)))

            # Determine pass/fail based on acceptance criteria
            acceptance_value = self.qualification_manager.acceptance_criteria.get(param)
            passed_status = "N/A"
            if isinstance(value, (int, float)) and acceptance_value is not None:
                if param == 'asymmetry' or param == 'pressure_drop': # These are max limits
                    passed_status = "Pass" if value <= acceptance_value else "Fail"
                else: # These are min limits (theoretical_plates, peak_capacity, resolution)
                    passed_status = "Pass" if value >= acceptance_value else "Fail"
            self.results_table.setItem(row, 2, QTableWidgetItem(passed_status))

        if results['passed']:
            QMessageBox.information(self, "Qualification Result", "Qualification Passed!")
        else:
            QMessageBox.warning(self, "Qualification Result", "Qualification Failed!")

    def update_plot(self, results):
        """Updates the plot with HETP and Asymmetry results."""
        self.ax1.clear()
        self.ax2.clear()

        # Plotting HETP values
        self.ax1.plot(results['flow_rates'], results['hetp_values'], marker='o', color='blue', label='HETP')
        self.ax1.set_title('Van Deemter Plot')
        self.ax1.set_xlabel('Flow Rate (mL/min)')
        self.ax1.set_ylabel('HETP (μm)')
        self.ax1.grid(True, linestyle='--', alpha=0.3)
        self.ax1.legend()

        # Plotting Peak Symmetry values
        self.ax2.plot(results['flow_rates'], results['symmetry_values'], marker='o', color='orange', label='Symmetry')
        self.ax2.set_title('Peak Symmetry vs Flow Rate')
        self.ax2.set_xlabel('Flow Rate (mL/min)')
        self.ax2.set_ylabel('Peak Symmetry Factor')
        self.ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Symmetry (1.0)')
        self.ax2.grid(True, linestyle='--', alpha=0.3)
        self.ax2.legend()

        self.plot_area.figure.tight_layout()
        self.plot_area.draw()

    def export_qualification_results(self):
        """Exports qualification results to a CSV file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Qualification Results", "", "CSV Files (*.csv)"
        )
        if filename:
            if self.qualification_manager.export_qualification_history(filename):
                QMessageBox.information(self, "Export Complete", "Qualification history exported successfully.")
            else:
                QMessageBox.warning(self, "Export Failed", "Failed to export qualification history.")

    def show_qualification_history(self):
        """Shows a dialog with the qualification history."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Qualification History")
        layout = QVBoxLayout()

        history_table = QTableWidget()
        history_table.setColumnCount(4)
        history_table.setHorizontalHeaderLabels([
            "Date", "Column Type", "Test Mixture", "Result"
        ])

        for date_str, data in self.qualification_manager.qualification_data.items():
            row = history_table.rowCount()
            history_table.insertRow(row)
            history_table.setItem(row, 0, QTableWidgetItem(date_str))
            history_table.setItem(row, 1, QTableWidgetItem(data['column_type']))
            history_table.setItem(row, 2, QTableWidgetItem(data['test_mixture']))
            history_table.setItem(row, 3, QTableWidgetItem(
                "Pass" if data['passed'] else "Fail"
            ))

        layout.addWidget(history_table)
        dialog.setLayout(layout)
        dialog.exec()

class MethodEditor(QGroupBox):
    """Enhanced method editor with advanced chromatography controls."""
    def __init__(self, parent=None):
        super().__init__("Method Editor", parent)
        self.method = FPLCMethod()
        self.setup_ui()
        logger.info("Method Editor UI initialized.")

    def setup_ui(self):
        """Sets up the UI elements for the method editor."""
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # Method metadata
        metadata_group = QGroupBox("Method Information")
        metadata_layout = QFormLayout()
        self.method_name = QLineEdit()
        self.method_desc = QLineEdit()
        self.author = QLineEdit()
        metadata_layout.addRow("Method Name:", self.method_name)
        metadata_layout.addRow("Description:", self.method_desc)
        metadata_layout.addRow("Author:", self.author)
        metadata_group.setLayout(metadata_layout)
        top_layout.addWidget(metadata_group)

        # Buffer configuration
        buffer_group = QGroupBox("Buffer Configuration")
        buffer_layout = QFormLayout()

        self.buffer_a_name = QLineEdit()
        self.buffer_a_composition = QLineEdit()
        buffer_layout.addRow("Buffer A Name:", self.buffer_a_name)
        buffer_layout.addRow("Buffer A Composition:", self.buffer_a_composition)

        self.buffer_b_name = QLineEdit()
        self.buffer_b_composition = QLineEdit()
        buffer_layout.addRow("Buffer B Name:", self.buffer_b_name)
        buffer_layout.addRow("Buffer B Composition:", self.buffer_b_composition)

        buffer_group.setLayout(buffer_layout)
        top_layout.addWidget(buffer_group)

        # Column configuration
        column_group = QGroupBox("Column Configuration")
        column_layout = QFormLayout()
        self.column_type = QComboBox()
        self.column_type.addItems(["SEC", "IEX", "HIC", "AC", "Custom"])
        self.column_volume = QDoubleSpinBox()
        self.column_volume.setRange(0.1, 1000.0)
        self.max_pressure = QDoubleSpinBox()
        self.max_pressure.setRange(0.1, 20.0)
        column_layout.addRow("Column Type:", self.column_type)
        column_layout.addRow("Column Volume (mL):", self.column_volume)
        column_layout.addRow("Max Pressure (MPa):", self.max_pressure)
        column_group.setLayout(column_layout)
        top_layout.addWidget(column_group)

        layout.addLayout(top_layout)

        # Steps table with enhanced functionality
        self.steps_table = QTableWidget()
        self.steps_table.setColumnCount(9)
        self.steps_table.setHorizontalHeaderLabels([
            "Step", "Duration (min)", "Flow Rate (mL/min)",
            "Buffer A (%)", "Buffer B (%)", "Column Volume",
            "Collection", "Notes", "Actions"
        ])
        self.steps_table.setColumnWidth(0, 120) # Step type
        self.steps_table.setColumnWidth(1, 100) # Duration
        self.steps_table.setColumnWidth(2, 120) # Flow Rate
        self.steps_table.setColumnWidth(3, 100) # Buffer A
        self.steps_table.setColumnWidth(4, 100) # Buffer B
        self.steps_table.setColumnWidth(5, 100) # Column Volume
        self.steps_table.setColumnWidth(6, 80)  # Collection
        self.steps_table.setColumnWidth(7, 150) # Notes
        self.steps_table.setColumnWidth(8, 80)  # Actions
        layout.addWidget(self.steps_table)

        # Step control buttons with integrated validation
        button_layout = QHBoxLayout()
        steps = ["Equilibration", "Sample Load", "Wash", "Elution", "Strip", "CIP"]
        for step in steps:
            btn = QPushButton(f"Add {step}")
            btn.clicked.connect(lambda checked, s=step: self.add_step(s)) # Use lambda with checked
            button_layout.addWidget(btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals for metadata and column info
        self.method_name.textChanged.connect(self.update_method_metadata)
        self.method_desc.textChanged.connect(self.update_method_metadata)
        self.author.textChanged.connect(self.update_method_metadata)
        self.column_type.currentTextChanged.connect(self.update_method_metadata)
        self.column_volume.valueChanged.connect(self.update_method_metadata)
        self.max_pressure.valueChanged.connect(self.update_method_metadata)
        self.buffer_a_name.textChanged.connect(self.update_method_metadata)
        self.buffer_a_composition.textChanged.connect(self.update_method_metadata)
        # FIX: Changed from calling to connecting the signal
        self.buffer_b_composition.textChanged.connect(self.update_method_metadata)

    def add_step(self, step_type):
        """Adds a new step to the method table and object."""
        row = self.steps_table.rowCount()
        self.steps_table.insertRow(row)

        self.steps_table.setItem(row, 0, QTableWidgetItem(step_type))

        duration_spin = QDoubleSpinBox()
        duration_spin.setRange(0.1, 1000.0)
        duration_spin.setValue(5.0) # Default
        duration_spin.valueChanged.connect(lambda value, r=row: self._update_step_data(r, 'duration', value))
        self.steps_table.setCellWidget(row, 1, duration_spin)

        flow_spin = QDoubleSpinBox()
        flow_spin.setRange(0.1, 100.0)
        flow_spin.setValue(1.0) # Default
        flow_spin.valueChanged.connect(lambda value, r=row: self._update_step_data(r, 'flow_rate', value))
        self.steps_table.setCellWidget(row, 2, flow_spin)

        buffer_a_spin = QSpinBox()
        buffer_a_spin.setRange(0, 100)
        buffer_a_spin.setValue(100) # Default
        buffer_a_spin.valueChanged.connect(lambda value, r=row: self._update_step_data(r, 'buffer_a', value))
        self.steps_table.setCellWidget(row, 3, buffer_a_spin)

        buffer_b_spin = QSpinBox()
        buffer_b_spin.setRange(0, 100)
        buffer_b_spin.setValue(0) # Default
        buffer_b_spin.valueChanged.connect(lambda value, r=row: self._update_step_data(r, 'buffer_b', value))
        self.steps_table.setCellWidget(row, 4, buffer_b_spin)

        # Column Volume (calculated based on flow rate and duration, or set manually)
        column_volume_label = QLabel("0.0") # Placeholder, will be updated by logic
        self.steps_table.setCellWidget(row, 5, column_volume_label)

        collection_checkbox = QCheckBox()
        collection_checkbox.setChecked(step_type in ["Elution", "Sample Load"])
        collection_checkbox.stateChanged.connect(lambda state, r=row: self._update_step_data(r, 'collection', state == Qt.CheckState.Checked.value))
        self.steps_table.setCellWidget(row, 6, collection_checkbox)

        notes_edit = QLineEdit()
        notes_edit.textChanged.connect(lambda text, r=row: self._update_step_data(r, 'notes', text))
        self.steps_table.setCellWidget(row, 7, notes_edit)

        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda checked, r=row: self.delete_step(r))
        self.steps_table.setCellWidget(row, 8, delete_button)

        # Add to method object with initial values
        self.method.add_step(
            step_type=step_type,
            duration=duration_spin.value(),
            flow_rate=flow_spin.value(),
            buffer_a=buffer_a_spin.value(),
            buffer_b=buffer_b_spin.value(),
            collection=collection_checkbox.isChecked(),
            notes=notes_edit.text()
        )
        logger.info(f"Added new step '{step_type}' to method editor.")

    def _update_step_data(self, row, param_name, value):
        """Updates a specific parameter for a step in the method object."""
        if 0 <= row < len(self.method.steps):
            self.method.steps[row][param_name] = value
            self.method.modified_date = datetime.now()
            logger.debug(f"Updated step {row}, param '{param_name}' to '{value}'.")
            # Special handling for buffer A/B to ensure they sum to 100
            if param_name == 'buffer_a':
                buffer_b_spin = self.steps_table.cellWidget(row, 4)
                if buffer_b_spin:
                    buffer_b_spin.blockSignals(True) # Block signals to prevent infinite loop
                    buffer_b_spin.setValue(100 - value)
                    buffer_b_spin.blockSignals(False)
                    self.method.steps[row]['buffer_b'] = 100 - value
            elif param_name == 'buffer_b':
                buffer_a_spin = self.steps_table.cellWidget(row, 3)
                if buffer_a_spin:
                    buffer_a_spin.blockSignals(True)
                    buffer_a_spin.setValue(100 - value)
                    buffer_a_spin.blockSignals(False)
                    self.method.steps[row]['buffer_a'] = 100 - value
        else:
            logger.warning(f"Attempted to update non-existent step at row {row}.")

    def delete_step(self, row):
        """Deletes a step from both table and method object."""
        if 0 <= row < self.steps_table.rowCount():
            step_type = self.steps_table.item(row, 0).text() if self.steps_table.item(row, 0) else "Unknown"
            self.steps_table.removeRow(row)
            if 0 <= row < len(self.method.steps):
                self.method.steps.pop(row)
                self.method.modified_date = datetime.now()
                logger.info(f"Deleted step '{step_type}' at row {row}.")
            else:
                logger.warning(f"Attempted to delete step from method object at non-existent index {row}.")

            # Reconnect signals for remaining delete buttons to ensure correct row indexing
            for i in range(row, self.steps_table.rowCount()):
                delete_button = self.steps_table.cellWidget(i, 8)
                if delete_button:
                    try:
                        delete_button.clicked.disconnect()
                    except TypeError: # Handle case where signal might not be connected yet
                        pass
                    delete_button.clicked.connect(lambda checked, r=i: self.delete_step(r))
        else:
            logger.warning(f"Attempted to delete non-existent row {row} from steps table.")


    def update_method_metadata(self):
        """Updates method object with current UI values for metadata and buffers."""
        self.method.name = self.method_name.text()
        self.method.description = self.method_desc.text()
        self.method.author = self.author.text()
        self.method.column_type = self.column_type.currentText()
        self.method.column_volume = self.column_volume.value()
        self.method.max_pressure = self.max_pressure.value()

        self.method.buffer_info['A'] = {
            'name': self.buffer_a_name.text(),
            'composition': self.buffer_a_composition.text()
        }
        self.method.buffer_info['B'] = {
            'name': self.buffer_b_name.text(),
            'composition': self.buffer_b_composition.text()
        }
        self.method.modified_date = datetime.now()
        logger.debug("Method metadata updated from UI.")

    def load_method_into_ui(self, method):
        """Loads an FPLCMethod object's data into the UI fields."""
        self.method = method
        self.method_name.setText(method.name)
        self.method_desc.setText(method.description)
        self.author.setText(method.author)
        self.column_type.setCurrentText(method.column_type)
        self.column_volume.setValue(method.column_volume)
        self.max_pressure.setValue(method.max_pressure)

        self.buffer_a_name.setText(method.buffer_info['A']['name'])
        self.buffer_a_composition.setText(method.buffer_info['A']['composition'])
        self.buffer_b_name.setText(method.buffer_info['B']['name'])
        self.buffer_b_composition.setText(method.buffer_info['B']['composition'])

        self.steps_table.setRowCount(0)
        for step_data in method.steps:
            row = self.steps_table.rowCount()
            self.steps_table.insertRow(row)

            self.steps_table.setItem(row, 0, QTableWidgetItem(step_data.get('type', '')))

            duration_spin = QDoubleSpinBox()
            duration_spin.setRange(0.1, 1000.0)
            duration_spin.setValue(step_data.get('duration', 0.0))
            duration_spin.valueChanged.connect(lambda value, r=row: self._update_step_data(r, 'duration', value))
            self.steps_table.setCellWidget(row, 1, duration_spin)

            flow_spin = QDoubleSpinBox()
            flow_spin.setRange(0.1, 100.0)
            flow_spin.setValue(step_data.get('flow_rate', 0.0))
            flow_spin.valueChanged.connect(lambda value, r=row: self._update_step_data(r, 'flow_rate', value))
            self.steps_table.setCellWidget(row, 2, flow_spin)

            buffer_a_spin = QSpinBox()
            buffer_a_spin.setRange(0, 100)
            buffer_a_spin.setValue(step_data.get('buffer_a', 0))
            buffer_a_spin.valueChanged.connect(lambda value, r=row: self._update_step_data(r, 'buffer_a', value))
            self.steps_table.setCellWidget(row, 3, buffer_a_spin)

            buffer_b_spin = QSpinBox()
            buffer_b_spin.setRange(0, 100)
            buffer_b_spin.setValue(step_data.get('buffer_b', 0))
            buffer_b_spin.valueChanged.connect(lambda value, r=row: self._update_step_data(r, 'buffer_b', value))
            self.steps_table.setCellWidget(row, 4, buffer_b_spin)

            column_volume_label = QLabel("0.0") # Placeholder, will be updated by logic
            self.steps_table.setCellWidget(row, 5, column_volume_label)

            collection_checkbox = QCheckBox()
            collection_checkbox.setChecked(step_data.get('collection', False))
            collection_checkbox.stateChanged.connect(lambda state, r=row: self._update_step_data(r, 'collection', state == Qt.CheckState.Checked.value))
            self.steps_table.setCellWidget(row, 6, collection_checkbox)

            notes_edit = QLineEdit()
            notes_edit.setText(step_data.get('notes', ''))
            notes_edit.textChanged.connect(lambda text, r=row: self._update_step_data(r, 'notes', text))
            self.steps_table.setCellWidget(row, 7, notes_edit)

            delete_button = QPushButton("Delete")
            delete_button.clicked.connect(lambda checked, r=row: self.delete_step(r))
            self.steps_table.setCellWidget(row, 8, delete_button)
        logger.info(f"Method '{method.name}' loaded into UI.")


class EnhancedChromatogramPlot(QWidget):
    """Advanced chromatogram visualization with peak detection and real-time data display."""
    def __init__(self, data_analysis):
        super().__init__()
        self.data_analysis = data_analysis
        self.setup_ui()
        logger.info("Chromatogram Plot UI initialized.")

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

        data_display_layout.addRow("UV:", self.current_uv_label)
        data_display_layout.addRow("Pressure:", self.current_pressure_label)
        data_display_layout.addRow("Conductivity:", self.current_conductivity_label)
        data_display_layout.addRow("Flow Rate:", self.current_flow_label)
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

    def update_realtime_data_labels(self, uv, pressure, conductivity, flow_rate):
        """Updates the real-time data display labels."""
        self.current_uv_label.setText(f"UV: {uv:.3f} mAU")
        self.current_pressure_label.setText(f"Pressure: {pressure:.3f} MPa")
        self.current_conductivity_label.setText(f"Conductivity: {conductivity:.2f} mS/cm")
        self.current_flow_label.setText(f"Flow: {flow_rate:.2f} mL/min")


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
        logger.info(f"Displayed {len(peaks)} peaks.")

    def auto_scale_plot(self):
        """Auto-scales the plot axes."""
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()
        logger.info("Chromatogram plot auto-scaled.")

    def toggle_peak_display(self):
        """Toggles the display of peaks on the chromatogram."""
        self.showing_peaks = not self.showing_peaks
        if self.showing_peaks:
            self.display_peaks()
            self.show_peaks.setText("Hide Peaks")
            logger.info("Peak display enabled.")
        else:
            # Clear peak markers
            for marker in self.peak_markers:
                if marker in self.ax1.lines:
                    marker.remove()
            self.peak_markers.clear()
            self.canvas.draw()
            self.show_peaks.setText("Show Peaks")
            logger.info("Peak display disabled.")

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
                logger.info("Export cancelled by user (no directory selected).")
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
                    logger.error(f"Failed to export raw data to {filename}.")

            if peak_data_cb.isChecked():
                filename = os.path.join(base_path, f"peak_data_{timestamp}.csv")
                if not self.data_analysis.export_peaks(filename):
                    success = False
                    logger.error(f"Failed to export peak data to {filename}.")

            if report_cb.isChecked():
                filename = os.path.join(base_path, f"run_report_{timestamp}.html")
                if not self.data_analysis.generate_report(
                    current_method,
                    current_run_data,
                    filename
                ):
                    success = False
                    logger.error(f"Failed to generate HTML report to {filename}.")

            if success:
                QMessageBox.information(
                    dialog,
                    "Export Complete",
                    f"Data exported successfully to:\n{base_path}"
                )
                logger.info(f"Data exported successfully to {base_path}.")
            else:
                QMessageBox.warning(
                    dialog,
                    "Export Warning",
                    "Some export operations failed. Check the log for details."
                )
                logger.warning("Some export operations failed.")

            dialog.accept()

        export_btn.clicked.connect(do_export)
        dialog.exec()

# --- User Management ---
class UserManager:
    """Manages user authentication and permissions."""
    def __init__(self):
        self.users = {}
        self.current_user = None
        self.load_default_users()
        logger.info("User Manager initialized.")

    def load_default_users(self):
        """Loads default user accounts."""
        self.add_user(username="admin", password="admin123", role="administrator", full_name="System Administrator")
        self.add_user(username="operator", password="op123", role="operator", full_name="System Operator")
        logger.info("Default users loaded.")

    def add_user(self, username, password, role, full_name):
        """Adds a new user account."""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        self.users[username] = {
            'password_hash': password_hash,
            'role': role,
            'full_name': full_name,
            'created_date': datetime.now().isoformat()
        }
        logger.info(f"User '{username}' with role '{role}' added.")

    def verify_user(self, username, password):
        """Verifies user credentials."""
        if username not in self.users:
            logger.warning(f"Login attempt for non-existent user: {username}")
            return False
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        is_valid = self.users[username]['password_hash'] == password_hash
        if not is_valid:
            logger.warning(f"Failed login attempt for user: {username}")
        return is_valid

    def login(self, username, password):
        """Logs in a user."""
        if self.verify_user(username, password):
            self.current_user = username
            logger.info(f"User logged in: {username} (Role: {self.get_user_role(username)})")
            return True
        return False

    def logout(self):
        """Logs out the current user."""
        if self.current_user:
            logger.info(f"User logged out: {self.current_user}")
            self.current_user = None

    def get_user_role(self, username):
        """Gets a user's role."""
        return self.users.get(username, {}).get('role')

    def check_permission(self, required_role):
        """Checks if the current user has the required role or higher."""
        if not self.current_user:
            return False

        user_role = self.get_user_role(self.current_user)
        if user_role == 'administrator':
            return True
        return user_role == required_role

class LoginDialog(QDialog):
    # This __init__ method does NOT take 'user_manager' as an argument.
    # It will handle credential validation internally using hardcoded values.
    def __init__(self, parent=None):
        super().__init__(parent)
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

        # --- Debug print statements (you can remove these once it works) ---
        print(f"DEBUG: Stored Username: '{self.valid_username}'")
        print(f"DEBUG: Entered Username: '{entered_username}'")
        print(f"DEBUG: Username Match: {entered_username == self.valid_username}")

        print(f"DEBUG: Stored Password Hash: '{self.valid_password_hash}'")
        print(f"DEBUG: Entered Password Hash: '{entered_password_hash}'")
        print(f"DEBUG: Password Hash Match: {entered_password_hash == self.valid_password_hash}")

        overall_match = (entered_username == self.valid_username and
                         entered_password_hash == self.valid_password_hash)
        print(f"DEBUG: Overall Match: {overall_match}")
        # --- End of debug print statements ---

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
        logger.info("Alarm Manager initialized.")

    def add_alarm(self, severity, message):
        """Adds a new alarm and emits a signal."""
        logger.warning(f"ALARM: {severity} - {message}")
        self.alarm_triggered.emit(severity, message)

# --- Main Application Window ---
class IndustrialFPLCSystem(QMainWindow):
    """Main FPLC control system with industrial features."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial FPLC Control System")
        self.setGeometry(100, 100, 1200, 800) # Set initial window size

        # Initialize core components
        self.user_manager = UserManager()
        self.current_user = None # Will be set after login
        self.system_state = SystemState.IDLE
        self.hardware = SimulatedFPLCHardware()
        self.data_analysis = DataAnalysis()
        self.fraction_collector = EnhancedFractionCollector()
        self.fraction_collector.set_hardware_comm(self.hardware.hardware_comm)
        self.diagnostics = SystemDiagnostics(self.hardware)
        self.buffer_blending = BufferBlending()
        self.column_qualification = ColumnQualification()
        self.alarm_manager = AlarmManager()

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

        # Initial system state and login prompt
        self.update_system_state(SystemState.IDLE)
        #self.prompt_login()
        self.initialize_system() # Initialize hardware after potential login

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
        self.uv_display = QLabel("UV: 0.000 mAU")
        self.pressure_display = QLabel("Pressure: 0.000 MPa")
        self.flow_display = QLabel("Flow: 0.0 mL/min")
        self.conductivity_display = QLabel("Conductivity: 0.0 mS/cm")

        status_layout.addWidget(self.system_status)
        status_layout.addWidget(self.logged_in_user_label)
        status_layout.addWidget(self.uv_display)
        status_layout.addWidget(self.pressure_display)
        status_layout.addWidget(self.flow_display)
        status_layout.addWidget(self.conductivity_display)
        status_group.setLayout(status_layout)
        control_panel.addWidget(status_group)

        main_layout.addLayout(control_panel)

        # Create tab widget for different views
        self.tab_widget = QTabWidget()

        # Add tabs in the desired order
        self.method_editor = MethodEditor()
        self.tab_widget.addTab(self.method_editor, "Method Editor")

        self.chromatogram = EnhancedChromatogramPlot(self.data_analysis)
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
        #self.login_action.triggered.connect(self.prompt_login)
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

    def update_menu_visibility(self):
        """Updates menu item visibility based on login status."""
        is_logged_in = (self.current_user is not None)
        is_admin = self.user_manager.check_permission('administrator')

        self.login_action.setEnabled(not is_logged_in)
        self.logout_action.setEnabled(is_logged_in)

        # Directly use the instance attributes for menu actions
        if hasattr(self, 'new_method_action'): # Check if attribute exists before using
            self.new_method_action.setEnabled(is_logged_in)
        if hasattr(self, 'open_method_action'):
            self.open_method_action.setEnabled(is_logged_in)
        if hasattr(self, 'save_method_action'):
            self.save_method_action.setEnabled(is_logged_in)
        if hasattr(self, 'run_diagnostics_action'):
            self.run_diagnostics_action.setEnabled(is_admin) # Only admin can run diagnostics
        if hasattr(self, 'export_logs_action'):
            self.export_logs_action.setEnabled(is_admin) # Only admin can export logs

        # For simplicity, let's keep most enabled for now, but this is the pattern.
        self.start_button.setEnabled(is_logged_in)
        self.pause_button.setEnabled(is_logged_in)
        self.stop_button.setEnabled(is_logged_in)
        self.emergency_stop.setEnabled(is_logged_in)
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
        self.logged_in_user_label.setText(f"Logged In: {self.current_user if self.current_user else 'None'}")

    """
    def prompt_login(self):
        Displays the login dialog and handles login outcome.
        login_dialog = LoginDialog(self)
        if login_dialog.exec() == QDialog.DialogCode.Accepted:
            self.current_user = self.user_manager.current_user
            self.status_bar.showMessage(f"Logged in as {self.current_user}.")
            self.update_menu_visibility()
            logger.info(f"User '{self.current_user}' successfully logged in.")
        else:
            self.current_user = None
            self.status_bar.showMessage("Not logged in.")
            self.update_menu_visibility()
            logger.info("Login cancelled or failed.")
            # If not logged in, disable critical functions
            self.update_system_state(SystemState.IDLE) # Ensure system is idle if not logged in
    """
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
        self.chromatogram.update_plot([], [], []) # Clear plot
        self.chromatogram.update_realtime_data_labels(0, 0, 0, 0) # Reset labels

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
            self.method_editor.load_method_into_ui(FPLCMethod())
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

    def show_diagnostics(self):
        """Placeholder for showing diagnostics UI."""
        QMessageBox.information(self, "Diagnostics", "Displaying system diagnostics (feature not fully implemented).")
        logger.info("Diagnostics UI requested.")

    def run_diagnostics(self):
        """Runs system diagnostics."""
        if not self.user_manager.check_permission('administrator'):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to run diagnostics.")
            logger.warning(f"Permission denied for {self.current_user} to run diagnostics.")
            return

        self.status_bar.showMessage("Running system diagnostics...")
        logger.info("Running system diagnostics.")
        result = self.diagnostics.run_diagnostics()
        if result:
            QMessageBox.information(self, "Diagnostics", "All diagnostics passed.")
            self.status_bar.showMessage("Diagnostics passed.")
            logger.info("All diagnostics passed.")
        else:
            QMessageBox.warning(self, "Diagnostics", "Some diagnostics failed. Check logs for details.")
            self.status_bar.showMessage("Diagnostics failed.")
            logger.warning("Some diagnostics failed.")

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
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_data)
        self.data_timer.setInterval(100) # 100ms interval for data collection

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_system_status)
        self.status_timer.setInterval(1000) # 1s interval for system checks
        self.status_timer.start()
        logger.info("System timers set up.")

    def setup_signals(self):
        """Connects system signals."""
        self.hardware.data_callback = self.on_data_received

        self.start_button.clicked.connect(self.start_run)
        self.pause_button.clicked.connect(self.pause_run)
        self.stop_button.clicked.connect(self.stop_run)
        self.emergency_stop.clicked.connect(self.emergency_stop_run)
        self.load_standards_button.clicked.connect(self.load_standards)

        self.alarm_manager.alarm_triggered.connect(self.handle_alarm)
        logger.info("System signals connected.")

    def check_system_status(self):
        """Periodic system status check."""
        try:
            if not self.hardware.connected:
                if self.system_state != SystemState.ERROR: # Avoid re-logging if already in error
                    self.update_system_state(SystemState.ERROR)
                    self.status_bar.showMessage("Hardware connection lost!")
                    self.alarm_manager.add_alarm("Critical", "Hardware connection lost. Please check connections.")
                return

            # Check system pressure against method's max pressure
            if self.system_state == SystemState.RUNNING:
                max_pressure_limit = self.method_editor.method.max_pressure
                if max_pressure_limit > 0 and self.hardware.pressure > max_pressure_limit:
                    self.alarm_manager.add_alarm(
                        "Critical",
                        f"Pressure exceeded maximum limit ({max_pressure_limit:.2f} MPa): {self.hardware.pressure:.2f} MPa"
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
            if not self.hardware.initialize_hardware():
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
                    "System diagnostics detected issues. Check maintenance tab for details."
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

    def on_data_received(self, time_point, uv, pressure, conductivity, flow_rate, buffer_a, buffer_b, temperature, ph):
        """Handles received data from hardware, updates UI and analysis."""
        try:
            # Add data to analysis
            self.data_analysis.add_data_point(
                time_point, uv, pressure, conductivity,
                flow_rate, buffer_a, buffer_b
            )

            # Update real-time data displays on chromatogram tab
            self.chromatogram.update_realtime_data_labels(uv, pressure, conductivity, flow_rate)

            # Update chromatogram plot
            self.chromatogram.update_plot(
                self.data_analysis.time_data,
                self.data_analysis.uv_data,
                self.data_analysis.conductivity_data
            )

            # Check for fraction collection
            if self.fraction_collector.is_collecting:
                self.fraction_collector.update(
                    time_point,
                    uv,
                    flow_rate * 0.1 # Approximate volume based on 0.1s update interval
                )

            # Check system parameters (e.g., pressure limits)
            # This is already handled in check_system_status, but can add more granular checks here
            # self.check_parameters({'pressure': pressure, ...})

        except Exception as e:
            logger.error(f"Error processing data in on_data_received: {str(e)}")
            self.data_timer.stop()
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
            # Generate unique run ID
            self.current_run_data['run_id'] = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_run_data['start_time'] = datetime.now()
            self.current_run_data['method_name'] = self.method_editor.method.name

            # Reset data analysis for a new run
            self.data_analysis = DataAnalysis()
            self.data_analysis.peak_analyzer.set_standards(self.data_analysis.standards) # Re-apply standards if loaded
            self.chromatogram.data_analysis = self.data_analysis # Ensure plot uses new data analysis instance
            self.chromatogram.update_plot([], [], []) # Clear plot for new run
            self.chromatogram.update_realtime_data_labels(0, 0, 0, 0) # Reset labels

            # Start hardware simulation
            self.hardware.start_run()

            # Start data collection timer
            self.data_timer.start()

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
            self.data_timer.stop()
            self.hardware.pause_run()
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
            self.data_timer.stop()
            self.hardware.stop_run()

            # Save run data if a run was active
            if self.current_run_data['run_id']:
                self.save_run_data()
                self.prompt_data_export() # Prompt for export after saving

            self.update_system_state(SystemState.IDLE)
            logger.info("Run stopped.")
            self.status_bar.showMessage("Run stopped.")

            # Clear current run data
            self.current_run_data = {
                'run_id': None,
                'start_time': None,
                'method_name': None
            }
            self.data_analysis = DataAnalysis() # Reset data analysis for next run
            self.chromatogram.data_analysis = self.data_analysis
            self.chromatogram.update_plot([], [], []) # Clear plot
            self.chromatogram.update_realtime_data_labels(0, 0, 0, 0) # Reset labels

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
            self.data_timer.stop()
            self.hardware.emergency_stop()
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
            self.chromatogram.update_realtime_data_labels(0, 0, 0, 0)

        except Exception as e:
            logger.critical(f"Emergency stop error: {str(e)}")
            QMessageBox.critical(self, "Emergency Stop Error", f"Error during emergency stop: {str(e)}")

    def update_data(self):
        """Updates real-time data displays and collects data."""
        try:
            data = self.hardware.get_current_data()
            self.on_data_received(
                data['time'], data['uv'], data['pressure'], data['conductivity'],
                data['flow_rate'], data['buffer_a'], data['buffer_b'],
                data['temperature'], data['ph']
            )
        except Exception as e:
            logger.error(f"Data update error: {str(e)}")
            self.data_timer.stop()
            self.update_system_state(SystemState.ERROR)
            self.alarm_manager.add_alarm("Error", f"Real-time data update failed: {str(e)}")

    def handle_alarm(self, severity, message):
        """Handles system alarms by displaying a message box."""
        if severity == "Critical":
            # Critical alarms might trigger emergency stop, already handled in check_system_status
            pass
        QMessageBox.warning(
            self,
            f"{severity} Alarm",
            message
        )
        logger.warning(f"Alarm handled: {severity} - {message}")

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

        if not self.hardware.connected:
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

    # --- Add Login Check Here ---
    login_dialog = LoginDialog()
    # Show the login dialog and wait for user interaction
    if login_dialog.exec() == QDialog.DialogCode.Accepted:
        # If login is successful, proceed to launch the main application
        logging.info("User successfully logged in.")
        window = IndustrialFPLCSystem() # Assuming IndustrialFPLCSystem is your main window class
        window.showMaximized() # Or window.show()
        sys.exit(app.exec())
    else:
        # If login failed or was cancelled (e.g., user closed the dialog)
        logging.warning("Login failed or cancelled. Exiting application.")
        sys.exit(1) # Exit the application

if __name__ == "__main__":
    main()
