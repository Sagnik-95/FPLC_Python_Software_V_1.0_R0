import random
from datetime import datetime
import csv
import numpy as np
import logging
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QGroupBox, QTableWidget, QTextEdit,
    QTableWidgetItem, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QLabel, QWidget, QDialog, QFormLayout, QTabWidget, QRadioButton,
    QMenuBar, QStatusBar, QComboBox, QMessageBox, QFileDialog, QCheckBox,
    QApplication, QInputDialog, QProgressBar
)
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

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
        self.logger = logging.getLogger('ColumnQualification')

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
            self.logger.info(f"Column qualification completed for {column_type}. Passed: {test_data['passed']}")
            return test_data

        except Exception as e:
            self.logger.error(f"Column qualification error: {str(e)}")
            raise

    def calculate_hetp(self, flow_rates):
        """Calculates simulated HETP values for given flow rates (Van Deemter curve)."""
        A = 0.5  # Eddy diffusion
        B = 0.1  # Longitudinal diffusion
        C = 0.05 # Mass transfer resistance
        hetp_values = [A + B / flow_rate + C * flow_rate for flow_rate in flow_rates if flow_rate > 0]
        return hetp_values

    def calculate_symmetry(self, flow_rates):
        """Calculates simulated symmetry values for given flow rates."""
        # Simple simulation: symmetry tends to degrade at very high/low flow rates
        symmetry_values = []
        for flow_rate in flow_rates:
            if flow_rate < 0.3:
                symmetry_values.append(random.uniform(1.2, 1.5))
            elif flow_rate > 0.8:
                symmetry_values.append(random.uniform(1.1, 1.4))
            else:
                symmetry_values.append(random.uniform(0.9, 1.1))
        return symmetry_values

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

    def _measure_peak_capacity(self):
        """Measures simulated peak capacity."""
        return random.uniform(40, 60)

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
                        f"{results.get('theoretical_plates', 'N/A'):.1f}",
                        f"{results.get('asymmetry', 'N/A'):.2f}",
                        f"{results.get('resolution', 'N/A'):.2f}",
                        f"{results.get('pressure_drop', 'N/A'):.2f}",
                        f"{results.get('peak_capacity', 'N/A'):.1f}",
                        data['passed']
                    ])
            self.logger.info(f"Qualification history exported to {filename}.")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting qualification history: {str(e)}")
            return False
    
class ColumnQualificationUI(QGroupBox):
    """UI for column qualification testing."""
    def __init__(self, qualification_manager, parent=None):
        super().__init__("Column Qualification", parent)
        self.qualification_manager = qualification_manager
        self.logger = logging.getLogger('ColumnQualificationUI')
        self.setup_ui()
        self.logger.info("Column Qualification UI initialized.")

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
            self.logger.error(f"Qualification process error: {str(e)}")
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