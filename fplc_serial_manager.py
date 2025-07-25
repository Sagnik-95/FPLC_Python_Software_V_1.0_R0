import serial
import threading
import time
import logging
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
import random # Added for simulation
import numpy as np # Added for simulation
from enum import Enum

# Assume ErrorSeverity and FPLCError are available or defined here for standalone testing
try:
    from fplc_error_handler import ErrorSeverity, FPLCError, ErrorHandler
except ImportError:
    # Fallback for standalone testing or if error_handler is not available
    class ErrorSeverity(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

    class FPLCError(Exception):
        def __init__(self, message: str, severity: ErrorSeverity, details: dict = None):
            super().__init__(message)
            self.severity = severity
            self.details = details or {}
            self.timestamp = time.time()

    class ErrorHandler:
        def __init__(self, log_file="fplc_errors.log"):
            self.log_file = log_file
            self.error_callbacks = []
            logging.basicConfig(filename=self.log_file, level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')
        def handle_error(self, error: FPLCError):
            logging.error(f"FPLC Error: {error.message} - {error.severity.value} - {error.details}")
            for callback in self.error_callbacks:
                try: callback({'message': error.message, 'severity': error.severity.value, 'details': error.details})
                except Exception as e: logging.error(f"Error in error callback: {e}")
        def add_error_callback(self, callback): self.error_callbacks.append(callback)


logger = logging.getLogger('FPLCSerialManager')

class FPLCSerialManager(QObject):
    """
    Manages serial communication with the FPLC hardware.
    Simulates hardware responses if no physical connection.
    """
    data_received = pyqtSignal(float, float, float, float, float, float, float, float, float) # time, uv, pressure, conductivity, flow, bufferA, bufferB, temp, pH
    connection_status_changed = pyqtSignal(bool)
    error_occurred = pyqtSignal(str, str) # No longer needed, use ErrorHandler

    def __init__(self, com_port: str = "COM3", baud_rate: int = 9600, error_handler: ErrorHandler = None):
        super().__init__()
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.serial_connection = None
        self.connected = False
        self.receiving_data = False
        self.read_thread = None
        self.error_handler = error_handler if error_handler else ErrorHandler() # Use provided or default
        self.logger = logging.getLogger('FPLCSerialManager')

        # Simulated data for testing without hardware
        self._sim_time = 0.0
        self._sim_uv = 0.0
        self._sim_pressure = 0.0
        self._sim_conductivity = 0.0
        self._sim_flow_rate = 0.0
        self._sim_buffer_a = 100.0
        self._sim_buffer_b = 0.0
        self._sim_temperature = 25.0
        self._sim_ph = 7.0

        # Control parameters for simulation
        self._target_flow_rate = 0.0
        self._target_buffer_a = 100.0
        self._target_pressure = 0.0 # New: Target pressure for simulation

        self.control_timer = QtCore.QTimer(self) # Use QTimer for simulation updates
        self.control_timer.setInterval(100) # Update every 100 ms (0.1 seconds)
        self.control_timer.timeout.connect(self._simulate_data)
        self.control_timer.start() # Start simulation immediately

        self.logger.info(f"FPLCSerialManager initialized for {com_port} at {baud_rate} baud.")

    def connect(self) -> bool:
        """Establishes serial connection or simulates connection."""
        if self.connected:
            self.logger.info("Already connected.")
            return True
        try:
            # Attempt to connect to physical serial port
            # self.serial_connection = serial.Serial(self.com_port, self.baud_rate, timeout=1)
            self.connected = True
            self.connection_status_changed.emit(True)
            self.logger.info(f"Connected to {self.com_port}.")

            # Start read thread if using physical hardware
            # self.receiving_data = True
            # self.read_thread = threading.Thread(target=self._read_from_port)
            # self.read_thread.daemon = True
            # self.read_thread.start()
            return True
        except serial.SerialException as e:
            self.error_handler.handle_error(FPLCError(
                f"Failed to connect to serial port {self.com_port}: {str(e)}. Simulating data.",
                ErrorSeverity.WARNING,
                {'port': self.com_port, 'error_type': 'SerialException'}
            ))
            self.connected = True # Still set to connected for simulation mode
            self.connection_status_changed.emit(True)
            self.logger.warning("Serial connection failed, proceeding with simulation.")
            return True # Return True to allow the app to run in simulation mode
        except Exception as e:
            self.error_handler.handle_error(FPLCError(
                f"An unexpected error occurred during serial connection: {str(e)}",
                ErrorSeverity.CRITICAL,
                {'port': self.com_port, 'error_type': 'UnexpectedError'}
            ))
            self.connected = False
            self.connection_status_changed.emit(False)
            return False

    def disconnect(self) -> None:
        """Closes serial connection."""
        if self.connected:
            self.receiving_data = False
            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join(timeout=1) # Wait for thread to finish
            if self.serial_connection:
                self.serial_connection.close()
            self.connected = False
            self.connection_status_changed.emit(False)
            self.logger.info("Disconnected from serial port.")
        self.control_timer.stop() # Stop simulation on disconnect

    def send_command(self, command: str) -> None:
        """Sends a command to the FPLC hardware or simulates it."""
        self.logger.info(f"Sending command: {command}")
        if self.connected:
            # In a real scenario, send command via serial
            # if self.serial_connection:
            #     self.serial_connection.write(command.encode() + b'\n')

            # Simulate command effects
            if command.startswith("SET_FLOW:"):
                try:
                    self._target_flow_rate = float(command.split(":")[1])
                    self.logger.debug(f"Simulated target flow rate set to: {self._target_flow_rate}")
                except ValueError:
                    self.error_handler.handle_error(FPLCError(
                        f"Invalid flow rate command: {command}",
                        ErrorSeverity.WARNING,
                        {'command': command}
                    ))
            elif command.startswith("SET_BUFFER_A:"):
                try:
                    self._target_buffer_a = float(command.split(":")[1])
                    self.logger.debug(f"Simulated target buffer A set to: {self._target_buffer_a}")
                except ValueError:
                    self.error_handler.handle_error(FPLCError(
                        f"Invalid buffer A command: {command}",
                        ErrorSeverity.WARNING,
                        {'command': command}
                    ))
            elif command == "FC_START":
                self.logger.info("Simulating Fraction Collector START.")
            elif command == "FC_STOP":
                self.logger.info("Simulating Fraction Collector STOP.")
            elif command == "FC_RESET":
                self.logger.info("Simulating Fraction Collector RESET.")
            else:
                self.logger.warning(f"Unknown or unhandled simulated command: {command}")
        else:
            self.error_handler.handle_error(FPLCError(
                f"Attempted to send command '{command}' but not connected.",
                ErrorSeverity.WARNING,
                {'command': command, 'status': 'disconnected'}
            ))

    def _read_from_port(self):
        """Reads data from the serial port in a separate thread (for real hardware)."""
        while self.receiving_data:
            try:
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line:
                        self._parse_and_emit_data(line)
            except serial.SerialException as e:
                self.error_handler.handle_error(FPLCError(
                    f"Serial read error: {str(e)}",
                    ErrorSeverity.ERROR,
                    {'component': 'SerialManager', 'action': 'read_thread'}
                ))
                self.disconnect()
                break
            except Exception as e:
                self.error_handler.handle_error(FPLCError(
                    f"Error in serial read thread: {str(e)}",
                    ErrorSeverity.ERROR,
                    {'component': 'SerialManager', 'action': 'read_thread'}
                ))
                self.disconnect()
                break
            time.sleep(0.01) # Small delay to prevent busy-waiting

    def _parse_and_emit_data(self, data_string: str):
        """Parses a data string and emits the data_received signal."""
        try:
            # Example format: "TIME:1.0,UV:100.5,PRESS:0.5,COND:1.2,FLOW:1.0,BUFA:100,BUFB:0,TEMP:25.0,PH:7.0"
            parts = data_string.split(',')
            data_dict = {}
            for part in parts:
                key_value = part.split(':')
                if len(key_value) == 2:
                    data_dict[key_value[0].strip()] = float(key_value[1].strip())

            time_point = data_dict.get('TIME', 0.0)
            uv = data_dict.get('UV', 0.0)
            pressure = data_dict.get('PRESS', 0.0)
            conductivity = data_dict.get('COND', 0.0)
            flow = data_dict.get('FLOW', 0.0)
            buffer_a = data_dict.get('BUFA', 0.0)
            buffer_b = data_dict.get('BUFB', 0.0)
            temperature = data_dict.get('TEMP', 0.0)
            ph = data_dict.get('PH', 0.0)

            self.data_received.emit(time_point, uv, pressure, conductivity, flow, buffer_a, buffer_b, temperature, ph)
        except ValueError as e:
            self.error_handler.handle_error(FPLCError(
                f"Failed to parse data string '{data_string}': {str(e)}",
                ErrorSeverity.WARNING,
                {'data_string': data_string, 'error_type': 'ValueError'}
            ))
        except Exception as e:
            self.error_handler.handle_error(FPLCError(
                f"An unexpected error occurred during data parsing: {str(e)}",
                ErrorSeverity.ERROR,
                {'data_string': data_string, 'error_type': 'UnexpectedError'}
            ))

    def _simulate_data(self):
        """Simulates FPLC data based on internal state and target values."""
        if not self.connected:
            return

        # Increment time
        self._sim_time += (self.control_timer.interval() / 1000.0) / 60.0 # Convert ms to minutes

        # Simulate flow rate approaching target
        if abs(self._sim_flow_rate - self._target_flow_rate) > 0.01:
            self._sim_flow_rate += (self._target_flow_rate - self._sim_flow_rate) * 0.1 + random.uniform(-0.01, 0.01)
        else:
            self._sim_flow_rate = self._target_flow_rate + random.uniform(-0.01, 0.01)

        # Simulate buffer A approaching target
        if abs(self._sim_buffer_a - self._target_buffer_a) > 0.1:
            self._sim_buffer_a += (self._target_buffer_a - self._sim_buffer_a) * 0.1 + random.uniform(-0.1, 0.1)
        else:
            self._sim_buffer_a = self._target_buffer_a + random.uniform(-0.1, 0.1)
        self._sim_buffer_b = 100.0 - self._sim_buffer_a

        # Simulate UV absorbance (e.g., a simple peak or baseline)
        # For a simple simulation, let's make UV fluctuate around a baseline,
        # and maybe simulate a peak if flow rate is stable for a while.
        baseline_uv = 10.0
        peak_uv = 100.0
        peak_start_time = 5.0
        peak_duration = 3.0

        if self._sim_time > peak_start_time and self._sim_time < (peak_start_time + peak_duration):
            # Simulate a Gaussian-like peak
            peak_center = peak_start_time + peak_duration / 2
            deviation = (self._sim_time - peak_center) / (peak_duration / 4)
            self._sim_uv = baseline_uv + peak_uv * np.exp(-0.5 * deviation**2) + random.uniform(-1, 1)
        else:
            self._sim_uv = baseline_uv + random.uniform(-0.5, 0.5)

        # Simulate pressure (proportional to flow rate, plus some noise)
        # Add a target pressure based on current flow and method max pressure
        # For simplicity, let's assume pressure increases with flow and buffer B percentage (viscosity)
        self._sim_pressure = (self._sim_flow_rate * 0.1) + (self._sim_buffer_b * 0.005) + random.uniform(-0.02, 0.02)
        self._sim_pressure = max(0.0, self._sim_pressure) # Pressure cannot be negative

        # Simulate conductivity (changes with buffer B, and some noise)
        self._sim_conductivity = (self._sim_buffer_b * 0.1) + random.uniform(-0.1, 0.1)
        self._sim_conductivity = max(0.0, self._sim_conductivity) # Conductivity cannot be negative

        # Simulate temperature and pH (relatively stable, with minor fluctuations)
        self._sim_temperature = 25.0 + random.uniform(-0.1, 0.1)
        self._sim_ph = 7.0 + (self._sim_buffer_b * 0.01) + random.uniform(-0.05, 0.05) # pH changes slightly with buffer B
        self._sim_ph = max(0.0, min(14.0, self._sim_ph)) # pH bounds

        # Emit the simulated data
        self.data_received.emit(
            self._sim_time,
            self._sim_uv,
            self._sim_pressure,
            self._sim_conductivity,
            self._sim_flow_rate,
            self._sim_buffer_a,
            self._sim_buffer_b,
            self._sim_temperature,
            self._sim_ph
        )

    def get_current_simulated_data(self):
        """Returns the current simulated data values."""
        return {
            'time': self._sim_time,
            'uv': self._sim_uv,
            'pressure': self._sim_pressure,
            'conductivity': self._sim_conductivity,
            'flow_rate': self._sim_flow_rate,
            'buffer_a': self._sim_buffer_a,
            'buffer_b': self._sim_buffer_b,
            'temperature': self._sim_temperature,
            'ph': self._sim_ph
        }

    def set_target_flow_rate(self, flow_rate: float):
        """Sets the target flow rate for simulation."""
        self._target_flow_rate = flow_rate
        self.logger.debug(f"Target flow rate updated to {flow_rate} mL/min.")

    def set_target_buffer_a(self, buffer_a_percent: float):
        """Sets the target buffer A percentage for simulation."""
        self._target_buffer_a = buffer_a_percent
        self.logger.debug(f"Target buffer A updated to {buffer_a_percent} %.")

    def reset_simulation(self):
        """Resets all simulated data values."""
        self._sim_time = 0.0
        self._sim_uv = 0.0
        self._sim_pressure = 0.0
        self._sim_conductivity = 0.0
        self._sim_flow_rate = 0.0
        self._sim_buffer_a = 100.0
        self._sim_buffer_b = 0.0
        self._sim_temperature = 25.0
        self._sim_ph = 7.0
        self._target_flow_rate = 0.0
        self._target_buffer_a = 100.0
        self.logger.info("FPLCSerialManager simulation data reset.")

