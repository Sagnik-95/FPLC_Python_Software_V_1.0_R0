from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
import logging
import random
from enum import Enum
import collections # For deque
import time
from fplc_serial_manager import FPLCSerialManager
from fplc_method import FPLCMethod
from fplc_error_handler import ErrorHandler

class FPLCPump:
    """Represents and controls a simulated FPLC pump."""
    def __init__(self, serial_manager: FPLCSerialManager, pump_id: str):
        self.serial_manager = serial_manager
        self.pump_id = pump_id
        self._target_flow_rate = 0.0 # mL/min
        self._current_flow_rate = 0.0 # Actual reported flow rate (simulated)
        self._current_pressure = 0.0 # Actual reported pressure (simulated)
        self.logger = logging.getLogger(f'FPLCPump-{pump_id}')

    def set_flow_rate(self, flow_rate_ml_per_min: float):
        """Sends command to set pump flow rate."""
        self._target_flow_rate = max(0.0, flow_rate_ml_per_min) # Ensure non-negative
        command = f"SET_FLOW:{self._target_flow_rate:.2f}"
        response = self.serial_manager.send_command(command)
        if "OK" in response:
            self.logger.debug(f"Pump {self.pump_id}: Flow rate command sent: {self._target_flow_rate} mL/min.")
            return True
        else:
            self.logger.error(f"Pump {self.pump_id}: Failed to set flow rate. Response: {response}")
            return False

    def get_current_flow_rate(self):
        # In a real system, you'd query the pump for its actual flow rate
        # For simulation, we'll assume the manager updates it based on commands
        return self.serial_manager._sim_flow_rate # Access simulated value directly for simplicity

    def get_current_pressure(self):
        # In a real system, you'd query a pressure sensor
        return self.serial_manager._sim_pressure # Access simulated value directly for simplicity

    def set_buffer_percentage(self, buffer_a_percent: float):
        """Sends command to set buffer A percentage (and implicitly buffer B)."""
        buffer_a_percent = max(0.0, min(100.0, buffer_a_percent)) # Clamp between 0-100
        command = f"SET_BUFFER_A:{buffer_a_percent:.1f}"
        response = self.serial_manager.send_command(command)
        if "OK" in response:
            self.logger.debug(f"Pump {self.pump_id}: Buffer A command sent: {buffer_a_percent}%.")
            return True
        else:
            self.logger.error(f"Pump {self.pump_id}: Failed to set buffer A. Response: {response}")
            return False

class FPLCValve:
    """Represents and controls a simulated FPLC valve."""
    def __init__(self, serial_manager: FPLCSerialManager, valve_id: str):
        self.serial_manager = serial_manager
        self.valve_id = valve_id
        self._current_position = "UNKNOWN" # e.g., "LOAD", "INJECT", "WASTE"
        self.logger = logging.getLogger(f'FPLCValve-{valve_id}')

    def set_position(self, position: str):
        """Sends command to set valve position."""
        command = f"SET_VALVE:{self.valve_id}:{position}"
        response = self.serial_manager.send_command(command)
        if "OK" in response:
            self._current_position = position
            self.logger.info(f"Valve {self.valve_id} set to {position}.")
            return True
        else:
            self.logger.error(f"Valve {self.valve_id}: Failed to set position. Response: {response}")
            return False

    def get_current_position(self):
        # In a real system, you'd query the valve for its actual position
        return self._current_position # For simulation, assume it's set correctly

class FPLCDetector:
    """Represents and reads data from a simulated FPLC detector (UV/Conductivity/pH)."""
    def __init__(self, serial_manager: FPLCSerialManager, detector_type: str):
        self.serial_manager = serial_manager
        self.detector_type = detector_type
        self._current_reading = 0.0
        self.logger = logging.getLogger(f'FPLCDetector-{detector_type}')

    def get_reading(self):
        # In a real system, you'd query the detector
        if self.detector_type == "UV":
            return self.serial_manager._sim_uv
        elif self.detector_type == "Conductivity":
            return self.serial_manager._sim_conductivity
        elif self.detector_type == "pH":
            return self.serial_manager._sim_ph
        return 0.0 # Default

class FPLCController(QObject):
    """
    Manages the FPLC run, executes methods, and implements MPC-like control logic.
    """
    # Signals to communicate with the main UI
    current_step_changed = pyqtSignal(str, int, float) # step_type, step_index, elapsed_time_in_step
    run_progress_updated = pyqtSignal(int) # percentage
    run_finished = pyqtSignal()
    control_data_updated = pyqtSignal(float, float, float, float) # set_flow, actual_flow, set_buffer_A, actual_buffer_A
    error_occurred = pyqtSignal(str)

    def __init__(self, serial_manager: FPLCSerialManager, method: FPLCMethod, parent=None):
        super().__init__(parent)
        self.serial_manager = serial_manager
        self.method = method
        self.logger = logging.getLogger('FPLCController')

        # Hardware instances
        self.pump_a = FPLCPump(serial_manager, "A")
        self.pump_b = FPLCPump(serial_manager, "B")
        self.injection_valve = FPLCValve(serial_manager, "INJ")
        self.column_valve = FPLCValve(serial_manager, "COL") # Example: for switching columns
        self.fraction_valve = FPLCValve(serial_manager, "FRAC") # Example: for diverting to waste/collector
        self.uv_detector = FPLCDetector(serial_manager, "UV")
        self.conductivity_detector = FPLCDetector(serial_manager, "Conductivity")
        self.ph_detector = FPLCDetector(serial_manager, "pH")
        self.error_handler = ErrorHandler()  # Store it if you need it

        # Run state
        self._is_running = False
        self._is_paused = False
        self._current_step_index = -1
        self._step_start_time = 0.0 # time.time() when step started
        self._run_start_time = 0.0  # time.time() when run started

        # Control loop timer
        self.control_timer = QTimer(self)
        self.control_timer.setInterval(1000) # Control loop runs every 1 second
        self.control_timer.timeout.connect(self._control_loop_iteration)

        # MPC-like control variables
        self._target_total_flow = 0.0
        self._target_buffer_a_percent = 0.0
        self._current_total_flow = 0.0
        self._current_buffer_a_percent = 0.0
        self._pressure_setpoint = 1.0 # MPa (arbitrary, adjust based on method/column)
        self._pressure_history = collections.deque(maxlen=10) # For simple trend analysis

        # PID-like gains for simple control (tune these for real hardware)
        self.flow_kp = 0.1
        self.gradient_kp = 0.1

    def start_run(self):
        if not self.serial_manager.connected:
            self.error_occurred.emit("Hardware not connected. Cannot start run.")
            return

        if self._is_running:
            self.logger.warning("Run already in progress.")
            return

        self._is_running = True
        self._is_paused = False
        self._current_step_index = 0
        self._run_start_time = time.time()
        self._step_start_time = time.time() # Start time for the first step

        self.logger.info(f"FPLC Controller: Starting run with method '{self.method.name}'.")
        self.control_timer.start() # Start the control loop
        self._execute_current_step() # Immediately execute the first step

    def pause_run(self):
        if self._is_running and not self._is_paused:
            self._is_paused = True
            self.control_timer.stop()
            self.logger.info("FPLC Controller: Run paused.")
            # Send commands to halt pumps, close valves etc.
            self.pump_a.set_flow_rate(0)
            self.pump_b.set_flow_rate(0)

    def resume_run(self):
        if self._is_running and self._is_paused:
            self._is_paused = False
            # Adjust step_start_time to account for pause duration
            pause_duration = time.time() - self._step_start_time
            self._step_start_time += pause_duration # Shift start time forward
            self._run_start_time += pause_duration # Shift run start time forward

            self.logger.info("FPLC Controller: Resuming run.")
            self.control_timer.start() # Resume the control loop
            # Re-apply last known control actions or current step parameters
            self._apply_current_step_parameters()

    def stop_run(self):
        if self._is_running:
            self._is_running = False
            self._is_paused = False
            self.control_timer.stop()
            self.logger.info("FPLC Controller: Stopping run.")
            # Send commands to fully stop hardware
            self.pump_a.set_flow_rate(0)
            self.pump_b.set_flow_rate(0)
            self.injection_valve.set_position("WASTE") # Ensure safe state
            self.column_valve.set_position("BYPASS") # Or other safe state
            self.fraction_valve.set_position("WASTE") # Divert to waste

            self.run_finished.emit() # Signal that the run has finished

    def emergency_stop(self):
        self.logger.critical("FPLC Controller: EMERGENCY STOP initiated!")
        self.stop_run() # Call regular stop to halt
        # Additional critical hardware specific commands if needed (e.g., power off specific modules)
        self.serial_manager.disconnect() # Force disconnect
        self.error_occurred.emit("EMERGENCY STOP: System halted due to critical error.")


    def _control_loop_iteration(self):
        """
        Main control loop that runs periodically.
        Reads sensor data, applies MPC-like logic, and sends commands.
        """
        if not self._is_running or self._is_paused:
            return

        current_time_in_step = (time.time() - self._step_start_time) / 60.0 # in minutes
        current_run_time = (time.time() - self._run_start_time) / 60.0 # in minutes

        # 1. Read current sensor data
        current_uv = self.uv_detector.get_reading()
        current_pressure = self.pump_a.get_current_pressure() # Assuming pressure is common to both pumps
        current_conductivity = self.conductivity_detector.get_reading()
        current_ph = self.ph_detector.get_reading()

        # Update pressure history for simple trend
        self._pressure_history.append(current_pressure)

        # 2. Apply MPC-like logic based on current step parameters
        current_step = self.method.steps[self._current_step_index]
        self._target_total_flow = current_step['flow_rate']
        self._target_buffer_a_percent = current_step['buffer_a']

        # --- Pressure Control (Simple Feedback) ---
        # Adjust total flow rate based on pressure feedback
        # If pressure is too high, reduce flow. If too low, increase flow.
        pressure_error = self._pressure_setpoint - current_pressure
        # Simple proportional control for flow adjustment
        flow_adjustment_from_pressure = pressure_error * self.flow_kp

        # Apply flow adjustment, but ensure it doesn't deviate too much from target
        adjusted_total_flow = self._target_total_flow + flow_adjustment_from_pressure
        adjusted_total_flow = max(0.1, min(adjusted_total_flow, 100.0)) # Clamp flow rate

        # --- Gradient Control (Simple Feedback) ---
        # Adjust individual pump speeds to achieve target buffer percentage
        # Assuming current_conductivity gives an indication of buffer B percentage
        # This is a very simplified model. A real model would be more complex.
        actual_buffer_b_from_cond = (current_conductivity / 50.0) * 100.0 # Reverse engineer from max cond
        actual_buffer_a_from_cond = 100.0 - actual_buffer_b_from_cond

        gradient_error = self._target_buffer_a_percent - actual_buffer_a_from_cond
        # Proportional adjustment for buffer A pump
        buffer_a_adjustment = gradient_error * self.gradient_kp

        # Calculate individual pump flow rates based on adjusted total flow and target percentages
        # Ensure percentages sum to 100 for calculation
        target_buffer_b_percent = 100.0 - self._target_buffer_a_percent

        flow_a = (adjusted_total_flow * (self._target_buffer_a_percent + buffer_a_adjustment)) / 100.0
        flow_b = (adjusted_total_flow * (target_buffer_b_percent - buffer_a_adjustment)) / 100.0

        # Ensure flows are non-negative and sum up to total flow (approx)
        flow_a = max(0.0, flow_a)
        flow_b = max(0.0, flow_b)
        # Re-normalize if they don't sum up due to clamping
        total_actual_flow_sum = flow_a + flow_b
        if total_actual_flow_sum > 0:
            scale_factor = adjusted_total_flow / total_actual_flow_sum
            flow_a *= scale_factor
            flow_b *= scale_factor

        # 3. Send commands to hardware
        self.pump_a.set_flow_rate(flow_a)
        self.pump_b.set_flow_rate(flow_b)
        self.pump_a.set_buffer_percentage(self._target_buffer_a_percent) # This command is for the mixer

        # Update current flow and buffer A for UI display
        self._current_total_flow = flow_a + flow_b
        self._current_buffer_a_percent = self._target_buffer_a_percent # Assume mixer responds instantly

        self.control_data_updated.emit(
            adjusted_total_flow, # Set flow
            self._current_total_flow, # Actual total flow
            self._target_buffer_a_percent, # Set buffer A
            actual_buffer_a_from_cond # Actual buffer A from conductivity
        )

        # 4. Check step completion
        if current_time_in_step >= current_step['duration']:
            self._current_step_index += 1
            if self._current_step_index < len(self.method.steps):
                self._step_start_time = time.time() # Reset step start time
                self._execute_current_step()
            else:
                self.stop_run() # All steps completed

        # Update run progress
        total_duration = sum(s['duration'] for s in self.method.steps)
        if total_duration > 0:
            progress_percent = int((current_run_time / total_duration) * 100)
            self.run_progress_updated.emit(min(100, progress_percent))

        self.current_step_changed.emit(
            current_step['type'],
            self._current_step_index + 1,
            current_time_in_step
        )

    def _execute_current_step(self):
        """Applies parameters for the current method step."""
        if self._current_step_index < 0 or self._current_step_index >= len(self.method.steps):
            self.logger.error("Attempted to execute an invalid step index.")
            self.stop_run()
            return

        step = self.method.steps[self._current_step_index]
        self.logger.info(f"Executing Step {self._current_step_index + 1}: {step['type']}")

        # Apply parameters to hardware components
        self._target_total_flow = step['flow_rate']
        self._target_buffer_a_percent = step['buffer_a']

        # Set pump flow rates and buffer percentages
        # The control loop will continuously adjust these
        self.pump_a.set_flow_rate(step['flow_rate'] * (step['buffer_a'] / 100.0))
        self.pump_b.set_flow_rate(step['flow_rate'] * (step['buffer_b'] / 100.0))
        self.pump_a.set_buffer_percentage(step['buffer_a']) # This command is for the mixer

        # Control injection valve (simplified)
        if step['step_type'] == "Sample Load":
            self.injection_valve.set_position("INJECT")
            self.fraction_valve.set_position("WASTE") # Divert sample to waste initially
        else:
            self.injection_valve.set_position("LOAD") # Prepare for next sample load

        # Control fraction collector valve
        if step['collection']:
            self.fraction_valve.set_position("COLLECT")
            # In a real system, you'd also tell the fraction collector to start
            self.serial_manager.send_command("FC_START")
        else:
            self.fraction_valve.set_position("WASTE")
            self.serial_manager.send_command("FC_STOP")

        # Update current pressure setpoint based on method's max pressure
        # This is a simplification. Real systems have more dynamic pressure targets.
        self._pressure_setpoint = self.method.max_pressure * 0.8 # Target 80% of max pressure

        self.current_step_changed.emit(
            step['type'],
            self._current_step_index + 1,
            0.0 # Start time in step
        )

    def _apply_current_step_parameters(self):
        """Re-applies the parameters of the current step, useful after pause/resume."""
        if self._current_step_index >= 0 and self._current_step_index < len(self.method.steps):
            step = self.method.steps[self._current_step_index]
            self.pump_a.set_flow_rate(step['flow_rate'] * (step['buffer_a'] / 100.0))
            self.pump_b.set_flow_rate(step['flow_rate'] * (step['buffer_b'] / 100.0))
            self.pump_a.set_buffer_percentage(step['buffer_a']) # This command is for the mixer

            if step['step_type'] == "Sample Load":
                self.injection_valve.set_position("INJECT")
            else:
                self.injection_valve.set_position("LOAD")

            if step['collection']:
                self.fraction_valve.set_position("COLLECT")
                self.serial_manager.send_command("FC_START")
            else:
                self.fraction_valve.set_position("WASTE")
                self.serial_manager.send_command("FC_STOP")
        else:
            self.logger.warning("No valid current step to re-apply parameters.")