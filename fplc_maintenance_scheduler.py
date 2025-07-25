from fplc_serial_manager import FPLCSerialManager
import time
from datetime import datetime
import logging
import random

class SystemDiagnostics:
    """Handles system diagnostics and maintenance procedures."""
    def __init__(self, serial_manager: FPLCSerialManager):
        self.serial_manager = serial_manager
        self.test_results = {}
        self.maintenance_history = []
        self.logger = logging.getLogger('SystemDiagnostics')
        self.logger.info("System Diagnostics initialized.")

    def run_diagnostics(self):
        """Runs a suite of simulated diagnostic tests."""
        if not self.serial_manager.connected:
            self.logger.error("Cannot run diagnostics: Hardware not connected.")
            raise ConnectionError("Hardware not connected for diagnostics.")

        tests = [
            self._test_pressure_sensor,
            self._test_uv_detector,
            self._test_conductivity_detector,
            self._test_pump_flow,
            self._test_valves
        ]

        self.test_results = {}
        all_passed = True
        for test in tests:
            result = test()
            self.test_results[test.__name__] = result
            if not result:
                all_passed = False
            self.logger.info(f"Diagnostic test '{test.__name__}' result: {'Passed' if result else 'Failed'}")

        self.logger.info(f"All diagnostics completed. Overall result: {'Passed' if all_passed else 'Failed'}")
        return all_passed

    def _test_pressure_sensor(self):
        """Simulates a pressure sensor test by reading value."""
        # In a real system, you'd apply a known pressure and check reading
        pressure_reading = self.serial_manager._sim_pressure # Get current simulated pressure
        self.logger.debug(f"Pressure sensor test: Current reading {pressure_reading:.2f} MPa")
        return random.random() > 0.1 # 90% pass rate

    def _test_uv_detector(self):
        """Simulates a UV detector test."""
        # In a real system, you'd flow a known standard and check response
        uv_reading = self.serial_manager._sim_uv
        self.logger.debug(f"UV detector test: Current reading {uv_reading:.2f} mAU")
        return random.random() > 0.1

    def _test_conductivity_detector(self):
        """Simulates a conductivity detector test."""
        # In a real system, flow known salt solutions
        cond_reading = self.serial_manager._sim_conductivity
        self.logger.debug(f"Conductivity detector test: Current reading {cond_reading:.2f} mS/cm")
        return random.random() > 0.1

    def _test_pump_flow(self):
        """Simulates a pump flow rate test."""
        # In a real system, set a flow rate and measure actual flow (e.g., gravimetrically)
        target_flow = 5.0
        self.serial_manager.send_command(f"SET_FLOW:{target_flow}")
        time.sleep(0.5) # Allow time for simulated flow to stabilize
        actual_flow = self.serial_manager._sim_flow_rate # Get simulated flow
        self.logger.debug(f"Pump flow test: Target {target_flow} mL/min, Actual {actual_flow:.2f} mL/min")
        return abs(target_flow - actual_flow) < 0.5 # Pass if within 0.5 mL/min

    def _test_valves(self):
        """Simulates a valve test by cycling positions."""
        valve_id = "INJ"
        initial_pos = self.serial_manager._sim_valve_positions.get(valve_id, "UNKNOWN")
        self.serial_manager.send_command(f"SET_VALVE:{valve_id}:INJECT")
        time.sleep(0.1)
        pos1 = self.serial_manager._sim_valve_positions.get(valve_id)
        self.serial_manager.send_command(f"SET_VALVE:{valve_id}:LOAD")
        time.sleep(0.1)
        pos2 = self.serial_manager._sim_valve_positions.get(valve_id)
        self.serial_manager.send_command(f"SET_VALVE:{valve_id}:{initial_pos}") # Restore
        self.logger.debug(f"Valve {valve_id} test: cycled from {initial_pos} to {pos1} to {pos2}")
        return pos1 == "INJECT" and pos2 == "LOAD"

    def log_maintenance(self, activity, user):
        """Logs a maintenance activity."""
        entry = {
            'date': datetime.now().isoformat(),
            'activity': activity,
            'user': user
        }
        self.maintenance_history.append(entry)
        self.logger.info(f"Maintenance logged: '{activity}' by {user}.")


class MaintenanceScheduler:
    def __init__(self):
        self.maintenance_tasks = {
            'pump_seals': {'interval': 180, 'last_done': None},
            'detector_calibration': {'interval': 90, 'last_done': None},
            'system_cleaning': {'interval': 30, 'last_done': None}
        }
    
    def check_maintenance_due(self):
        due_tasks = []
        for task, info in self.maintenance_tasks.items():
            if self.is_maintenance_due(task):
                due_tasks.append(task)
        return due_tasks
