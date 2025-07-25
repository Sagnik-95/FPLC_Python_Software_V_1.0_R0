import numpy as np
import logging
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Protocol

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SystemTestSuite')

# --- Custom Exceptions ---
class TestSuiteError(Exception):
    """Base exception for SystemTestSuite related errors."""
    pass

class HardwareInteractionError(TestSuiteError):
    """Exception raised for errors during interaction with hardware."""
    pass

class TestConfigError(TestSuiteError):
    """Exception raised for invalid test configuration."""
    pass

class TestSetupError(TestSuiteError):
    """Exception raised if test setup (e.g., priming) fails."""
    pass

# --- Configuration for Tests ---
@dataclass
class TestConfig:
    """
    Configuration parameters for various system tests.
    """
    # Pressure Stability Test
    pressure_test_duration_sec: int = 60
    pressure_test_sample_rate_hz: int = 5 # Samples per second
    pressure_stability_threshold_mpa: float = 0.05 # Max allowed deviation from mean pressure

    # Flow Accuracy Test
    flow_test_target_flow_ml_min: float = 1.0
    flow_test_duration_sec: int = 30
    flow_test_sample_rate_hz: int = 2
    flow_accuracy_tolerance_ml_min: float = 0.02 # Max allowed difference from target flow

    # Detector Baseline Test
    detector_baseline_test_duration_sec: int = 120
    detector_baseline_test_sample_rate_hz: int = 1
    detector_baseline_drift_limit_au: float = 0.001 # Max allowed peak-to-peak drift
    detector_baseline_noise_limit_au: float = 0.0001 # Max allowed RMS noise (optional, but good)

    # General Test Parameters
    warm_up_time_sec: int = 30 # Time for system to stabilize before starting tests

    def __post_init__(self):
        # Basic validation for configuration parameters
        if self.pressure_test_duration_sec <= 0 or self.pressure_test_sample_rate_hz <= 0:
            raise TestConfigError("Pressure test duration and sample rate must be positive.")
        if self.pressure_stability_threshold_mpa < 0:
            raise TestConfigError("Pressure stability threshold cannot be negative.")

        if self.flow_test_duration_sec <= 0 or self.flow_test_sample_rate_hz <= 0:
            raise TestConfigError("Flow test duration and sample rate must be positive.")
        if self.flow_test_target_flow_ml_min <= 0 or self.flow_accuracy_tolerance_ml_min < 0:
            raise TestConfigError("Flow target and tolerance must be non-negative, target must be positive.")

        if self.detector_baseline_test_duration_sec <= 0 or self.detector_baseline_test_sample_rate_hz <= 0:
            raise TestConfigError("Detector baseline test duration and sample rate must be positive.")
        if self.detector_baseline_drift_limit_au < 0 or self.detector_baseline_noise_limit_au < 0:
            raise TestConfigError("Detector limits cannot be negative.")
        
        if self.warm_up_time_sec < 0:
            raise TestConfigError("Warm-up time cannot be negative.")


# --- Data Model for Test Results ---
@dataclass
class TestResult:
    """
    Represents the outcome and details of a single system test.
    """
    test_name: str
    status: str # "PASS", "FAIL", "ERROR", "SKIP"
    message: str = ""
    start_time: Optional[datetime] = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict) # Contains metrics, thresholds, raw data if needed

    def __post_init__(self):
        if not self.test_name:
            raise ValueError("Test name cannot be empty.")
        if self.status not in ["PASS", "FAIL", "ERROR", "SKIP"]:
            raise ValueError(f"Invalid test status: {self.status}")
        if self.start_time and not isinstance(self.start_time, datetime):
            raise ValueError("Start time must be a datetime object.")
        if self.end_time and not isinstance(self.end_time, datetime):
            raise ValueError("End time must be a datetime object.")
        if not isinstance(self.details, dict):
            raise ValueError("Details must be a dictionary.")

    def duration_sec(self) -> float:
        """Calculates the duration of the test in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

# --- Hardware Interface Abstraction ---
class FPLCSystemHardware(Protocol):
    """
    An abstract interface for interacting with FPLC hardware components.
    This allows for mocking the hardware during testing.
    """
    def get_pressure(self) -> float:
        """Returns the current pressure reading in MPa."""
        ...

    def set_flow_rate(self, flow_rate_ml_min: float):
        """Sets the target flow rate in mL/min."""
        ...

    def get_flow_rate(self) -> float:
        """Returns the current measured flow rate in mL/min."""
        ...

    def get_detector_signal(self) -> float:
        """Returns the current detector signal (e.g., UV absorbance in AU)."""
        ...

    def prime_pump(self, duration_sec: int = 10):
        """Primes the pump for a specified duration."""
        ...

    def stop_pump(self):
        """Stops the pump."""
        ...
    
    def zero_detector(self):
        """Performs a detector auto-zero operation."""
        ...

# --- Mock Hardware Implementation (for testing/simulation) ---
class MockHardwareInterface(FPLCSystemHardware):
    """
    A mock implementation of the FPLCSystemHardware interface for testing.
    Simulates hardware responses.
    """
    def __init__(self, initial_pressure: float = 0.5, initial_flow: float = 0.0, initial_detector: float = 0.0):
        self._current_pressure = initial_pressure
        self._current_flow = initial_flow
        self._current_detector_signal = initial_detector
        logger.info("MockHardwareInterface initialized.")

    def get_pressure(self) -> float:
        # Simulate pressure fluctuations around a set point
        self._current_pressure += random.uniform(-0.01, 0.01) # Small random walk
        self._current_pressure = max(0.0, self._current_pressure) # Ensure non-negative
        return self._current_pressure

    def set_flow_rate(self, flow_rate_ml_min: float):
        logger.debug(f"Mock: Setting flow rate to {flow_rate_ml_min} mL/min")
        self._current_flow = flow_rate_ml_min + random.uniform(-0.01, 0.01) # Simulate slight deviation
        time.sleep(0.5) # Simulate delay
        logger.debug(f"Mock: Flow rate set to {self._current_flow:.2f} mL/min")

    def get_flow_rate(self) -> float:
        # Simulate flow reading with minor fluctuations
        return self._current_flow + random.uniform(-0.005, 0.005)

    def get_detector_signal(self) -> float:
        # Simulate detector noise around a baseline
        return self._current_detector_signal + random.uniform(-0.0005, 0.0005)

    def prime_pump(self, duration_sec: int = 10):
        logger.info(f"Mock: Priming pump for {duration_sec} seconds...")
        time.sleep(duration_sec / 2) # Simulate half the time for priming
        self._current_flow = 0.5 + random.uniform(-0.1, 0.1) # Simulate some flow during priming
        time.sleep(duration_sec / 2)
        self._current_flow = 0.0 # Flow stops after priming
        logger.info("Mock: Pump priming complete.")

    def stop_pump(self):
        logger.info("Mock: Stopping pump.")
        self._current_flow = 0.0
        self._current_pressure = 0.0 # Pressure drops
        time.sleep(0.1)

    def zero_detector(self):
        logger.info("Mock: Zeroing detector.")
        self._current_detector_signal = 0.0
        time.sleep(0.5) # Simulate delay


# --- System Test Suite ---
class SystemTestSuite:
    """
    Performs a suite of diagnostic tests on an FPLC system's components.
    """
    def __init__(self, hardware_interface: FPLCSystemHardware, config: Optional[TestConfig] = None):
        """
        Initializes the SystemTestSuite.
        Args:
            hardware_interface (FPLCSystemHardware): An object implementing the hardware interface.
            config (Optional[TestConfig]): Configuration for the tests. If None, default is used.
        """
        if not isinstance(hardware_interface, FPLCSystemHardware):
            raise TypeError("hardware_interface must be an instance of FPLCSystemHardware.")
        self.hardware = hardware_interface
        self.config = config if config else TestConfig()
        
        self.tests: Dict[str, Callable[[], TestResult]] = {
            'pressure_stability': self.test_pressure_stability,
            'flow_accuracy': self.test_flow_accuracy,
            'detector_baseline': self.test_detector_baseline
        }
        logger.info("SystemTestSuite initialized.")

    def _collect_data(self, data_source_func: Callable[[], float], duration_sec: int, sample_rate_hz: int) -> List[float]:
        """Helper method to collect data from a hardware source over time."""
        num_samples = int(duration_sec * sample_rate_hz)
        if num_samples <= 0:
            logger.warning(f"Insufficient duration/sample rate for data collection (duration={duration_sec}, rate={sample_rate_hz}). Returning empty list.")
            return []
        
        data_points = []
        interval = 1.0 / sample_rate_hz # seconds per sample
        logger.debug(f"Collecting {num_samples} samples over {duration_sec}s at {sample_rate_hz}Hz...")
        for i in range(num_samples):
            try:
                data_points.append(data_source_func())
            except Exception as e:
                logger.error(f"Error collecting data point {i+1}: {e}")
                raise HardwareInteractionError(f"Failed to collect data: {e}")
            time.sleep(interval)
        logger.debug(f"Collected {len(data_points)} data points.")
        return data_points

    def _perform_warm_up(self):
        """Performs a system warm-up period to stabilize conditions."""
        logger.info(f"Performing system warm-up for {self.config.warm_up_time_sec} seconds...")
        # In a real system, this might involve turning on components, circulating solvent, etc.
        try:
            self.hardware.set_flow_rate(self.config.flow_test_target_flow_ml_min / 2) # Run at half flow for warm-up
            time.sleep(self.config.warm_up_time_sec)
            self.hardware.set_flow_rate(0.0) # Reset flow after warm-up
        except Exception as e:
            logger.error(f"Error during warm-up: {e}")
            raise TestSetupError(f"Failed during warm-up phase: {e}")
        logger.info("Warm-up complete.")

    def test_pressure_stability(self) -> TestResult:
        """
        Tests the stability of the system's pressure.
        Measures pressure over a period and checks deviation from the mean.
        """
        test_name = "pressure_stability"
        start_time = datetime.now()
        logger.info(f"Starting '{test_name}' test...")
        
        try:
            # Setup: Set a baseline flow for pressure generation (e.g., target flow)
            self.hardware.set_flow_rate(self.config.flow_test_target_flow_ml_min)
            time.sleep(5) # Give some time for pressure to stabilize at new flow
            
            pressure_readings = self._collect_data(
                self.hardware.get_pressure,
                self.config.pressure_test_duration_sec,
                self.config.pressure_test_sample_rate_hz
            )
            
            if not pressure_readings:
                raise TestSetupError("No pressure data collected for analysis.")

            mean_pressure = np.mean(pressure_readings)
            max_deviation = np.max(np.abs(np.array(pressure_readings) - mean_pressure))
            
            status = 'PASS' if max_deviation < self.config.pressure_stability_threshold_mpa else 'FAIL'
            message = f"Max deviation {max_deviation:.3f} MPa {'<' if status == 'PASS' else '>='} threshold {self.config.pressure_stability_threshold_mpa:.3f} MPa."
            
            self.hardware.stop_pump() # Clean up

            return TestResult(
                test_name=test_name,
                status=status,
                message=message,
                start_time=start_time,
                end_time=datetime.now(),
                details={
                    'mean_pressure_mpa': mean_pressure,
                    'max_deviation_mpa': max_deviation,
                    'threshold_mpa': self.config.pressure_stability_threshold_mpa,
                    'readings_count': len(pressure_readings),
                    'raw_data_sample': pressure_readings[:5] # Store a small sample
                }
            )
        except (HardwareInteractionError, TestSetupError, ValueError) as e:
            logger.error(f"Error during '{test_name}' test: {e}")
            return TestResult(
                test_name=test_name,
                status='ERROR',
                message=f"Test encountered an error: {e}",
                start_time=start_time,
                end_time=datetime.now()
            )
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred during '{test_name}': {e}", exc_info=True)
            return TestResult(
                test_name=test_name,
                status='ERROR',
                message=f"An unexpected error occurred: {e}",
                start_time=start_time,
                end_time=datetime.now()
            )

    def test_flow_accuracy(self) -> TestResult:
        """
        Tests the accuracy of the system's flow rate delivery.
        Sets a target flow and compares it to measured flow.
        """
        test_name = "flow_accuracy"
        start_time = datetime.now()
        logger.info(f"Starting '{test_name}' test...")

        try:
            self.hardware.set_flow_rate(self.config.flow_test_target_flow_ml_min)
            time.sleep(5) # Give time for flow to stabilize

            flow_readings = self._collect_data(
                self.hardware.get_flow_rate,
                self.config.flow_test_duration_sec,
                self.config.flow_test_sample_rate_hz
            )
            
            if not flow_readings:
                raise TestSetupError("No flow data collected for analysis.")

            actual_mean_flow = np.mean(flow_readings)
            flow_difference = abs(actual_mean_flow - self.config.flow_test_target_flow_ml_min)
            
            status = 'PASS' if flow_difference < self.config.flow_accuracy_tolerance_ml_min else 'FAIL'
            message = f"Flow difference {flow_difference:.3f} mL/min {'<' if status == 'PASS' else '>='} tolerance {self.config.flow_accuracy_tolerance_ml_min:.3f} mL/min."

            self.hardware.stop_pump() # Clean up

            return TestResult(
                test_name=test_name,
                status=status,
                message=message,
                start_time=start_time,
                end_time=datetime.now(),
                details={
                    'actual_mean_flow_ml_min': actual_mean_flow,
                    'target_flow_ml_min': self.config.flow_test_target_flow_ml_min,
                    'difference_ml_min': flow_difference,
                    'tolerance_ml_min': self.config.flow_accuracy_tolerance_ml_min,
                    'readings_count': len(flow_readings),
                    'raw_data_sample': flow_readings[:5]
                }
            )
        except (HardwareInteractionError, TestSetupError, ValueError) as e:
            logger.error(f"Error during '{test_name}' test: {e}")
            return TestResult(
                test_name=test_name,
                status='ERROR',
                message=f"Test encountered an error: {e}",
                start_time=start_time,
                end_time=datetime.now()
            )
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred during '{test_name}': {e}", exc_info=True)
            return TestResult(
                test_name=test_name,
                status='ERROR',
                message=f"An unexpected error occurred: {e}",
                start_time=start_time,
                end_time=datetime.now()
            )

    def test_detector_baseline(self) -> TestResult:
        """
        Tests the stability and noise of the detector baseline.
        Measures detector signal with no flow and checks for drift and noise.
        """
        test_name = "detector_baseline"
        start_time = datetime.now()
        logger.info(f"Starting '{test_name}' test...")

        try:
            self.hardware.stop_pump() # Ensure no flow
            self.hardware.zero_detector() # Perform auto-zero
            time.sleep(2) # Allow detector to stabilize after zeroing

            baseline_readings = self._collect_data(
                self.hardware.get_detector_signal,
                self.config.detector_baseline_test_duration_sec,
                self.config.detector_baseline_test_sample_rate_hz
            )
            
            if not baseline_readings:
                raise TestSetupError("No detector baseline data collected for analysis.")

            drift = np.max(baseline_readings) - np.min(baseline_readings)
            
            # Calculate RMS noise (Root Mean Square)
            # Remove mean to calculate true noise
            noise_data = np.array(baseline_readings) - np.mean(baseline_readings)
            rms_noise = np.sqrt(np.mean(noise_data**2))

            status_drift = 'PASS' if drift < self.config.detector_baseline_drift_limit_au else 'FAIL'
            status_noise = 'PASS' if rms_noise < self.config.detector_baseline_noise_limit_au else 'FAIL'
            
            overall_status = 'PASS' if status_drift == 'PASS' and status_noise == 'PASS' else 'FAIL'
            
            message = (
                f"Drift: {drift:.5f} AU ({status_drift}). "
                f"Noise (RMS): {rms_noise:.5f} AU ({status_noise})."
            )

            return TestResult(
                test_name=test_name,
                status=overall_status,
                message=message,
                start_time=start_time,
                end_time=datetime.now(),
                details={
                    'baseline_drift_au': drift,
                    'drift_limit_au': self.config.detector_baseline_drift_limit_au,
                    'rms_noise_au': rms_noise,
                    'noise_limit_au': self.config.detector_baseline_noise_limit_au,
                    'readings_count': len(baseline_readings),
                    'raw_data_sample': baseline_readings[:5]
                }
            )
        except (HardwareInteractionError, TestSetupError, ValueError) as e:
            logger.error(f"Error during '{test_name}' test: {e}")
            return TestResult(
                test_name=test_name,
                status='ERROR',
                message=f"Test encountered an error: {e}",
                start_time=start_time,
                end_time=datetime.now()
            )
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred during '{test_name}': {e}", exc_info=True)
            return TestResult(
                test_name=test_name,
                status='ERROR',
                message=f"An unexpected error occurred: {e}",
                start_time=start_time,
                end_time=datetime.now()
            )

    def run_all_tests(self) -> Dict[str, TestResult]:
        """
        Executes all configured system tests and returns a dictionary of results.
        Includes a system warm-up phase before running tests.
        """
        overall_results: Dict[str, TestResult] = {}
        logger.info("--- Starting All System Tests ---")
        try:
            self._perform_warm_up()
        except TestSetupError as e:
            logger.error(f"System warm-up failed: {e}. Skipping all tests.")
            for test_name in self.tests.keys():
                overall_results[test_name] = TestResult(
                    test_name=test_name,
                    status='SKIP',
                    message=f"Warm-up failed: {e}",
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
            self.hardware.stop_pump() # Ensure pump is off
            return overall_results

        for test_name, test_func in self.tests.items():
            result = test_func()
            overall_results[test_name] = result
            logger.info(f"Test '{test_name}' finished: {result.status}. {result.message}")
            # Add a small delay between tests to allow system to settle
            time.sleep(2)
        
        self.hardware.stop_pump() # Ensure pump is off after all tests
        logger.info("--- All System Tests Completed ---")
        self._generate_summary_report(overall_results)
        return overall_results

    def _generate_summary_report(self, results: Dict[str, TestResult]):
        """Generates and logs a summary report of all test results."""
        logger.info("\n--- System Test Summary Report ---")
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.status == 'PASS')
        failed_tests = sum(1 for r in results.values() if r.status == 'FAIL')
        error_tests = sum(1 for r in results.values() if r.status == 'ERROR')
        skipped_tests = sum(1 for r in results.values() if r.status == 'SKIP')

        logger.info(f"Total Tests Run: {total_tests}")
        logger.info(f"Tests Passed: {passed_tests}")
        logger.info(f"Tests Failed: {failed_tests}")
        logger.info(f"Tests with Errors: {error_tests}")
        logger.info(f"Tests Skipped: {skipped_tests}")
        logger.info("-" * 40)

        for test_name, result in results.items():
            logger.info(f"[{result.status}] {test_name}: {result.message}")
            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, float):
                        logger.info(f"    - {key}: {value:.4f}")
                    else:
                        logger.info(f"    - {key}: {value}")
        logger.info("-" * 40)


# --- Example Usage ---
if __name__ == "__main__":
    from datetime import datetime # Import datetime here for consistent logging

    # 1. Initialize Hardware Interface (use Mock for testing without physical hardware)
    mock_hardware = MockHardwareInterface()

    # 2. Define Test Configuration (optional, defaults will be used if not provided)
    test_config = TestConfig(
        pressure_stability_threshold_mpa=0.03, # Tighter threshold
        flow_accuracy_tolerance_ml_min=0.015, # Tighter tolerance
        detector_baseline_drift_limit_au=0.0005,
        detector_baseline_noise_limit_au=0.00005,
        warm_up_time_sec=10 # Shorter warm-up for quick demo
    )

    # 3. Initialize the Test Suite
    try:
        suite = SystemTestSuite(hardware_interface=mock_hardware, config=test_config)

        # 4. Run All Tests
        results = suite.run_all_tests()

        # You can now programmatically check results if needed
        # if results['pressure_stability'].status == 'PASS':
        #     print("\nPressure stability test passed!")

    except TestSuiteError as e:
        logger.error(f"System Test Suite initialization or critical error: {e}")
    except Exception as e:
        logger.critical(f"An unhandled error occurred in the main execution block: {e}", exc_info=True)

    # Example of running a single test manually
    print("\n--- Running a Single Test Manually ---")
    try:
        # Re-initialize mock to ensure clean state if running independently
        mock_hardware_single = MockHardwareInterface()
        suite_single_test = SystemTestSuite(hardware_interface=mock_hardware_single, config=test_config)
        
        # Manually run detector baseline test
        detector_result = suite_single_test.test_detector_baseline()
        print(f"Manual Detector Baseline Test Result: {detector_result.status}")
        print(f"  Message: {detector_result.message}")
        print(f"  Details: {detector_result.details}")

    except TestSuiteError as e:
        logger.error(f"Single test execution error: {e}")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during single test execution: {e}", exc_info=True)