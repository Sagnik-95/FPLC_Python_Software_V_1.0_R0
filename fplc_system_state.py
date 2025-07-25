from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any
import time
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SystemState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    CALIBRATING = "Calibrating"
    WASHING = "Washing"
    EQUILIBRATING = "Equilibrating"

@dataclass(frozen=True) # Make SystemStatus immutable after creation
class SystemStatus:
    """
    Represents the current status of the FPLC system.
    Attributes are immutable to ensure state consistency.
    """
    state: SystemState
    pressure: float
    flow_rate: float
    gradient: float
    uv_280: float
    uv_254: float
    conductivity: float
    ph: float
    temperature: float
    last_update: float

class StateManager:
    """
    Manages the state and status of the FPLC system.
    Provides methods for updating the state with validation and maintains a history.
    Ensures thread-safe access to the system status.
    """
    def __init__(self, initial_status: Dict[str, Any] = None):
        """
        Initializes the StateManager with an optional initial status.
        :param initial_status: A dictionary containing initial values for SystemStatus.
                                If not provided, defaults will be used.
        """
        if initial_status is None:
            initial_status = {}

        # Define default values
        defaults = {
            "state": SystemState.IDLE,
            "pressure": 0.0,
            "flow_rate": 0.0,
            "gradient": 0.0,
            "uv_280": 0.0,
            "uv_254": 0.0,
            "conductivity": 0.0,
            "ph": 7.0,
            "temperature": 25.0,
            "last_update": time.time()
        }

        # Merge defaults with initial_status, prioritizing initial_status
        merged_status = {**defaults, **initial_status}

        try:
            self._status = SystemStatus(
                state=merged_status["state"],
                pressure=float(merged_status["pressure"]),
                flow_rate=float(merged_status["flow_rate"]),
                gradient=float(merged_status["gradient"]),
                uv_280=float(merged_status["uv_280"]),
                uv_254=float(merged_status["uv_254"]),
                conductivity=float(merged_status["conductivity"]),
                ph=float(merged_status["ph"]),
                temperature=float(merged_status["temperature"]),
                last_update=float(merged_status["last_update"])
            )
        except (KeyError, ValueError, TypeError) as e:
            logging.error(f"Failed to initialize SystemStatus due to invalid initial_status: {e}")
            raise ValueError(f"Invalid initial status provided: {e}")

        self._state_history = []
        self._state_history.append(self._status) # Record initial state
        self._lock = threading.RLock() # Reentrant lock for thread safety
        logging.info(f"StateManager initialized with state: {self._status.state.value}")

    def update_state(self, **kwargs) -> None:
        """
        Updates the FPLC system's status.
        Performs validation on input values and manages state transitions.
        :param kwargs: Keyword arguments representing the status attributes to update.
        """
        with self._lock:
            current_status_dict = asdict(self._status)
            new_status_data = current_status_dict.copy()

            # --- Input Validation and Type Coercion ---
            for key, value in kwargs.items():
                if not hasattr(self._status, key):
                    logging.warning(f"Attempted to update non-existent attribute: {key}")
                    continue

                # Basic type checking and value range validation
                try:
                    if key == "state":
                        if not isinstance(value, SystemState):
                            raise TypeError(f"State must be a SystemState enum, got {type(value)}")
                        if not self._is_valid_transition(self._status.state, value):
                            logging.warning(f"Invalid state transition from {self._status.state.value} to {value.value}")
                            raise ValueError(f"Invalid state transition: {self._status.state.value} -> {value.value}")
                    elif key in ["pressure", "flow_rate", "gradient", "uv_280", "uv_254", "conductivity", "ph", "temperature"]:
                        value = float(value)
                        if value < 0 and key not in ["gradient"]: # Gradient can be negative for specific applications
                            logging.warning(f"Attempted to set {key} to a negative value: {value}")
                            raise ValueError(f"{key} cannot be negative.")
                        # Add more specific range checks here if needed, e.g., pH between 0 and 14
                    
                    new_status_data[key] = value
                except (ValueError, TypeError) as e:
                    logging.error(f"Validation error for attribute '{key}': {e}. Update aborted for this attribute.")
                    return # Abort update if any attribute fails validation

            new_status_data["last_update"] = time.time()

            try:
                # Create a new immutable SystemStatus object
                new_status = SystemStatus(**new_status_data)
                self._status = new_status
                self._state_history.append(self._status)
                logging.info(f"System state updated to: {self._status.state.value}. New status: {self._status}")
            except Exception as e:
                logging.error(f"Failed to create new SystemStatus object: {e}")

    def _is_valid_transition(self, old_state: SystemState, new_state: SystemState) -> bool:
        """
        Defines valid state transitions.
        This can be expanded into a more complex state machine if necessary.
        """
        if old_state == new_state:
            return True # No change is always valid

        valid_transitions = {
            SystemState.IDLE: [SystemState.RUNNING, SystemState.CALIBRATING, SystemState.MAINTENANCE, SystemState.WASHING, SystemState.EQUILIBRATING, SystemState.ERROR],
            SystemState.RUNNING: [SystemState.PAUSED, SystemState.IDLE, SystemState.ERROR],
            SystemState.PAUSED: [SystemState.RUNNING, SystemState.IDLE, SystemState.ERROR],
            SystemState.ERROR: [SystemState.IDLE, SystemState.MAINTENANCE], # Must be reset or fixed
            SystemState.MAINTENANCE: [SystemState.IDLE],
            SystemState.CALIBRATING: [SystemState.IDLE, SystemState.ERROR],
            SystemState.WASHING: [SystemState.IDLE, SystemState.ERROR],
            SystemState.EQUILIBRATING: [SystemState.IDLE, SystemState.ERROR]
        }
        return new_state in valid_transitions.get(old_state, [])

    def get_state(self) -> SystemStatus:
        """
        Returns the current immutable SystemStatus of the FPLC system.
        """
        with self._lock:
            return self._status

    def get_state_history(self) -> list[SystemStatus]:
        """
        Returns a copy of the state history for analysis.
        """
        with self._lock:
            return list(self._state_history) # Return a copy to prevent external modification

# Example Usage:
if __name__ == "__main__":
    # Initialize with some custom values
    state_manager = StateManager(initial_status={"ph": 7.2, "temperature": 22.5})

    print(f"Current State: {state_manager.get_state()}")

    # Valid state updates
    state_manager.update_state(state=SystemState.RUNNING, flow_rate=1.5, pressure=10.2)
    state_manager.update_state(uv_280=0.5, uv_254=0.1)
    state_manager.update_state(state=SystemState.PAUSED)

    # Invalid state transition
    state_manager.update_state(state=SystemState.CALIBRATING) # Should warn/error as not allowed from PAUSED

    # Invalid value
    state_manager.update_state(pressure=-5.0) # Should warn/error

    # Attempt to update non-existent attribute
    state_manager.update_state(non_existent_attribute="test")

    state_manager.update_state(state=SystemState.IDLE)
    state_manager.update_state(state=SystemState.MAINTENANCE)
    
    print("\n--- State History ---")
    for i, status in enumerate(state_manager.get_state_history()):
        print(f"[{i}] {status}")

    # Demonstrate thread safety (simplified example)
    def update_flow():
        for _ in range(3):
            time.sleep(0.1)
            flow = state_manager.get_state().flow_rate + 0.1
            state_manager.update_state(flow_rate=flow)
            logging.info(f"Thread updated flow to {flow:.2f}")

    def read_state():
        for _ in range(5):
            time.sleep(0.05)
            status = state_manager.get_state()
            logging.info(f"Thread read state: {status.state.value}, Pressure: {status.pressure:.2f}")

    # t1 = threading.Thread(target=update_flow)
    # t2 = threading.Thread(target=read_state)

    # t1.start()
    # t2.start()

    # t1.join()
    # t2.join()
    # print("\nThreaded operations complete.")

    # Show final state
    print(f"\nFinal State: {state_manager.get_state()}")