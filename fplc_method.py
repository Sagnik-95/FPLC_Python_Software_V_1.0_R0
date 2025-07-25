import json
from datetime import datetime
import sys
import logging
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
from fplc_method_validator import MethodStep, MethodValidator

# --- Custom Exceptions ---
class FPLCError(Exception):
    """Base exception for FPLC related errors."""
    pass

class InvalidParameterError(FPLCError):
    """Exception raised for invalid input parameters."""
    pass

class MethodValidationError(FPLCError):
    """Exception raised when an FPLC method fails validation."""
    pass

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fplc_log.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('FPLC_App')

# --- FPLC Core Logic Classes ---
class FPLCMethod:
    """
    Represents a complete FPLC method with all parameters and steps.

    This class encapsulates the details required to define and manage an FPLC
    chromatography run, including metadata, column information, buffer
    compositions, and a sequence of operational steps.
    """
    def __init__(self, name: str = "", description: str = "", author: str = ""):
        """
        Initializes a new FPLCMethod instance.

        Args:
            name (str): The name of the FPLC method.
            description (str): A brief description of the method's purpose.
            author (str): The creator of the method.
        """
        if not isinstance(name, str) or not name:
            raise InvalidParameterError("Method 'name' cannot be empty.")
        if not isinstance(description, str):
            raise InvalidParameterError("Method 'description' must be a string.")
        if not isinstance(author, str) or not author:
            raise InvalidParameterError("Method 'author' cannot be empty.")

        self.name: str = name
        self.description: str = description
        self.author: str = author
        self.created_date: datetime = datetime.now()
        self.modified_date: datetime = datetime.now()
        self.column_type: str = ""
        self.column_volume: float = 0.0  # mL
        self.max_pressure: float = 0.0   # MPa
        self.steps: list = []
        self.sample_info: dict = {}
        self.buffer_info: dict = {
            'A': {'name': 'Buffer A', 'composition': 'Deionized Water'},
            'B': {'name': 'Buffer B', 'composition': '1M NaCl'}
        }
        logger.info(f"FPLCMethod '{self.name}' initialized by '{self.author}'.")

    def set_column_parameters(self, column_type: str, column_volume: float, max_pressure: float):
        """
        Sets the parameters for the chromatography column.

        Args:
            column_type (str): The type or name of the chromatography column.
            column_volume (float): The volume of the column in mL. Must be positive.
            max_pressure (float): The maximum allowed pressure for the column in MPa. Must be positive.

        Raises:
            InvalidParameterError: If any parameter is invalid.
        """
        if not isinstance(column_type, str) or not column_type:
            raise InvalidParameterError("Column 'column_type' cannot be empty.")
        if not isinstance(column_volume, (int, float)) or column_volume <= 0:
            raise InvalidParameterError("Column 'column_volume' must be a positive number.")
        if not isinstance(max_pressure, (int, float)) or max_pressure <= 0:
            raise InvalidParameterError("Column 'max_pressure' must be a positive number.")

        self.column_type = column_type
        self.column_volume = float(column_volume)
        self.max_pressure = float(max_pressure)
        self.modified_date = datetime.now()
        logger.info(f"Column parameters set for '{self.name}': Type='{column_type}', Volume={column_volume} mL, Max Pressure={max_pressure} MPa.")

    def set_buffer_info(self, buffer_name: str, name: str, composition: str):
        """
        Sets the name and composition for a specific buffer (A or B).

        Args:
            buffer_name (str): The identifier for the buffer ('A' or 'B').
            name (str): The common name of the buffer (e.g., "Binding Buffer").
            composition (str): The chemical composition of the buffer.

        Raises:
            InvalidParameterError: If buffer_name is not 'A' or 'B', or if name/composition are empty.
        """
        if buffer_name not in ['A', 'B']:
            raise InvalidParameterError("Buffer 'buffer_name' must be 'A' or 'B'.")
        if not isinstance(name, str) or not name:
            raise InvalidParameterError(f"Buffer '{buffer_name}' 'name' cannot be empty.")
        if not isinstance(composition, str) or not composition:
            raise InvalidParameterError(f"Buffer '{buffer_name}' 'composition' cannot be empty.")

        self.buffer_info[buffer_name]['name'] = name
        self.buffer_info[buffer_name]['composition'] = composition
        self.modified_date = datetime.now()
        logger.info(f"Buffer '{buffer_name}' info updated: Name='{name}', Composition='{composition}'.")

    def set_sample_info(self, sample_name: str, concentration: float, volume: float, notes: str = ""):
        """
        Sets information about the sample to be run.

        Args:
            sample_name (str): The name or identifier of the sample.
            concentration (float): The concentration of the sample. Must be positive.
            volume (float): The volume of the sample in mL. Must be positive.
            notes (str): Any additional notes about the sample.

        Raises:
            InvalidParameterError: If any parameter is invalid.
        """
        if not isinstance(sample_name, str) or not sample_name:
            raise InvalidParameterError("Sample 'sample_name' cannot be empty.")
        if not isinstance(concentration, (int, float)) or concentration <= 0:
            raise InvalidParameterError("Sample 'concentration' must be a positive number.")
        if not isinstance(volume, (int, float)) or volume <= 0:
            raise InvalidParameterError("Sample 'volume' must be a positive number.")
        if not isinstance(notes, str):
            raise InvalidParameterError("Sample 'notes' must be a string.")

        self.sample_info = {
            'name': sample_name,
            'concentration': float(concentration),
            'volume': float(volume),
            'notes': notes
        }
        self.modified_date = datetime.now()
        logger.info(f"Sample info set for method '{self.name}': Sample='{sample_name}', Conc.={concentration}, Vol.={volume}.")

    def add_step(self, step_type: str, duration: float, flow_rate: float, buffer_a: float, buffer_b: float,
                 collection: bool = False, collection_volume: float = 0.0, notes: str = "",
                 gradient_start_b: float = None, gradient_end_b: float = None):
        """
        Adds a new operational step to the FPLC method.

        Args:
            step_type (str): The type of step (e.g., "Load", "Wash", "Elution", "Gradient", "Clean").
            duration (float): The duration of the step in minutes. Must be non-negative.
            flow_rate (float): The flow rate during the step in mL/min. Must be positive.
            buffer_a (float): Percentage of Buffer A (0-100).
            buffer_b (float): Percentage of Buffer B (0-100).
            collection (bool): Whether fractions are collected during this step.
            collection_volume (float): If `collection` is True, the volume per fraction in mL. Must be positive if collection is True.
            notes (str): Any specific notes for this step.
            gradient_start_b (float, optional): For "Gradient" steps, the starting percentage of Buffer B.
            gradient_end_b (float, optional): For "Gradient" steps, the ending percentage of Buffer B.

        Raises:
            InvalidParameterError: If any parameter is invalid or inconsistent with step_type.
        """
        if not isinstance(step_type, str) or not step_type:
            raise InvalidParameterError("Step 'step_type' cannot be empty.")
        if step_type not in ["Load", "Wash", "Elution", "Gradient", "Clean", "Equilibrate"]:
            logger.warning(f"Uncommon step type '{step_type}' used. Consider using standard types.")

        if not isinstance(duration, (int, float)) or duration < 0:
            raise InvalidParameterError("Step 'duration' must be a non-negative number.")
        if not isinstance(flow_rate, (int, float)) or flow_rate <= 0:
            raise InvalidParameterError("Step 'flow_rate' must be a positive number.")
        if not isinstance(buffer_a, (int, float)) or not (0 <= buffer_a <= 100):
            raise InvalidParameterError("Step 'buffer_a' percentage must be between 0 and 100.")
        if not isinstance(buffer_b, (int, float)) or not (0 <= buffer_b <= 100):
            raise InvalidParameterError("Step 'buffer_b' percentage must be between 0 and 100.")
        if abs(buffer_a + buffer_b - 100) > 0.01: # Allow for minor floating point inaccuracies
            raise InvalidParameterError("Sum of 'buffer_a' and 'buffer_b' percentages must be 100.")

        if collection and (not isinstance(collection_volume, (int, float)) or collection_volume <= 0):
            raise InvalidParameterError("If 'collection' is True, 'collection_volume' must be a positive number.")
        if not isinstance(notes, str):
            raise InvalidParameterError("Step 'notes' must be a string.")

        if step_type == "Gradient":
            if gradient_start_b is None or not (0 <= gradient_start_b <= 100):
                raise InvalidParameterError("For 'Gradient' step, 'gradient_start_b' must be specified and between 0 and 100.")
            if gradient_end_b is None or not (0 <= gradient_end_b <= 100):
                raise InvalidParameterError("For 'Gradient' step, 'gradient_end_b' must be specified and between 0 and 100.")
            if not isinstance(gradient_start_b, (int, float)) or not isinstance(gradient_end_b, (int, float)):
                raise InvalidParameterError("Gradient start/end percentages must be numbers.")
            if buffer_b != gradient_start_b: # For consistency, buffer_b should be start_b for gradient
                 logger.warning(f"For 'Gradient' step, 'buffer_b' ({buffer_b}%) should typically match 'gradient_start_b' ({gradient_start_b}%). Auto-correcting buffer_b.")
                 buffer_b = gradient_start_b
                 buffer_a = 100 - buffer_b
        elif gradient_start_b is not None or gradient_end_b is not None:
            logger.warning(f"Gradient parameters provided for non-gradient step '{step_type}'. They will be ignored.")
            gradient_start_b = None
            gradient_end_b = None


        step = {
            'type': step_type,
            'duration': float(duration),
            'flow_rate': float(flow_rate),
            'buffer_a': float(buffer_a),
            'buffer_b': float(buffer_b),
            'collection': collection,
            'collection_volume': float(collection_volume) if collection else 0.0,
            'notes': notes,
            'gradient_start_b': float(gradient_start_b) if gradient_start_b is not None else None,
            'gradient_end_b': float(gradient_end_b) if gradient_end_b is not None else None,
        }
        self.steps.append(step)
        self.modified_date = datetime.now()
        logger.info(f"Added step '{step_type}' to method '{self.name}'. Total steps: {len(self.steps)}.")

    def get_step(self, index: int) -> dict:
        """Retrieves a step by its index."""
        if not isinstance(index, int) or not (0 <= index < len(self.steps)):
            raise IndexError("Step index out of bounds.")
        return self.steps[index]

    def remove_step(self, index: int):
        """Removes a step by its index."""
        if not isinstance(index, int) or not (0 <= index < len(self.steps)):
            raise IndexError("Step index out of bounds.")
        removed_step_type = self.steps[index]['type']
        del self.steps[index]
        self.modified_date = datetime.now()
        logger.info(f"Removed step at index {index} (Type: '{removed_step_type}') from method '{self.name}'.")

    def validate_method(self) -> bool:
        """
        Performs a comprehensive validation of the FPLC method.

        Checks for completeness and consistency of method parameters and steps.

        Returns:
            bool: True if the method is valid, False otherwise.

        Raises:
            MethodValidationError: If any validation rule is violated.
        """
        if not self.name:
            raise MethodValidationError("Method name is not set.")
        if not self.author:
            raise MethodValidationError("Method author is not set.")
        if not self.column_type or self.column_volume <= 0 or self.max_pressure <= 0:
            raise MethodValidationError("Column parameters are incomplete or invalid.")
        if not self.buffer_info['A']['name'] or not self.buffer_info['B']['name']:
            raise MethodValidationError("Buffer information is incomplete.")
        if not self.steps:
            logger.warning(f"Method '{self.name}' has no defined steps.")
            # raise MethodValidationError("Method has no defined steps.") # Could be an error or just a warning depending on strictness

        for i, step in enumerate(self.steps):
            if step['type'] == "Gradient":
                if step['gradient_start_b'] is None or step['gradient_end_b'] is None:
                    raise MethodValidationError(f"Gradient step {i} is missing start/end buffer B percentages.")
                if not (0 <= step['gradient_start_b'] <= 100 and 0 <= step['gradient_end_b'] <= 100):
                     raise MethodValidationError(f"Gradient percentages for step {i} are out of valid range (0-100).")
            if step['collection'] and step['collection_volume'] <= 0:
                raise MethodValidationError(f"Collection step {i} has invalid collection volume.")
            if not (0 <= step['buffer_a'] <= 100 and 0 <= step['buffer_b'] <= 100):
                raise MethodValidationError(f"Buffer percentages for step {i} are out of valid range (0-100).")
            if abs(step['buffer_a'] + step['buffer_b'] - 100) > 0.01:
                raise MethodValidationError(f"Buffer percentages for step {i} do not sum to 100%.")

        logger.info(f"Method '{self.name}' validated successfully.")
        return True

    def generate_summary(self) -> str:
        """
        Generates a human-readable summary of the FPLC method.

        Returns:
            str: A formatted string containing the method summary.
        """
        summary = [
            f"--- FPLC Method Summary: {self.name} ---",
            f"Description: {self.description}",
            f"Author: {self.author}",
            f"Created: {self.created_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Last Modified: {self.modified_date.strftime('%Y-%m-%d %H:%M:%S')}",
            "\n--- Column Information ---",
            f"Column Type: {self.column_type}",
            f"Column Volume: {self.column_volume:.2f} mL",
            f"Max Pressure: {self.max_pressure:.2f} MPa",
            "\n--- Buffer Information ---",
            f"Buffer A Name: {self.buffer_info['A']['name']}",
            f"Buffer A Composition: {self.buffer_info['A']['composition']}",
            f"Buffer B Name: {self.buffer_info['B']['name']}",
            f"Buffer B Composition: {self.buffer_info['B']['composition']}",
        ]

        if self.sample_info:
            summary.append("\n--- Sample Information ---")
            summary.append(f"Sample Name: {self.sample_info.get('name', 'N/A')}")
            summary.append(f"Concentration: {self.sample_info.get('concentration', 'N/A')} mg/mL")
            summary.append(f"Volume: {self.sample_info.get('volume', 'N/A')} mL")
            if self.sample_info.get('notes'):
                summary.append(f"Sample Notes: {self.sample_info.get('notes')}")

        summary.append("\n--- Method Steps ---")
        if not self.steps:
            summary.append("No steps defined for this method.")
        else:
            for i, step in enumerate(self.steps):
                step_str = f"  Step {i+1} - Type: {step['type']}, Duration: {step['duration']:.2f} min, Flow Rate: {step['flow_rate']:.2f} mL/min"
                if step['type'] == "Gradient":
                    step_str += f", Buffer B Gradient: {step['gradient_start_b']:.1f}% to {step['gradient_end_b']:.1f}%"
                else:
                    step_str += f", Buffer A: {step['buffer_a']:.1f}%, Buffer B: {step['buffer_b']:.1f}%"
                if step['collection']:
                    step_str += f", Collect Fractions (Vol: {step['collection_volume']:.2f} mL)"
                if step['notes']:
                    step_str += f" (Notes: {step['notes']})"
                summary.append(step_str)

        return "\n".join(summary)

    def to_json(self) -> str:
        """
        Converts the FPLCMethod object to a JSON string.

        Returns:
            str: A JSON formatted string representing the FPLC method.

        Raises:
            FPLCError: If there's an issue during JSON serialization.
        """
        try:
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
        except TypeError as e:
            logger.error(f"Error serializing FPLC method '{self.name}' to JSON: {e}")
            raise FPLCError(f"Failed to serialize method to JSON: {e}") from e

    @classmethod
    def from_json(cls, json_str: str):
        """
        Creates an FPLCMethod object from a JSON string.

        Args:
            json_str (str): A JSON formatted string representing an FPLC method.

        Returns:
            FPLCMethod: An instance of FPLCMethod populated with data from the JSON.

        Raises:
            FPLCError: If the JSON string is invalid or data is missing.
            InvalidParameterError: If critical parameters loaded from JSON are invalid.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON string: {e}")
            raise FPLCError(f"Invalid JSON string provided: {e}") from e

        # Validate essential fields before creating the object
        required_fields = ['name', 'author']
        for field in required_fields:
            if field not in data or not data[field]:
                raise InvalidParameterError(f"Missing or empty required field in JSON: '{field}'")

        try:
            method = cls(name=data.get('name', ''), description=data.get('description', ''), author=data.get('author', ''))

            method.created_date = datetime.fromisoformat(data.get('created_date', datetime.now().isoformat()))
            method.modified_date = datetime.fromisoformat(data.get('modified_date', datetime.now().isoformat()))

            method.set_column_parameters(
                data.get('column_type', ''),
                data.get('column_volume', 0.0),
                data.get('max_pressure', 0.0)
            )

            # Safely load buffer info, providing defaults if entirely missing or malformed
            loaded_buffer_info = data.get('buffer_info', {})
            method.buffer_info['A'].update(loaded_buffer_info.get('A', {}))
            method.buffer_info['B'].update(loaded_buffer_info.get('B', {}))
            # Ensure default names/compositions if not provided in JSON
            if not method.buffer_info['A'].get('name'): method.buffer_info['A']['name'] = 'Buffer A'
            if not method.buffer_info['A'].get('composition'): method.buffer_info['A']['composition'] = 'Deionized Water'
            if not method.buffer_info['B'].get('name'): method.buffer_info['B']['name'] = 'Buffer B'
            if not method.buffer_info['B'].get('composition'): method.buffer_info['B']['composition'] = '1M NaCl'


            method.sample_info = data.get('sample_info', {}) # Already validated in set_sample_info

            # Load steps, ensuring each step is validated as it's added
            loaded_steps = data.get('steps', [])
            for step_data in loaded_steps:
                try:
                    method.add_step(
                        step_type=step_data.get('type', ''),
                        duration=step_data.get('duration', 0.0),
                        flow_rate=step_data.get('flow_rate', 0.0),
                        buffer_a=step_data.get('buffer_a', 0.0),
                        buffer_b=step_data.get('buffer_b', 0.0),
                        collection=step_data.get('collection', False),
                        collection_volume=step_data.get('collection_volume', 0.0),
                        notes=step_data.get('notes', ''),
                        gradient_start_b=step_data.get('gradient_start_b'),
                        gradient_end_b=step_data.get('gradient_end_b')
                    )
                except InvalidParameterError as e:
                    logger.warning(f"Skipping invalid step during JSON load for method '{method.name}': {e} - Step data: {step_data}")
                    # Decide whether to raise here or just log and continue. Raising is stricter.
                    # raise FPLCError(f"Invalid step encountered during JSON loading: {e}") from e

            logger.info(f"Loaded method '{method.name}' from JSON successfully.")
            method.validate_method() # Final validation after loading everything
            return method
        except (KeyError, TypeError, ValueError, InvalidParameterError) as e:
            logger.error(f"Error parsing FPLC method data from JSON for '{data.get('name', 'Unknown')}': {e}")
            raise FPLCError(f"Incomplete or malformed data in JSON for FPLC method: {e}") from e

# --- Example Usage (for testing and demonstration) ---
if __name__ == "__main__":
    logger.info("Starting FPLC Method Application Example.")

    try:
        # 1. Create a new FPLC Method
        my_method = FPLCMethod(name="Protein_Purification_SEC", description="Size Exclusion Chromatography for protein purification.", author="Dr. Smith")
        my_method.set_column_parameters(column_type="Superdex 200 Increase 10/300 GL", column_volume=24.0, max_pressure=5.0)
        my_method.set_buffer_info('A', "PBS Buffer", "1x PBS, pH 7.4")
        my_method.set_buffer_info('B', "High Salt Buffer", "1x PBS, 0.5M NaCl, pH 7.4")
        my_method.set_sample_info("My_Protein_Sample", concentration=10.5, volume=0.5, notes="E. coli lysate, clarified")

        # 2. Add steps to the method
        my_method.add_step("Equilibrate", duration=2.0, flow_rate=0.5, buffer_a=100.0, buffer_b=0.0)
        my_method.add_step("Load", duration=my_method.sample_info['volume']/0.5, flow_rate=0.5, buffer_a=100.0, buffer_b=0.0, notes="Load sample onto column")
        my_method.add_step("Wash", duration=5.0, flow_rate=0.5, buffer_a=100.0, buffer_b=0.0, notes="Wash unbound material")
        my_method.add_step("Elution", duration=30.0, flow_rate=0.5, buffer_a=100.0, buffer_b=0.0, collection=True, collection_volume=0.5, notes="Isocratic elution with PBS")
        my_method.add_step("Clean", duration=10.0, flow_rate=1.0, buffer_a=0.0, buffer_b=100.0, notes="High salt wash for cleaning")
        my_method.add_step("Equilibrate", duration=5.0, flow_rate=0.5, buffer_a=100.0, buffer_b=0.0, notes="Re-equilibrate for next run")

        # Example of a gradient step
        my_method.add_step("Gradient", duration=20.0, flow_rate=0.8, buffer_a=80.0, buffer_b=20.0,
                           gradient_start_b=20.0, gradient_end_b=80.0, collection=True, collection_volume=1.0, notes="Ion exchange gradient elution")

        # 3. Validate the method
        if my_method.validate_method():
            logger.info(f"Method '{my_method.name}' is valid.")

        # 4. Generate and print summary
        print(my_method.generate_summary())

        # 5. Convert to JSON
        json_output = my_method.to_json()
        print("\n--- JSON Representation ---")
        print(json_output)

        # 6. Save to file
        with open("protein_purification_sec.json", "w") as f:
            f.write(json_output)
        logger.info("Method saved to protein_purification_sec.json")

        # 7. Load from JSON
        with open("protein_purification_sec.json", "r") as f:
            loaded_json_str = f.read()

        loaded_method = FPLCMethod.from_json(loaded_json_str)
        print("\n--- Loaded Method Summary ---")
        print(loaded_method.generate_summary())
        logger.info(f"Successfully loaded and verified method '{loaded_method.name}' from file.")

        # --- Demonstration of Error Handling and Validation ---
        print("\n--- Demonstrating Error Handling ---")

        # Invalid method name
        try:
            FPLCMethod(name="", description="Test", author="User")
        except InvalidParameterError as e:
            logger.error(f"Caught expected error: {e}")

        # Invalid column volume
        try:
            invalid_method = FPLCMethod("Invalid Column", "Test", "User")
            invalid_method.set_column_parameters("C18", -10.0, 1.0)
        except InvalidParameterError as e:
            logger.error(f"Caught expected error: {e}")

        # Invalid step parameters (buffer percentages)
        try:
            error_method = FPLCMethod("Error Method", "Test", "User")
            error_method.set_column_parameters("TestCol", 1.0, 1.0)
            error_method.add_step("Wash", 5.0, 0.5, 110.0, 0.0) # Invalid buffer_a
        except InvalidParameterError as e:
            logger.error(f"Caught expected error: {e}")

        # Invalid gradient step
        try:
            error_method_grad = FPLCMethod("Error Gradient Method", "Test", "User")
            error_method_grad.set_column_parameters("TestCol", 1.0, 1.0)
            error_method_grad.add_step("Gradient", 10.0, 0.5, 90.0, 10.0, gradient_start_b=5.0, gradient_end_b=105.0) # Invalid end gradient
        except InvalidParameterError as e:
            logger.error(f"Caught expected error: {e}")

        # Method validation failure
        try:
            incomplete_method = FPLCMethod("Incomplete Method", "Demonstrates validation", "Validator")
            # Missing column parameters, steps, etc.
            incomplete_method.validate_method()
        except MethodValidationError as e:
            logger.error(f"Caught expected method validation error: {e}")

        # Attempt to load malformed JSON
        malformed_json = '{"name": "Broken Method", "description": "This is not valid JSON", "author": "Bot",'
        try:
            FPLCMethod.from_json(malformed_json)
        except FPLCError as e:
            logger.error(f"Caught expected JSON decoding error: {e}")

        # Attempt to load JSON with missing required fields
        incomplete_json = '{"name": "Another Broken Method", "description": "Missing author"}'
        try:
            FPLCMethod.from_json(incomplete_json)
        except InvalidParameterError as e:
            logger.error(f"Caught expected JSON loading error (missing field): {e}")


    except Exception as e:
        logger.critical(f"An unhandled error occurred during example execution: {e}", exc_info=True)

    logger.info("FPLC Method Application Example Finished.")

class MethodEditor(QGroupBox):
    """Enhanced method editor with advanced chromatography controls."""
    def __init__(self, parent=None):
        super().__init__("Method Editor", parent)
        self.method = FPLCMethod(name="New Method", author="Default Author")
        self.logger = logging.getLogger('MethodEditor')
        self.setup_ui()
        self.logger.info("Method Editor UI initialized.")

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
        self.buffer_b_name.textChanged.connect(self.update_method_metadata) # FIX: Added missing connection
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
        self.logger.info(f"Added new step '{step_type}' to method editor.")

    def _update_step_data(self, row, param_name, value):
        """Updates a specific parameter for a step in the method object."""
        if 0 <= row < len(self.method.steps):
            self.method.steps[row][param_name] = value
            self.method.modified_date = datetime.now()
            self.logger.debug(f"Updated step {row}, param '{param_name}' to '{value}'.")
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
            self.logger.warning(f"Attempted to update non-existent step at row {row}.")

    def delete_step(self, row):
        """Deletes a step from both table and method object."""
        if 0 <= row < self.steps_table.rowCount():
            step_type = self.steps_table.item(row, 0).text() if self.steps_table.item(row, 0) else "Unknown"
            self.steps_table.removeRow(row)
            if 0 <= row < len(self.method.steps):
                self.method.steps.pop(row)
                self.method.modified_date = datetime.now()
                self.logger.info(f"Deleted step '{step_type}' at row {row}.")
            else:
                self.logger.warning(f"Attempted to delete step from method object at non-existent index {row}.")

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
            self.logger.warning(f"Attempted to delete non-existent row {row} from steps table.")

    def load_method_template(self, template_name):
        template_data = self.method_template_manager.templates.get(template_name)
        if template_data:
            # Convert dictionary steps to MethodStep objects
            method_steps_for_validation = []
            for step_dict in template_data:
                # Ensure all required keys are present or handle missing keys
                # 'step_type', 'duration', 'flow_rate', 'gradient', 'collection'
                method_steps_for_validation.append(MethodStep(
                    step_type=step_dict.get('type', 'step'), # 'type' from template_data
                    duration=step_dict.get('duration', 0.0),
                    flow_rate=step_dict.get('flow_rate', 0.0),
                    gradient=step_dict.get('gradient', 0.0),
                    collection=step_dict.get('collection', False)
                ))
            return method_steps_for_validation
        return []

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
        self.logger.debug("Method metadata updated from UI.")

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
        self.logger.info(f"Method '{method.name}' loaded into UI.")