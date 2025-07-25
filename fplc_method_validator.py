import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Management ---
@dataclass
class SystemConfig:
    max_pressure_psi: float = 3000.0  # Example max pressure in PSI
    max_flow_rate_ml_min: float = 10.0 # Example max flow rate in mL/min
    max_method_duration_minutes: float = 1440.0 # 24 hours in minutes

    @classmethod
    def from_json(cls, config_path: str):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except FileNotFoundError:
            logging.warning(f"Configuration file not found at {config_path}. Using default system settings.")
            return cls()
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {config_path}. Using default system settings.")
            return cls()
        except TypeError as e:
            logging.error(f"Error in configuration data types from {config_path}: {e}. Using default system settings.")
            return cls()

# --- Custom Exceptions ---
class MethodError(Exception):
    """Base exception for method-related errors."""
    pass

class InvalidMethodStepError(MethodError):
    """Exception raised for invalid individual method steps."""
    def __init__(self, message: str, step_index: Optional[int] = None, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.step_index = step_index
        self.errors = errors if errors is not None else []

class InvalidMethodError(MethodError):
    """Exception raised for issues with the overall method structure or total duration."""
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors if errors is not None else []

class TemplateError(MethodError):
    """Exception raised for issues with method templates."""
    pass

# --- Enums ---
class StepType(Enum):
    """Defines the types of method steps."""
    STEP = 'step'
    GRADIENT = 'gradient'

# --- Data Models ---
@dataclass
class MethodStep:
    """Represents a single method step (isocratic)."""
    step_type: StepType
    duration: float
    flow_rate: float
    gradient: float
    collection: bool

    def __post_init__(self):
        """Validate step properties upon initialization."""
        if not isinstance(self.step_type, StepType):
            raise ValueError(f"step_type must be an instance of StepType Enum, got {self.step_type}")
        if self.duration <= 0:
            raise ValueError("Duration must be positive.")
        if self.flow_rate <= 0: # Flow rate should generally be positive
            raise ValueError("Flow rate must be positive.")
        if not (0 <= self.gradient <= 100):
            raise ValueError("Isocratic gradient must be between 0 and 100%.")
        if not isinstance(self.collection, bool):
            raise ValueError("Collection must be a boolean.")

@dataclass
class MethodGradientStep:
    """Represents a method step involving a gradient change."""
    step_type: StepType
    duration: float
    flow_rate: float
    start_gradient: float
    end_gradient: float
    collection: bool

    def __post_init__(self):
        """Validate gradient step properties upon initialization."""
        if not isinstance(self.step_type, StepType):
            raise ValueError(f"step_type must be an instance of StepType Enum, got {self.step_type}")
        if self.step_type != StepType.GRADIENT:
            raise ValueError(f"step_type for MethodGradientStep must be {StepType.GRADIENT.value}, got {self.step_type.value}")
        if self.duration <= 0:
            raise ValueError("Duration must be positive.")
        if self.flow_rate <= 0:
            raise ValueError("Flow rate must be positive.")
        if not (0 <= self.start_gradient <= 100):
            raise ValueError("Start gradient must be between 0 and 100%.")
        if not (0 <= self.end_gradient <= 100):
            raise ValueError("End gradient must be between 0 and 100%.")
        if not isinstance(self.collection, bool):
            raise ValueError("Collection must be a boolean.")

# Type alias for a general method step
AnyMethodStep = Union[MethodStep, MethodGradientStep]

# --- Method Template Manager ---
class MethodTemplateManager:
    """Manages predefined method templates."""
    def __init__(self, template_file: Optional[str] = None):
        self._templates: Dict[str, List[Dict]] = {}
        self.template_file = template_file
        if self.template_file:
            self._load_templates()
        else:
            self._templates = {
                'Size Exclusion': [
                    {'type': 'step', 'duration': 30, 'flow_rate': 1.0, 'gradient': 0, 'collection': True},
                    {'type': 'step', 'duration': 60, 'flow_rate': 1.0, 'gradient': 0, 'collection': True}
                ],
                'Ion Exchange': [
                    {'type': 'step', 'duration': 10, 'flow_rate': 1.0, 'gradient': 0, 'collection': False},
                    {'type': 'gradient', 'duration': 30, 'flow_rate': 1.0, 'start': 0, 'end': 100, 'collection': True},
                    {'type': 'step', 'duration': 10, 'flow_rate': 1.0, 'gradient': 100, 'collection': False}
                ]
            }
            logging.info("Initialized with default templates.")


    def _load_templates(self):
        """Loads templates from a JSON file."""
        if not self.template_file:
            return

        try:
            with open(self.template_file, 'r') as f:
                self._templates = json.load(f)
            logging.info(f"Loaded templates from {self.template_file}")
        except FileNotFoundError:
            logging.warning(f"Template file not found at {self.template_file}. Starting with no loaded templates.")
            self._templates = {}
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {self.template_file}. Starting with no loaded templates.")
            self._templates = {}
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading templates: {e}. Starting with no loaded templates.")
            self._templates = {}

    def _save_templates(self):
        """Saves current templates to the JSON file."""
        if not self.template_file:
            logging.warning("No template file specified. Cannot save templates.")
            return

        try:
            with open(self.template_file, 'w') as f:
                json.dump(self._templates, f, indent=4)
            logging.info(f"Templates saved to {self.template_file}")
        except IOError as e:
            logging.error(f"Error saving templates to {self.template_file}: {e}")

    def add_template(self, name: str, steps_data: List[Dict]):
        """
        Adds a new method template.
        Validates the structure of the steps before adding.
        """
        if not name:
            raise TemplateError("Template name cannot be empty.")
        if name in self._templates:
            logging.warning(f"Template '{name}' already exists. Overwriting.")

        parsed_steps: List[AnyMethodStep] = []
        try:
            for i, step_data in enumerate(steps_data):
                step_type_str = step_data.get('type')
                if not step_type_str:
                    raise TemplateError(f"Step {i+1} in template '{name}' is missing 'type'.")

                try:
                    step_type = StepType(step_type_str)
                except ValueError:
                    raise TemplateError(f"Step {i+1} in template '{name}' has an unknown step type: '{step_type_str}'.")

                if step_type == StepType.STEP:
                    parsed_steps.append(MethodStep(
                        step_type=step_type,
                        duration=step_data['duration'],
                        flow_rate=step_data['flow_rate'],
                        gradient=step_data['gradient'],
                        collection=step_data['collection']
                    ))
                elif step_type == StepType.GRADIENT:
                    parsed_steps.append(MethodGradientStep(
                        step_type=step_type,
                        duration=step_data['duration'],
                        flow_rate=step_data['flow_rate'],
                        start_gradient=step_data['start'],
                        end_gradient=step_data['end'],
                        collection=step_data['collection']
                    ))
        except (KeyError, ValueError) as e:
            raise TemplateError(f"Invalid step data in template '{name}': {e}. Please check structure.")
        except Exception as e:
            raise TemplateError(f"An unexpected error occurred while parsing steps for template '{name}': {e}")

        self._templates[name] = steps_data
        logging.info(f"Template '{name}' added/updated successfully.")
        self._save_templates()

    def get_template(self, name: str) -> List[AnyMethodStep]:
        """
        Retrieves a method template by name and parses its steps into MethodStep/MethodGradientStep objects.
        """
        template_data = self._templates.get(name)
        if not template_data:
            raise TemplateError(f"Template '{name}' not found.")

        parsed_steps: List[AnyMethodStep] = []
        for i, step_data in enumerate(template_data):
            try:
                step_type_str = step_data.get('type')
                if not step_type_str:
                    raise TemplateError(f"Step {i+1} in template '{name}' is missing 'type'.")

                step_type = StepType(step_type_str)

                if step_type == StepType.STEP:
                    parsed_steps.append(MethodStep(
                        step_type=step_type,
                        duration=step_data['duration'],
                        flow_rate=step_data['flow_rate'],
                        gradient=step_data['gradient'],
                        collection=step_data['collection']
                    ))
                elif step_type == StepType.GRADIENT:
                    parsed_steps.append(MethodGradientStep(
                        step_type=step_type,
                        duration=step_data['duration'],
                        flow_rate=step_data['flow_rate'],
                        start_gradient=step_data['start'],
                        end_gradient=step_data['end'],
                        collection=step_data['collection']
                    ))
            except (KeyError, ValueError) as e:
                logging.error(f"Error parsing step {i+1} in template '{name}': {e}. Skipping this step or template.")
                raise TemplateError(f"Failed to parse template '{name}' due to invalid step data at step {i+1}: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while parsing step {i+1} in template '{name}': {e}.")
                raise TemplateError(f"Failed to parse template '{name}' due to an unexpected error at step {i+1}: {e}")
        return parsed_steps

    def list_templates(self) -> List[str]:
        """Lists all available template names."""
        return list(self._templates.keys())

    def delete_template(self, name: str):
        """Deletes a template by name."""
        if name not in self._templates:
            logging.warning(f"Template '{name}' does not exist.")
            return
        del self._templates[name]
        logging.info(f"Template '{name}' deleted.")
        self._save_templates()

# --- Method Validator ---
class MethodValidator:
    """Validates method steps and overall method integrity."""
    def __init__(self, config: SystemConfig):
        self.config = config

    def _validate_common_step_properties(self, step: AnyMethodStep, index: int) -> List[str]:
        """Validates properties common to all step types."""
        errors = []
        if step.flow_rate > self.config.max_flow_rate_ml_min:
            errors.append(f"Flow rate {step.flow_rate} mL/min exceeds maximum {self.config.max_flow_rate_ml_min} mL/min.")
        if step.duration <= 0:
            errors.append("Duration must be positive.")
        # Pressure validation would go here, often requiring more context (e.g., column type, solvent viscosity)
        # For simplicity, let's assume a simplified pressure check related to flow rate.
        # In a real system, this would be a complex calculation or lookup.
        # if step.flow_rate * SOME_CONSTANT > self.config.max_pressure_psi:
        #     errors.append(f"Simulated pressure exceeds maximum {self.config.max_pressure_psi} PSI.")
        return errors

    def _validate_isocratic_step(self, step: MethodStep, index: int) -> List[str]:
        """Validates properties specific to an isocratic step."""
        errors = self._validate_common_step_properties(step, index)
        if not (0 <= step.gradient <= 100):
            errors.append("Gradient must be between 0 and 100%.")
        return errors

    def _validate_gradient_step(self, step: MethodGradientStep, index: int) -> List[str]:
        """Validates properties specific to a gradient step."""
        errors = self._validate_common_step_properties(step, index)
        if not (0 <= step.start_gradient <= 100):
            errors.append("Start gradient must be between 0 and 100%.")
        if not (0 <= step.end_gradient <= 100):
            errors.append("End gradient must be between 0 and 100%.")
        if step.start_gradient == step.end_gradient:
            errors.append("Gradient step has identical start and end gradient values; consider using an isocratic step.")
        return errors

    def validate_step(self, step: AnyMethodStep, index: int = 0) -> List[str]:
        """
        Validates a single method step.
        Returns a list of error messages. An empty list means no errors.
        """
        if isinstance(step, MethodStep):
            return self._validate_isocratic_step(step, index)
        elif isinstance(step, MethodGradientStep):
            return self._validate_gradient_step(step, index)
        else:
            return [f"Step {index+1}: Unknown step type: {type(step)}"]

    def validate_method(self, steps: List[AnyMethodStep]) -> None:
        """
        Validates an entire method composed of multiple steps.
        Raises InvalidMethodError or InvalidMethodStepError if validation fails.
        """
        method_errors = []
        total_duration = 0.0

        if not steps:
            method_errors.append("Method must contain at least one step.")
            raise InvalidMethodError("Method validation failed.", errors=method_errors)

        for i, step in enumerate(steps):
            step_errors = self.validate_step(step, i)
            if step_errors:
                for err in step_errors:
                    method_errors.append(f"Step {i+1}: {err}")
            total_duration += step.duration

        if total_duration > self.config.max_method_duration_minutes:
            method_errors.append(f"Method duration ({total_duration:.1f} minutes) exceeds maximum {self.config.max_method_duration_minutes} minutes (24 hours).")

        # Additional validation logic:
        # Ensure a collection step exists if desired (business rule example)
        if not any(step.collection for step in steps):
            logging.warning("No collection steps found in the method. Is this intentional?")

        # Check for abrupt gradient changes between steps (e.g., if step N ends at 20% and N+1 starts at 80% without a gradient in between)
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i+1]

            current_end_gradient = None
            if isinstance(current_step, MethodStep):
                current_end_gradient = current_step.gradient
            elif isinstance(current_step, MethodGradientStep):
                current_end_gradient = current_step.end_gradient

            next_start_gradient = None
            if isinstance(next_step, MethodStep):
                next_start_gradient = next_step.gradient
            elif isinstance(next_step, MethodGradientStep):
                next_start_gradient = next_step.start_gradient

            if current_end_gradient is not None and next_start_gradient is not None:
                if abs(current_end_gradient - next_start_gradient) > 0.01 and \
                   not (isinstance(current_step, MethodGradientStep) or isinstance(next_step, MethodGradientStep)):
                    # This check is for large, instantaneous jumps between two non-gradient steps.
                    # A small tolerance (0.01) is used for floating point comparisons.
                    logging.warning(f"Abrupt gradient change detected between step {i+1} (ends at {current_end_gradient}%) and step {i+2} (starts at {next_start_gradient}%). Consider adding a gradient step.")


        if method_errors:
            raise InvalidMethodError("Method validation failed.", errors=method_errors)
        
        logging.info("Method validated successfully.")


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Configuration Setup
    # Create a config.json file (optional)
    # with content like:
    # {
    #     "max_pressure_psi": 3500.0,
    #     "max_flow_rate_ml_min": 12.0,
    #     "max_method_duration_minutes": 1440.0
    # }
    config_file = "config.json"
    system_configuration = SystemConfig.from_json(config_file)
    logging.info(f"System Configuration: Max Flow Rate={system_configuration.max_flow_rate_ml_min} mL/min, Max Duration={system_configuration.max_method_duration_minutes} min")

    # 2. Template Manager Usage
    template_manager = MethodTemplateManager(template_file="method_templates.json")

    # Add a new template
    try:
        template_manager.add_template(
            "Washing Protocol",
            [
                {'type': 'step', 'duration': 5, 'flow_rate': 2.0, 'gradient': 0, 'collection': False},
                {'type': 'step', 'duration': 15, 'flow_rate': 2.5, 'gradient': 5, 'collection': False}
            ]
        )
    except TemplateError as e:
        logging.error(f"Failed to add template: {e}")

    # List templates
    print("\nAvailable Templates:")
    for template_name in template_manager.list_templates():
        print(f"- {template_name}")

    # Get and use a template
    try:
        size_exclusion_template_steps = template_manager.get_template("Size Exclusion")
        print("\nSize Exclusion Template Steps:")
        for step in size_exclusion_template_steps:
            print(step)
    except TemplateError as e:
        logging.error(f"Error getting template: {e}")

    # 3. Method Validation Usage
    validator = MethodValidator(config=system_configuration)

    # Valid Method Example
    print("\n--- Valid Method Test ---")
    valid_method_steps: List[AnyMethodStep] = [
        MethodStep(step_type=StepType.STEP, duration=10, flow_rate=1.0, gradient=0, collection=True),
        MethodGradientStep(step_type=StepType.GRADIENT, duration=20, flow_rate=1.0, start_gradient=0, end_gradient=50, collection=True),
        MethodStep(step_type=StepType.STEP, duration=5, flow_rate=1.0, gradient=50, collection=False)
    ]
    try:
        validator.validate_method(valid_method_steps)
        print("Valid method passed validation.")
    except InvalidMethodError as e:
        print(f"Valid method validation failed: {e}")
        for err in e.errors:
            print(f"  - {err}")

    # Invalid Method Example: High Flow Rate
    print("\n--- Invalid Method Test: High Flow Rate ---")
    invalid_method_high_flow: List[AnyMethodStep] = [
        MethodStep(step_type=StepType.STEP, duration=10, flow_rate=15.0, gradient=0, collection=True), # Exceeds max
    ]
    try:
        validator.validate_method(invalid_method_high_flow)
        print("Invalid method (high flow) passed validation (unexpected).")
    except InvalidMethodError as e:
        print(f"Invalid method (high flow) validation failed as expected: {e}")
        for err in e.errors:
            print(f"  - {err}")

    # Invalid Method Example: Long Duration
    print("\n--- Invalid Method Test: Long Duration ---")
    invalid_method_long_duration: List[AnyMethodStep] = [
        MethodStep(step_type=StepType.STEP, duration=721, flow_rate=1.0, gradient=0, collection=True),
        MethodStep(step_type=StepType.STEP, duration=720, flow_rate=1.0, gradient=0, collection=True) # Total 1441 min > 1440 min
    ]
    try:
        validator.validate_method(invalid_method_long_duration)
        print("Invalid method (long duration) passed validation (unexpected).")
    except InvalidMethodError as e:
        print(f"Invalid method (long duration) validation failed as expected: {e}")
        for err in e.errors:
            print(f"  - {err}")

    # Invalid Method Example: Invalid Gradient Value in Isocratic Step
    print("\n--- Invalid Method Test: Invalid Isocratic Gradient ---")
    invalid_method_gradient: List[AnyMethodStep] = [
        MethodStep(step_type=StepType.STEP, duration=10, flow_rate=1.0, gradient=101, collection=True), # Invalid gradient
    ]
    try:
        validator.validate_method(invalid_method_gradient)
        print("Invalid method (gradient) passed validation (unexpected).")
    except InvalidMethodError as e:
        print(f"Invalid method (gradient) validation failed as expected: {e}")
        for err in e.errors:
            print(f"  - {err}")

    # Test an abrupt gradient change warning
    print("\n--- Method Test: Abrupt Gradient Change Warning ---")
    abrupt_gradient_method: List[AnyMethodStep] = [
        MethodStep(step_type=StepType.STEP, duration=10, flow_rate=1.0, gradient=10, collection=True),
        MethodStep(step_type=StepType.STEP, duration=10, flow_rate=1.0, gradient=90, collection=True) # Abrupt change
    ]
    try:
        validator.validate_method(abrupt_gradient_method)
        print("Method with abrupt gradient change passed validation (check logs for warning).")
    except InvalidMethodError as e:
        print(f"Method with abrupt gradient change validation failed (unexpected): {e}")

    # Clean up (optional: delete the test template)
    try:
        template_manager.delete_template("Washing Protocol")
    except TemplateError as e:
        logging.error(f"Error deleting template: {e}")