import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SampleManager')

# --- Custom Exceptions ---
class SampleManagerError(Exception):
    """Base exception for SampleManager related errors."""
    pass

class DuplicateSampleError(SampleManagerError):
    """Exception raised when attempting to register a sample with an ID that already exists."""
    pass

class SampleNotFoundError(SampleManagerError):
    """Exception raised when a requested sample ID does not exist."""
    pass

class InvalidFractionError(SampleManagerError):
    """Exception raised for issues with fraction data (e.g., invalid volume, duplicate fraction number)."""
    pass

class InvalidAnalysisResultError(SampleManagerError):
    """Exception raised for invalid analysis results."""
    pass

# --- Enums ---
class SampleStatus(Enum):
    """Defines the current status of a sample in the system."""
    REGISTERED = "Registered"
    INJECTED = "Injected"
    COLLECTING_FRACTIONS = "Collecting Fractions"
    FRACTIONS_COLLECTED = "Fractions Collected"
    ANALYZED = "Analyzed"
    COMPLETED = "Completed"
    ERROR = "Error"

    def __str__(self):
        return self.value

# --- Data Models ---
@dataclass
class Fraction:
    """Represents a collected fraction from a sample."""
    fraction_number: int
    volume_ml: float
    collection_time: datetime = field(default_factory=datetime.now)
    peak_data: Dict[str, Any] = field(default_factory=dict) # Can be detailed peak analysis results

    def __post_init__(self):
        if not isinstance(self.fraction_number, int) or self.fraction_number <= 0:
            raise ValueError("Fraction number must be a positive integer.")
        if not isinstance(self.volume_ml, (int, float)) or self.volume_ml < 0:
            # Volume can be 0 if a fraction tube was skipped, but usually positive
            raise ValueError("Volume must be a non-negative number.")
        if not isinstance(self.collection_time, datetime):
            raise ValueError("Collection time must be a datetime object.")
        if not isinstance(self.peak_data, dict):
            raise ValueError("Peak data must be a dictionary.")

@dataclass
class AnalysisResult:
    """Represents the results obtained from analyzing a sample (e.g., purity, concentration)."""
    analysis_date: datetime = field(default_factory=datetime.now)
    result_data: Dict[str, Any] = field(default_factory=dict) # e.g., {'purity': 98.5, 'concentration': 1.2}
    comments: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.analysis_date, datetime):
            raise ValueError("Analysis date must be a datetime object.")
        if not isinstance(self.result_data, dict):
            raise ValueError("Result data must be a dictionary.")

@dataclass
class Sample:
    """Represents a single sample managed by the system."""
    sample_id: str
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., {'name': 'Protein X Batch 1', 'concentration': '2mg/ml'}
    registration_time: datetime = field(default_factory=datetime.now)
    injection_time: Optional[datetime] = None
    status: SampleStatus = SampleStatus.REGISTERED
    collection_fractions: List[Fraction] = field(default_factory=list)
    analysis_results: Optional[AnalysisResult] = None
    notes: List[str] = field(default_factory=list) # For any additional observations

    def __post_init__(self):
        if not self.sample_id:
            raise ValueError("Sample ID cannot be empty.")
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary.")
        if not isinstance(self.registration_time, datetime):
            raise ValueError("Registration time must be a datetime object.")
        if not isinstance(self.status, SampleStatus):
            raise ValueError("Status must be a SampleStatus enum member.")
        if not isinstance(self.collection_fractions, list) or not all(isinstance(f, Fraction) for f in self.collection_fractions):
            raise ValueError("Collection fractions must be a list of Fraction objects.")
        if self.analysis_results is not None and not isinstance(self.analysis_results, AnalysisResult):
            raise ValueError("Analysis results must be an AnalysisResult object or None.")
        if not isinstance(self.notes, list) or not all(isinstance(n, str) for n in self.notes):
            raise ValueError("Notes must be a list of strings.")

    def add_note(self, note: str):
        """Adds a note to the sample's history."""
        self.notes.append(f"{datetime.now()}: {note}")
        logger.debug(f"Note added to sample {self.sample_id}: {note}")

    def update_status(self, new_status: SampleStatus):
        """Updates the status of the sample."""
        if not isinstance(new_status, SampleStatus):
            raise TypeError("New status must be a SampleStatus enum.")
        self.status = new_status
        logger.info(f"Sample '{self.sample_id}' status updated to {new_status}.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Sample object to a dictionary for serialization."""
        data = self.__dict__.copy()
        data['registration_time'] = data['registration_time'].isoformat()
        if data['injection_time']:
            data['injection_time'] = data['injection_time'].isoformat()
        data['status'] = data['status'].value
        data['collection_fractions'] = [f.__dict__ for f in data['collection_fractions']]
        for fraction_data in data['collection_fractions']:
            fraction_data['collection_time'] = fraction_data['collection_time'].isoformat()
        if data['analysis_results']:
            data['analysis_results'] = data['analysis_results'].__dict__.copy()
            data['analysis_results']['analysis_date'] = data['analysis_results']['analysis_date'].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a Sample object from a dictionary (for deserialization)."""
        data['registration_time'] = datetime.fromisoformat(data['registration_time'])
        if data.get('injection_time'):
            data['injection_time'] = datetime.fromisoformat(data['injection_time'])
        data['status'] = SampleStatus(data['status'])
        data['collection_fractions'] = [
            Fraction(
                fraction_number=f['fraction_number'],
                volume_ml=f['volume_ml'],
                collection_time=datetime.fromisoformat(f['collection_time']),
                peak_data=f['peak_data']
            ) for f in data.get('collection_fractions', [])
        ]
        if data.get('analysis_results'):
            ar_data = data['analysis_results']
            data['analysis_results'] = AnalysisResult(
                analysis_date=datetime.fromisoformat(ar_data['analysis_date']),
                result_data=ar_data['result_data'],
                comments=ar_data.get('comments')
            )
        return cls(**data)


# --- Sample Manager ---
class SampleManager:
    """
    Manages the registration, tracking, and analysis results for samples.
    Provides persistence capabilities.
    """
    def __init__(self, data_file: Optional[str] = None):
        """
        Initializes the SampleManager.
        Args:
            data_file (Optional[str]): Path to a JSON file for saving/loading sample data.
        """
        self._samples: Dict[str, Sample] = {}
        self.data_file = data_file
        if self.data_file:
            self._load_samples()
        logger.info(f"SampleManager initialized. Data file: {self.data_file if self.data_file else 'None'}")

    def _load_samples(self):
        """Loads sample data from the configured JSON file."""
        if not self.data_file:
            return

        try:
            with open(self.data_file, 'r') as f:
                raw_data = json.load(f)
                self._samples = {
                    sample_id: Sample.from_dict(sample_data)
                    for sample_id, sample_data in raw_data.items()
                }
            logger.info(f"Loaded {len(self._samples)} samples from {self.data_file}.")
        except FileNotFoundError:
            logger.warning(f"Sample data file not found at {self.data_file}. Starting with an empty sample list.")
            self._samples = {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.data_file}. Starting with an empty sample list.")
            self._samples = {}
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading samples: {e}. Starting with an empty sample list.")
            self._samples = {}

    def _save_samples(self):
        """Saves current sample data to the configured JSON file."""
        if not self.data_file:
            logger.warning("No data file specified. Cannot save samples.")
            return

        try:
            with open(self.data_file, 'w') as f:
                # Convert all Sample objects to dictionaries for JSON serialization
                serializable_samples = {
                    sample_id: sample.to_dict()
                    for sample_id, sample in self._samples.items()
                }
                json.dump(serializable_samples, f, indent=4)
            logger.info(f"Saved {len(self._samples)} samples to {self.data_file}.")
        except IOError as e:
            logger.error(f"Error saving samples to {self.data_file}: {e}")

    def register_sample(self, sample_id: str, metadata: Dict[str, Any]) -> Sample:
        """
        Registers a new sample with the manager.
        Args:
            sample_id (str): A unique identifier for the sample.
            metadata (Dict[str, Any]): A dictionary of arbitrary metadata for the sample.

        Returns:
            Sample: The newly registered Sample object.

        Raises:
            DuplicateSampleError: If a sample with the given ID already exists.
            ValueError: If sample_id is empty or metadata is not a dictionary.
        """
        if not sample_id:
            raise ValueError("Sample ID cannot be empty.")
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary.")

        if sample_id in self._samples:
            logger.error(f"Attempted to register duplicate sample ID: {sample_id}")
            raise DuplicateSampleError(f"Sample with ID '{sample_id}' already exists.")

        try:
            new_sample = Sample(sample_id=sample_id, metadata=metadata)
            self._samples[sample_id] = new_sample
            self._save_samples()
            logger.info(f"Sample '{sample_id}' registered successfully.")
            return new_sample
        except ValueError as e:
            logger.error(f"Validation error during sample registration for '{sample_id}': {e}")
            raise SampleManagerError(f"Failed to register sample '{sample_id}': {e}")

    def get_sample(self, sample_id: str) -> Sample:
        """
        Retrieves a sample by its ID.
        Args:
            sample_id (str): The ID of the sample to retrieve.

        Returns:
            Sample: The requested Sample object.

        Raises:
            SampleNotFoundError: If the sample ID does not exist.
        """
        sample = self._samples.get(sample_id)
        if not sample:
            logger.warning(f"Attempted to retrieve non-existent sample ID: {sample_id}")
            raise SampleNotFoundError(f"Sample with ID '{sample_id}' not found.")
        return sample

    def update_sample_injection_time(self, sample_id: str, injection_time: Optional[datetime] = None):
        """
        Updates the injection time for a sample and sets its status to INJECTED.
        Args:
            sample_id (str): The ID of the sample.
            injection_time (Optional[datetime]): The time of injection. Defaults to now if None.

        Raises:
            SampleNotFoundError: If the sample ID does not exist.
            ValueError: If injection_time is not a datetime object.
        """
        sample = self.get_sample(sample_id) # This will raise SampleNotFoundError if not found
        if injection_time is None:
            injection_time = datetime.now()
        if not isinstance(injection_time, datetime):
            raise ValueError("Injection time must be a datetime object.")

        sample.injection_time = injection_time
        sample.update_status(SampleStatus.INJECTED)
        self._save_samples()
        logger.info(f"Sample '{sample_id}' injection time set to {injection_time}.")

    def track_fraction(self, sample_id: str, fraction_number: int, volume_ml: float, peak_data: Dict[str, Any]):
        """
        Tracks a collected fraction for a specified sample.
        Args:
            sample_id (str): The ID of the sample to which the fraction belongs.
            fraction_number (int): The unique number of the fraction.
            volume_ml (float): The volume collected in this fraction (in mL).
            peak_data (Dict[str, Any]): Relevant peak data associated with this fraction
                                        (e.g., peak area, retention time, standard match).

        Raises:
            SampleNotFoundError: If the sample ID does not exist.
            InvalidFractionError: If the fraction number is not positive or already exists for the sample,
                                  or if volume is negative.
        """
        sample = self.get_sample(sample_id) # This will raise SampleNotFoundError if not found

        if not isinstance(fraction_number, int) or fraction_number <= 0:
            raise InvalidFractionError("Fraction number must be a positive integer.")
        if not isinstance(volume_ml, (int, float)) or volume_ml < 0:
            raise InvalidFractionError("Volume must be a non-negative number.")
        if not isinstance(peak_data, dict):
            raise InvalidFractionError("Peak data must be a dictionary.")

        # Check for duplicate fraction number for the same sample
        if any(f.fraction_number == fraction_number for f in sample.collection_fractions):
            logger.error(f"Duplicate fraction number {fraction_number} for sample '{sample_id}'.")
            raise InvalidFractionError(f"Fraction number {fraction_number} already exists for sample '{sample_id}'.")

        try:
            new_fraction = Fraction(
                fraction_number=fraction_number,
                volume_ml=volume_ml,
                peak_data=peak_data
            )
            sample.collection_fractions.append(new_fraction)
            sample.update_status(SampleStatus.COLLECTING_FRACTIONS) # Or FRACTIONS_COLLECTED if this is the last one
            self._save_samples()
            logger.info(f"Fraction {fraction_number} ({volume_ml}mL) tracked for sample '{sample_id}'.")
        except ValueError as e:
            logger.error(f"Validation error during fraction tracking for sample '{sample_id}', fraction {fraction_number}: {e}")
            raise InvalidFractionError(f"Failed to track fraction {fraction_number}: {e}")

    def record_analysis_results(self, sample_id: str, result_data: Dict[str, Any], comments: Optional[str] = None):
        """
        Records the final analysis results for a sample.
        Args:
            sample_id (str): The ID of the sample.
            result_data (Dict[str, Any]): A dictionary containing the analysis results
                                        (e.g., purity, concentration).
            comments (Optional[str]): Any additional comments about the analysis.

        Raises:
            SampleNotFoundError: If the sample ID does not exist.
            InvalidAnalysisResultError: If result_data is not a dictionary.
        """
        sample = self.get_sample(sample_id) # This will raise SampleNotFoundError if not found

        if not isinstance(result_data, dict):
            raise InvalidAnalysisResultError("Result data must be a dictionary.")

        try:
            analysis_result = AnalysisResult(
                result_data=result_data,
                comments=comments
            )
            sample.analysis_results = analysis_result
            sample.update_status(SampleStatus.ANALYZED)
            self._save_samples()
            logger.info(f"Analysis results recorded for sample '{sample_id}'.")
        except ValueError as e:
            logger.error(f"Validation error during analysis result recording for sample '{sample_id}': {e}")
            raise InvalidAnalysisResultError(f"Failed to record analysis results for sample '{sample_id}': {e}")

    def get_all_samples(self) -> List[Sample]:
        """Returns a list of all registered samples."""
        return list(self._samples.values())

    def delete_sample(self, sample_id: str):
        """
        Deletes a sample from the manager.
        Args:
            sample_id (str): The ID of the sample to delete.

        Raises:
            SampleNotFoundError: If the sample ID does not exist.
        """
        if sample_id not in self._samples:
            raise SampleNotFoundError(f"Cannot delete. Sample with ID '{sample_id}' not found.")
        
        del self._samples[sample_id]
        self._save_samples()
        logger.info(f"Sample '{sample_id}' deleted successfully.")


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the SampleManager with a data file for persistence
    sample_manager = SampleManager(data_file="samples_data.json")

    # --- Registering Samples ---
    print("\n--- Registering Samples ---")
    try:
        sample1 = sample_manager.register_sample(
            "Sample-001", {"protein_name": "BSA", "concentration": "5mg/mL"}
        )
        print(f"Registered: {sample1.sample_id} - Status: {sample1.status}")

        sample2 = sample_manager.register_sample(
            "Sample-002", {"protein_name": "Lysozyme", "batch": "L-2024-01"}
        )
        print(f"Registered: {sample2.sample_id} - Status: {sample2.status}")

        # Attempt to register a duplicate (will raise an error)
        sample_manager.register_sample("Sample-001", {"protein_name": "Should Fail"})
    except DuplicateSampleError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Validation Error: {e}")

    # --- Updating Injection Time ---
    print("\n--- Updating Injection Time ---")
    try:
        sample_manager.update_sample_injection_time("Sample-001")
        sample1_updated = sample_manager.get_sample("Sample-001")
        print(f"Sample-001 Injected at: {sample1_updated.injection_time.strftime('%Y-%m-%d %H:%M:%S')} - Status: {sample1_updated.status}")
    except SampleNotFoundError as e:
        print(f"Error: {e}")

    # --- Tracking Fractions ---
    print("\n--- Tracking Fractions ---")
    try:
        sample_manager.track_fraction(
            "Sample-001", 1, 1.5, {"peak_time": 10.2, "area": 1500, "matched_standard": "Protein A"}
        )
        sample_manager.track_fraction(
            "Sample-001", 2, 2.0, {"peak_time": 11.5, "area": 2100, "matched_standard": "Protein A"}
        )
        sample_manager.track_fraction(
            "Sample-002", 1, 0.8, {"peak_time": 5.1, "area": 800}
        )
        # Attempt to track a duplicate fraction number for the same sample
        sample_manager.track_fraction(
            "Sample-001", 1, 1.0, {"peak_time": 10.0, "area": 100}
        )
    except InvalidFractionError as e:
        print(f"Error: {e}")
    except SampleNotFoundError as e:
        print(f"Error: {e}")

    # --- Recording Analysis Results ---
    print("\n--- Recording Analysis Results ---")
    try:
        sample_manager.record_analysis_results(
            "Sample-001",
            {"purity": 99.1, "concentration_mg_ml": 4.8},
            "High purity, excellent run."
        )
        sample1_final = sample_manager.get_sample("Sample-001")
        print(f"Sample-001 Analysis Results: {sample1_final.analysis_results.result_data} - Status: {sample1_final.status}")
        print(f"  Comments: {sample1_final.analysis_results.comments}")

        # Try to add results to a non-existent sample
        sample_manager.record_analysis_results("NonExistentSample", {"purity": 50})
    except SampleNotFoundError as e:
        print(f"Error: {e}")
    except InvalidAnalysisResultError as e:
        print(f"Error: {e}")

    # --- Listing All Samples ---
    print("\n--- All Samples ---")
    all_samples = sample_manager.get_all_samples()
    for s in all_samples:
        print(f"ID: {s.sample_id}, Status: {s.status}, Fractions: {len(s.collection_fractions)}")
        if s.analysis_results:
            print(f"  Analysis: {s.analysis_results.result_data}")

    # --- Demonstrating persistence ---
    print("\n--- Demonstrating Persistence ---")
    # To see persistence in action, run the script once to save data.
    # Then comment out the registration lines and run again.
    # The samples from the first run should be loaded.
    print("Restart the script or create a new SampleManager instance to see loaded data.")
    new_manager = SampleManager(data_file="samples_data.json")
    print(f"New manager loaded {len(new_manager.get_all_samples())} samples from file.")
    if new_manager.get_sample("Sample-001").status == SampleStatus.ANALYZED:
        print("Sample-001's status successfully loaded from file.")


    # --- Deleting a Sample ---
    print("\n--- Deleting a Sample ---")
    try:
        sample_manager.delete_sample("Sample-002")
        print("Sample-002 deleted.")
        print(f"Samples remaining: {len(sample_manager.get_all_samples())}")
        sample_manager.get_sample("Sample-002") # This should now fail
    except SampleNotFoundError as e:
        print(f"Caught expected error after deletion: {e}")