import numpy as np
import logging
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ChromatographyPeakAnalysis')

# --- Configuration for Peak Analysis ---
@dataclass
class PeakAnalysisConfig:
    """
    Configuration parameters for peak detection and analysis.
    """
    # Peak detection parameters for scipy.signal.find_peaks
    min_peak_height: float = 0.1       # Minimum height of a peak (in UV units)
    min_peak_distance: int = 10        # Minimum horizontal distance (in data points) between neighboring peaks
    min_peak_prominence: Optional[float] = None # Minimum prominence of a peak (more robust than height)
    min_peak_width: Optional[float] = None # Minimum width of a peak (in data points)

    # Tolerance for matching peaks to standards (in time units)
    retention_time_tolerance: float = 0.5 # e.g., +/- 0.5 minutes

    # Parameters for peak property calculations
    peak_integration_window_points: int = 20 # Points to consider on each side for simple area integration
    half_height_interp_steps: int = 10       # Number of interpolation steps for precise half-height width

    # Baseline correction parameters (simple moving average for now)
    baseline_window_points: int = 50   # Window size for moving average baseline estimation

    def __post_init__(self):
        if self.min_peak_height <= 0:
            raise ValueError("min_peak_height must be positive.")
        if self.min_peak_distance <= 0:
            raise ValueError("min_peak_distance must be positive.")
        if self.retention_time_tolerance < 0:
            raise ValueError("retention_time_tolerance cannot be negative.")
        if self.peak_integration_window_points <= 0:
            raise ValueError("peak_integration_window_points must be positive.")
        if self.half_height_interp_steps <= 0:
            raise ValueError("half_height_interp_steps must be positive.")
        if self.baseline_window_points <= 0:
            raise ValueError("baseline_window_points must be positive.")

# --- Custom Exceptions ---
class PeakAnalysisError(Exception):
    """Base exception for peak analysis related errors."""
    pass

class InvalidDataError(PeakAnalysisError):
    """Exception raised for invalid input data."""
    pass

class PeakCalculationError(PeakAnalysisError):
    """Exception raised for errors during peak property calculation."""
    pass

# --- Data Models ---
@dataclass
class ChromatographicPeak:
    """
    Represents a detected chromatographic peak and its properties.
    """
    index: int                  # Index of the peak apex in the original data array
    retention_time: float       # Time at the peak apex
    height: float               # Height of the peak (from baseline if corrected)
    area: float                 # Calculated area of the peak (from baseline if corrected)
    width_at_half_height: float # Peak width at half maximum (FWHM)
    prominence: float           # Prominence of the peak
    start_time: float           # Estimated start time of the peak
    end_time: float             # Estimated end time of the peak
    standard_match: Optional[str] = None # Name of the matched standard, if any
    # Additional properties can be added here, e.g., asymmetry, plate count

    def __post_init__(self):
        if self.index < 0:
            raise ValueError("Peak index cannot be negative.")
        if self.retention_time < 0:
            raise ValueError("Retention time cannot be negative.")
        if self.height < 0:
            logger.warning(f"Peak height is negative ({self.height}). This might indicate issues with baseline correction or data quality.")
        if self.area < 0:
            logger.warning(f"Peak area is negative ({self.area}). This might indicate issues with baseline correction or data quality.")
        if self.width_at_half_height < 0:
            raise ValueError("Peak width at half height cannot be negative.")
        if self.prominence < 0:
            raise ValueError("Peak prominence cannot be negative.")
        if self.start_time < 0 or self.end_time < 0:
            raise ValueError("Peak start/end times cannot be negative.")
        if self.start_time >= self.end_time:
            raise ValueError("Peak start time must be less than end time.")


@dataclass
class ChromatographyStandard:
    """
    Represents a known standard for matching against detected peaks.
    """
    name: str
    retention_time: float
    retention_time_tolerance: float = 0.5 # Specific tolerance for this standard, can override global
    concentration: Optional[float] = None # For quantitative analysis

    def __post_init__(self):
        if not self.name:
            raise ValueError("Standard name cannot be empty.")
        if self.retention_time < 0:
            raise ValueError("Standard retention time cannot be negative.")
        if self.retention_time_tolerance < 0:
            raise ValueError("Standard retention time tolerance cannot be negative.")
        if self.concentration is not None and self.concentration < 0:
            raise ValueError("Standard concentration cannot be negative.")

class PeakAnalysis:
    """
    Handles peak detection and comprehensive analysis of chromatographic data.
    """
    def __init__(self, config: Optional[PeakAnalysisConfig] = None):
        """
        Initializes the PeakAnalysis instance.
        Args:
            config (Optional[PeakAnalysisConfig]): Configuration for peak analysis.
                                                   If None, default configuration is used.
        """
        self.config = config if config else PeakAnalysisConfig()
        self.peaks: List[ChromatographicPeak] = []
        self.baseline: Optional[np.ndarray] = None
        self.standards: List[ChromatographyStandard] = []
        self.time_data: Optional[np.ndarray] = None
        self.uv_data: Optional[np.ndarray] = None
        self.data_interval: float = 1.0 # Assuming 1 unit/point by default, should be calculated from time_data

    def set_standards(self, standards: List[ChromatographyStandard]):
        """
        Sets the standards for peak comparison.
        Args:
            standards (List[ChromatographyStandard]): A list of ChromatographyStandard objects.
        """
        if not isinstance(standards, list) or not all(isinstance(s, ChromatographyStandard) for s in standards):
            raise TypeError("Standards must be a list of ChromatographyStandard objects.")
        self.standards = standards
        logger.info(f"Peak analysis standards set: {len(standards)} standards loaded.")

    def _estimate_baseline_moving_average(self, data: np.ndarray) -> np.ndarray:
        """
        Estimates a simple baseline using a moving average approach.
        More sophisticated baseline correction algorithms (e.g., polynomial fitting,
        local minimums) might be needed for complex chromatograms.
        """
        if len(data) < self.config.baseline_window_points:
            logger.warning("Data length is less than baseline window points. Baseline estimation skipped.")
            return np.zeros_like(data, dtype=float)

        # Using convolution for a simple moving average
        window = np.ones(self.config.baseline_window_points) / self.config.baseline_window_points
        baseline = np.convolve(data, window, mode='same')
        return baseline

    def detect_peaks(self, time_data: List[float], uv_data: List[float]):
        """
        Detects peaks in UV data, calculates their properties, and stores them.
        Applies a simple moving average baseline correction before peak detection.

        Args:
            time_data (List[float]): List or array of time points.
            uv_data (List[float]): List or array of UV absorbance data.

        Raises:
            InvalidDataError: If input data is invalid or inconsistent.
            PeakCalculationError: If an error occurs during peak property calculation.
        """
        if not time_data or not uv_data:
            self.peaks = []
            self.time_data = None
            self.uv_data = None
            self.baseline = None
            logger.warning("No data provided for peak detection. Clearing previous results.")
            return

        if len(time_data) != len(uv_data):
            raise InvalidDataError("Time data and UV data must have the same length.")
        if len(time_data) < 2:
            raise InvalidDataError("Insufficient data points for peak detection (at least 2 required).")

        self.time_data = np.asarray(time_data, dtype=float)
        self.uv_data = np.asarray(uv_data, dtype=float)
        self.data_interval = (self.time_data[-1] - self.time_data[0]) / (len(self.time_data) - 1) if len(self.time_data) > 1 else 1.0

        # 1. Baseline Correction
        self.baseline = self._estimate_baseline_moving_average(self.uv_data)
        uv_data_corrected = self.uv_data - self.baseline
        logger.info("Baseline estimated and subtracted.")

        # 2. Peak Detection using scipy.signal.find_peaks
        # Pass all relevant parameters from config
        peak_indices, properties = find_peaks(
            uv_data_corrected,
            height=self.config.min_peak_height,
            distance=self.config.min_peak_distance,
            prominence=self.config.min_peak_prominence,
            width=self.config.min_peak_width
        )
        logger.info(f"Detected {len(peak_indices)} raw peaks using find_peaks.")

        self.peaks = []
        for i, peak_idx in enumerate(peak_indices):
            try:
                peak_time = self.time_data[peak_idx]
                peak_height_corrected = uv_data_corrected[peak_idx]
                peak_prominence = properties['prominences'][i]

                # Estimate peak start/end and calculate area based on corrected data
                peak_start_idx, peak_end_idx = self._estimate_peak_boundaries(uv_data_corrected, peak_idx, properties.get('left_bases', [])[i], properties.get('right_bases', [])[i])
                peak_area = self._calculate_peak_area(uv_data_corrected, peak_start_idx, peak_end_idx)

                # Calculate peak width at half maximum
                width_at_half_height = self._calculate_peak_width_at_half_height(uv_data_corrected, peak_idx, properties['peak_heights'][i])

                matched_standard = self.match_peak_to_standard(peak_time)

                peak = ChromatographicPeak(
                    index=peak_idx,
                    retention_time=peak_time,
                    height=peak_height_corrected,
                    area=peak_area,
                    width_at_half_height=width_at_half_height,
                    prominence=peak_prominence,
                    start_time=self.time_data[peak_start_idx],
                    end_time=self.time_data[peak_end_idx],
                    standard_match=matched_standard
                )
                self.peaks.append(peak)
                logger.debug(f"Added peak: {peak}")
            except (IndexError, ValueError, PeakCalculationError) as e:
                logger.error(f"Error processing peak at index {peak_idx}: {e}. Skipping this peak.")
                continue # Skip to the next peak if an error occurs for one peak

        logger.info(f"Successfully processed {len(self.peaks)} chromatographic peaks.")

    def match_peak_to_standard(self, peak_time: float) -> Optional[str]:
        """
        Matches a detected peak to known standards based on retention time.

        Args:
            peak_time (float): The retention time of the detected peak.

        Returns:
            Optional[str]: The name of the matched standard, or None if no match.
        """
        for standard in self.standards:
            tolerance = standard.retention_time_tolerance if standard.retention_time_tolerance is not None else self.config.retention_time_tolerance
            if abs(standard.retention_time - peak_time) <= tolerance:
                return standard.name
        return None

    def _estimate_peak_boundaries(self, data: np.ndarray, peak_idx: int, left_base: int, right_base: int) -> Tuple[int, int]:
        """
        Estimates the start and end indices of a peak based on its prominence
        properties (left/right bases provided by find_peaks).
        These bases usually represent the points where the peak descends to the level
        of its prominence.
        """
        if not (0 <= left_base <= peak_idx <= right_base < len(data)):
            # Fallback to a simpler window if base indices are out of bounds or invalid
            logger.warning(f"Invalid peak base indices ({left_base}, {right_base}) for peak {peak_idx}. Using fixed window.")
            window_size = self.config.peak_integration_window_points
            start_idx = max(0, peak_idx - window_size)
            end_idx = min(len(data) - 1, peak_idx + window_size)
            return start_idx, end_idx

        # The bases provided by find_peaks are usually good estimates
        return left_base, right_base

    def _calculate_peak_area(self, data: np.ndarray, start_idx: int, end_idx: int) -> float:
        """
        Calculates peak area using trapezoidal integration for a specified region.
        The data is assumed to be baseline-corrected.
        """
        if start_idx >= end_idx:
            logger.warning(f"Invalid integration range: start_idx ({start_idx}) >= end_idx ({end_idx}). Area is 0.")
            return 0.0
        if start_idx < 0 or end_idx >= len(data):
            raise PeakCalculationError(f"Integration range [{start_idx}:{end_idx}] is out of bounds for data of length {len(data)}.")

        # Ensure data values are non-negative for meaningful area
        integration_data = data[start_idx:end_idx+1]
        integration_data[integration_data < 0] = 0 # Clamp to zero for area calculation if any point is below zero

        # Using numpy.trapz, which integrates along the last axis.
        # time_step here represents the dx in the integral.
        # It's crucial that time_data is evenly spaced or this calculation needs adaptation.
        area = np.trapz(integration_data, dx=self.data_interval)
        return area

    def _calculate_peak_width_at_half_height(self, data: np.ndarray, peak_idx: int, peak_height_from_props: float) -> float:
        """
        Calculates peak width at half maximum (FWHM) with interpolation for precision.
        Args:
            data (np.ndarray): The UV data (should be baseline corrected).
            peak_idx (int): The index of the peak apex.
            peak_height_from_props (float): The actual height of the peak above its base,
                                            as calculated by find_peaks's 'peak_heights'.
        Returns:
            float: The FWHM in time units.
        Raises:
            PeakCalculationError: If calculation fails (e.g., peak too narrow).
        """
        if peak_idx < 0 or peak_idx >= len(data):
            raise PeakCalculationError(f"Peak index {peak_idx} is out of bounds for data of length {len(data)}.")
        if peak_height_from_props <= 0:
            logger.warning(f"Peak height at index {peak_idx} is non-positive ({peak_height_from_props}). Cannot calculate FWHM.")
            return 0.0

        half_max_level = peak_height_from_props / 2.0

        # Find initial bounds for linear interpolation
        left_idx = peak_idx
        while left_idx > 0 and data[left_idx - 1] > half_max_level:
            left_idx -= 1

        right_idx = peak_idx
        while right_idx < len(data) - 1 and data[right_idx + 1] > half_max_level:
            right_idx += 1

        if left_idx == peak_idx and right_idx == peak_idx:
            # Peak is too narrow, or half-height level is not crossed
            logger.warning(f"Peak at index {peak_idx} is too narrow or flat for FWHM calculation. Returning 0.")
            return 0.0

        # Linear interpolation to find more precise crossing points
        left_time = self.time_data[left_idx]
        left_uv = data[left_idx]
        if left_idx > 0:
            next_left_time = self.time_data[left_idx - 1]
            next_left_uv = data[left_idx - 1]
            if left_uv != next_left_uv: # Avoid division by zero
                frac_left = (half_max_level - next_left_uv) / (left_uv - next_left_uv)
                interpolated_left_time = next_left_time + frac_left * self.data_interval
            else: # Flat region or exact match
                interpolated_left_time = left_time
        else:
            interpolated_left_time = left_time # At the very start of the data

        right_time = self.time_data[right_idx]
        right_uv = data[right_idx]
        if right_idx < len(data) - 1:
            prev_right_time = self.time_data[right_idx + 1]
            prev_right_uv = data[right_idx + 1]
            if right_uv != prev_right_uv: # Avoid division by zero
                frac_right = (half_max_level - prev_right_uv) / (right_uv - prev_right_uv)
                interpolated_right_time = prev_right_time + frac_right * self.data_interval
            else: # Flat region or exact match
                interpolated_right_time = right_time
        else:
            interpolated_right_time = right_time # At the very end of the data

        fwhm = interpolated_right_time - interpolated_left_time
        return max(0.0, fwhm) # Ensure FWHM is not negative due to floating point
    
    def get_peaks(self) -> List[ChromatographicPeak]:
        """Returns the list of detected and analyzed peaks."""
        return self.peaks

    def get_baseline(self) -> Optional[np.ndarray]:
        """Returns the estimated baseline data."""
        return self.baseline
    
    def get_corrected_uv_data(self) -> Optional[np.ndarray]:
        """Returns the UV data after baseline correction."""
        if self.uv_data is not None and self.baseline is not None:
            return self.uv_data - self.baseline
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # Create some dummy chromatographic data
    # Simulate a baseline with some noise
    time_points = np.linspace(0, 100, 1000) # 1000 points over 100 minutes
    baseline_noise = np.sin(time_points / 20) * 0.1 + np.random.normal(0, 0.05, 1000) + 0.5
    uv_response = baseline_noise.copy()

    # Add some peaks
    # Peak 1
    uv_response += 5 * np.exp(-((time_points - 20)**2) / (2 * 5**2)) # Height 5, center 20, width 5
    # Peak 2 (smaller, closer)
    uv_response += 2 * np.exp(-((time_points - 28)**2) / (2 * 2**2)) # Height 2, center 28, width 2
    # Peak 3 (large, broad)
    uv_response += 7 * np.exp(-((time_points - 60)**2) / (2 * 8**2)) # Height 7, center 60, width 8
    # Peak 4 (small, near end)
    uv_response += 1.5 * np.exp(-((time_points - 85)**2) / (2 * 3**2)) # Height 1.5, center 85, width 3

    # Initialize PeakAnalysis with custom configuration
    custom_config = PeakAnalysisConfig(
        min_peak_height=0.5,
        min_peak_distance=5, # allow closer peaks
        min_peak_prominence=0.5,
        peak_integration_window_points=30, # larger window for area
        retention_time_tolerance=0.8 # larger tolerance for matching
    )
    analysis = PeakAnalysis(config=custom_config)

    # Set standards
    standards_list = [
        ChromatographyStandard(name="Protein A", retention_time=20.0, retention_time_tolerance=1.0),
        ChromatographyStandard(name="Impurity X", retention_time=28.5), # Uses default tolerance
        ChromatographyStandard(name="Protein B", retention_time=60.0),
        ChromatographyStandard(name="Degradant Y", retention_time=84.5)
    ]
    try:
        analysis.set_standards(standards_list)
    except TypeError as e:
        logger.error(f"Error setting standards: {e}")

    # Detect and analyze peaks
    try:
        analysis.detect_peaks(time_points.tolist(), uv_response.tolist())
        detected_peaks = analysis.get_peaks()

        print(f"\n--- Detected Peaks ({len(detected_peaks)}) ---")
        if detected_peaks:
            for i, peak in enumerate(detected_peaks):
                print(f"Peak {i+1}:")
                print(f"  Retention Time: {peak.retention_time:.2f} min")
                print(f"  Height (corrected): {peak.height:.2f}")
                print(f"  Area (corrected): {peak.area:.2f}")
                print(f"  FWHM: {peak.width_at_half_height:.2f} min")
                print(f"  Prominence: {peak.prominence:.2f}")
                print(f"  Start/End Time: {peak.start_time:.2f} - {peak.end_time:.2f} min")
                print(f"  Standard Match: {peak.standard_match if peak.standard_match else 'None'}")
                print("-" * 30)
        else:
            print("No peaks detected.")

    except InvalidDataError as e:
        logger.error(f"Data error during peak detection: {e}")
    except PeakCalculationError as e:
        logger.error(f"Calculation error during peak analysis: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")

    # Example of invalid data
    print("\n--- Testing Invalid Data ---")
    try:
        analysis.detect_peaks([1, 2], [1, 2, 3]) # Mismatched lengths
    except InvalidDataError as e:
        print(f"Caught expected error for mismatched data lengths: {e}")

    try:
        analysis.detect_peaks([], []) # Empty data
        print(f"Caught expected warning for empty data. Peaks: {analysis.get_peaks()}")
    except Exception as e:
        print(f"Unexpected error for empty data: {e}")

    # Example of using a template for standards (conceptual, not implemented in this class directly)
    # A higher-level "ChromatographySystem" class might manage standards and methods.
    # For now, it's just a direct list.

    # Access baseline and corrected UV data if needed for plotting
    # import matplotlib.pyplot as plt
    # if analysis.uv_data is not None and analysis.baseline is not None:
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(analysis.time_data, uv_response, label='Original UV Data', alpha=0.7)
    #     plt.plot(analysis.time_data, analysis.baseline, label='Estimated Baseline', linestyle='--')
    #     plt.plot(analysis.time_data, analysis.get_corrected_uv_data(), label='Baseline Corrected UV Data', alpha=0.9)
    #     for peak in detected_peaks:
    #         plt.vlines(peak.retention_time, 0, peak.height + 0.1, color='r', linestyle=':', alpha=0.7, label='Peak Apex' if peak == detected_peaks[0] else "")
    #         # Illustrate FWHM
    #         # plt.hlines(peak.height/2, peak.retention_time - peak.width_at_half_height/2, peak.retention_time + peak.width_at_half_height/2, color='g', linestyle='-.', alpha=0.7)
    #         plt.plot([peak.start_time, peak.end_time], [analysis.get_corrected_uv_data()[peak.index], analysis.get_corrected_uv_data()[peak.index]], 'k--', lw=1) # Simple base line for peak
    #     plt.xlabel('Time (min)')
    #     plt.ylabel('UV Absorbance')
    #     plt.title('Chromatogram with Detected Peaks and Baseline')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()