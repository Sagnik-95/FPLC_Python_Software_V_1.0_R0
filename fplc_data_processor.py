import numpy as np
from scipy import signal
from typing import List, Tuple
import pandas as pd
from datetime import datetime
import json
import csv
import logging
from PyQt6.QtWidgets import QMessageBox # Assuming this is available from FPLC_15.py context

# Set up a basic logger for this module (if not already configured globally)
try:
    logger = logging.getLogger('DataAnalysisModule')
    if not logger.handlers: # Prevent adding handlers multiple times if module is reloaded
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger('DataAnalysisModule')
except Exception as e:
    print(f"Error setting up logger in combined_data_analysis.py: {e}")
    logger = None # Fallback if logging setup fails


class DataProcessor:
    """
    Handles core signal processing operations for chromatography data,
    including smoothing, baseline correction, peak detection, and integration.
    """
    def __init__(self, peak_threshold: float = 0.1, min_peak_width: int = 10):
        """
        Initializes the DataProcessor with parameters for peak detection.

        Args:
            peak_threshold (float): Minimum height of a peak to be considered.
            min_peak_width (int): Minimum width of a peak in data points.
        """
        self.baseline_window = 50 # Default window size for baseline calculation (if needed by baseline_als)
        self.peak_threshold = peak_threshold
        self.min_peak_width = min_peak_width
        if logger:
            logger.info(f"DataProcessor initialized with peak_threshold={self.peak_threshold}, min_peak_width={self.min_peak_width}")

    def smooth_data(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Applies a Savitzky-Golay filter to smooth the data.

        Args:
            data (np.ndarray): The input data array to be smoothed.
            window_size (int): The length of the filter window (must be an odd integer).

        Returns:
            np.ndarray: The smoothed data array.
        """
        # Ensure window_size is odd and less than or equal to data length
        if window_size % 2 == 0:
            window_size += 1 # Make it odd
        if window_size > len(data):
            window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
            if window_size < 1: # Handle very short data
                return data

        try:
            return signal.savgol_filter(data, window_size, 2) # 2nd order polynomial
        except ValueError as e:
            if logger:
                logger.error(f"Error smoothing data: {e}. Data length: {len(data)}, window_size: {window_size}")
            # Fallback for small data sets where window_size might be too large
            if len(data) < 3: # Savgol filter requires at least 3 points for order 2
                return data
            # Try with a smaller window if original fails
            return signal.savgol_filter(data, min(len(data) - 1 if len(data) % 2 == 0 else len(data), 3), 2)


    def detect_peaks(self, data: np.ndarray) -> List[Tuple[int, float]]:
        """
        Detects peaks in the given data array.

        Args:
            data (np.ndarray): The input data array (e.g., UV absorbance).

        Returns:
            List[Tuple[int, float]]: A list of tuples, where each tuple contains
                                      (peak_index, peak_height).
        """
        peaks, properties = signal.find_peaks(
            data,
            height=self.peak_threshold,
            width=self.min_peak_width
        )
        if logger:
            logger.debug(f"Detected {len(peaks)} peaks with threshold {self.peak_threshold} and width {self.min_peak_width}.")
        return list(zip(peaks, properties['peak_heights']))

    def calculate_baseline(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates the baseline of the data using asymmetric least squares.
        This function requires the 'baseline_als' function, which is not
        standard in scipy.signal. It's often found in external libraries
        or implemented manually. For demonstration, a placeholder or
        a simple moving average could be used if baseline_als is unavailable.
        For now, let's assume `baseline_als` is available or implement a simple one.

        Args:
            data (np.ndarray): The input data array.

        Returns:
            np.ndarray: The calculated baseline.
        """
        # Placeholder for baseline_als if it's not imported/available.
        # A full implementation of baseline_als is complex.
        # For simplicity, let's use a very basic rolling minimum or a simple average
        # if scipy.signal.baseline_als is truly missing.
        # The original code implies signal.baseline_als exists, let's assume it does for now.
        # If it doesn't, you'd need to provide an implementation or use a different method.

        # A common simple baseline: rolling minimum or median filter
        if len(data) < self.baseline_window:
            if logger:
                logger.warning(f"Data length ({len(data)}) too short for baseline_window ({self.baseline_window}). Returning zeros for baseline.")
            return np.zeros_like(data)

        # Using a simple rolling minimum as a fallback if baseline_als is not defined
        # This is a very basic approximation.
        try:
            # Attempt to use the original baseline_als if it's somehow available
            # If not, this will likely fail, and we'll need a robust alternative.
            # Assuming a custom baseline_als or similar function is provided elsewhere
            # or that the user has scipy-baseline installed.
            # For a truly self-contained example without external libs beyond numpy/scipy:
            # A simple moving average or median filter can serve as a basic baseline.
            # Let's use a median filter as a more robust simple baseline.
            from scipy.ndimage import median_filter
            baseline = median_filter(data, size=self.baseline_window)
            if logger:
                logger.debug("Calculated baseline using median filter.")
            return baseline
        except ImportError:
            if logger:
                logger.warning("scipy.ndimage.median_filter not found. Falling back to simpler baseline (e.g., zeros).")
            return np.zeros_like(data) # Fallback if median_filter is also not available/importable
        except Exception as e:
            if logger:
                logger.error(f"Error in baseline calculation: {e}. Returning zeros for baseline.")
            return np.zeros_like(data)


    def integrate_peak(self, data: np.ndarray, start_idx: int, end_idx: int) -> float:
        """
        Integrates the area under a peak using the trapezoidal rule.

        Args:
            data (np.ndarray): The data array containing the peak.
            start_idx (int): The starting index of the peak region.
            end_idx (int): The ending index of the peak region.

        Returns:
            float: The calculated area under the peak.
        """
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(data), end_idx)
        if start_idx >= end_idx:
            return 0.0 # No valid range to integrate

        area = np.trapz(data[start_idx:end_idx])
        if logger:
            logger.debug(f"Integrated peak from {start_idx} to {end_idx}, area: {area:.2f}")
        return area

    def analyze_chromatogram(self, time: np.ndarray, signal: np.ndarray) -> pd.DataFrame:
        """
        Performs a full chromatogram analysis: smoothing, baseline correction,
        peak detection, and integration.

        Args:
            time (np.ndarray): Array of time points corresponding to the signal.
            signal (np.ndarray): Array of signal values (e.g., UV absorbance).

        Returns:
            pd.DataFrame: A DataFrame containing analysis results for each detected peak,
                          including retention_time, height, area, and width.
        """
        if len(time) != len(signal) or len(time) == 0:
            if logger:
                logger.warning("Time and signal data lengths mismatch or are empty. Returning empty DataFrame.")
            return pd.DataFrame()

        smoothed = self.smooth_data(signal)
        baseline = self.calculate_baseline(smoothed)
        corrected = smoothed - baseline # Baseline-corrected signal

        peaks = self.detect_peaks(corrected) # Detect peaks on the corrected signal

        results = []
        for peak_idx, height in peaks:
            # Determine peak width for integration
            # A more sophisticated peak width determination (e.g., FWHM) could be used.
            # For simplicity, using a window around the detected peak index.
            # The min_peak_width from DataProcessor is used to define a region around the peak.
            peak_start = max(0, peak_idx - self.min_peak_width // 2)
            peak_end = min(len(signal), peak_idx + self.min_peak_width // 2 + 1) # +1 for slicing

            area = self.integrate_peak(corrected, peak_start, peak_end)
            
            # Calculate actual width based on the integration window or FWHM if implemented
            actual_width = (peak_end - peak_start) * (time[1] - time[0]) if len(time) > 1 else 0.0

            results.append({
                'Peak Index': peak_idx, # Keep index for potential plotting
                'Retention Time (min)': time[peak_idx],
                'Height (mAU)': height,
                'Area (mAU*min)': area,
                'Width (min)': actual_width
            })
        
        if logger:
            logger.info(f"Analyzed chromatogram, found {len(results)} peaks.")
        return pd.DataFrame(results)


class DataAnalysis:
    """
    Handles advanced data analysis and reporting, orchestrating the use of DataProcessor
    and managing raw/processed data, standards, and report generation.
    """
    def __init__(self):
        self.time_data = []
        self.uv_data = []
        self.pressure_data = []
        self.conductivity_data = []
        self.flow_data = []
        self.buffer_a_data = []
        self.buffer_b_data = []
        
        # Instantiate DataProcessor
        self.data_processor = DataProcessor(peak_threshold=10, min_peak_width=5) # Adjust these as needed
        
        self.standards = [] # List of standard dictionaries: [{'name': 'Protein A', 'retention_time': 10.5}]
        self.analyzed_peaks_df = pd.DataFrame() # Stores DataFrame from DataProcessor.analyze_chromatogram
        self.logger = logging.getLogger('DataAnalysis')
        self.logger.info("Data Analysis initialized.")

    def add_data_point(self, time_point, uv, pressure, conductivity,
                       flow_rate, buffer_a, buffer_b):
        """Adds a new data point to the analysis storage."""
        self.time_data.append(time_point)
        self.uv_data.append(uv)
        self.pressure_data.append(pressure)
        self.conductivity_data.append(conductivity)
        self.flow_data.append(flow_rate)
        self.buffer_a_data.append(buffer_a)
        self.buffer_b_data.append(buffer_b)

    def load_standards(self, filename):
        """Loads chromatography standards from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.standards = data.get('standards', [])
            self.logger.info(f"Standards loaded successfully from {filename}.")
        except FileNotFoundError:
            self.logger.error(f"Standards file not found: {filename}")
            QMessageBox.critical(None, "File Error", f"Standards file not found: {filename}")
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON from standards file: {filename}")
            QMessageBox.critical(None, "File Error", f"Invalid JSON in standards file: {filename}")
        except Exception as e:
            self.logger.error(f"Error loading standards: {str(e)}")
            QMessageBox.critical(None, "Error", f"Error loading standards: {str(e)}")

    def match_peak_to_standard(self, peak_time):
        """Matches detected peak to known standards based on retention time."""
        for standard in self.standards:
            if abs(standard['retention_time'] - peak_time) < 0.5:  # Allow a tolerance
                return standard['name']
        return None

    def perform_full_chromatogram_analysis(self) -> pd.DataFrame:
        """
        Performs the complete chromatogram analysis using DataProcessor
        and then matches detected peaks to loaded standards.
        """
        if not self.time_data or not self.uv_data:
            self.logger.warning("No data available for chromatogram analysis.")
            self.analyzed_peaks_df = pd.DataFrame()
            return self.analyzed_peaks_df

        time_np = np.array(self.time_data)
        uv_np = np.array(self.uv_data)

        # Use DataProcessor to analyze the chromatogram
        self.analyzed_peaks_df = self.data_processor.analyze_chromatogram(time_np, uv_np)

        # Add standard matching to the DataFrame
        if not self.analyzed_peaks_df.empty and self.standards:
            self.analyzed_peaks_df['Standard Match'] = self.analyzed_peaks_df['Retention Time (min)'].apply(
                self.match_peak_to_standard
            )
            self.logger.info("Peaks matched to standards.")
        else:
            self.analyzed_peaks_df['Standard Match'] = "N/A" # Add column even if no standards or peaks

        return self.analyzed_peaks_df

    def export_data(self, filename):
        """Exports all collected raw data to a CSV file."""
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Time (min)', 'UV (mAU)', 'Pressure (MPa)',
                    'Conductivity (mS/cm)', 'Flow Rate (mL/min)',
                    'Buffer A (%)', 'Buffer B (%)'
                ])

                for i in range(len(self.time_data)):
                    writer.writerow([
                        f"{self.time_data[i]:.2f}",
                        f"{self.uv_data[i]:.2f}",
                        f"{self.pressure_data[i]:.3f}",
                        f"{self.conductivity_data[i]:.2f}",
                        f"{self.flow_data[i]:.2f}",
                        f"{self.buffer_a_data[i]:.1f}",
                        f"{self.buffer_b_data[i]:.1f}"
                    ])
            self.logger.info(f"Raw data exported to {filename}.")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting raw data: {str(e)}")
            return False

    def export_peaks(self, filename):
        """Exports detected peaks information to a CSV file from the analyzed DataFrame."""
        if self.analyzed_peaks_df.empty:
            self.logger.warning("No analyzed peaks to export.")
            return False
        try:
            # Ensure the order of columns for consistency with previous export format
            export_df = self.analyzed_peaks_df.copy()
            # Add a 'Peak Number' column if not already present
            if 'Peak Number' not in export_df.columns:
                export_df.insert(0, 'Peak Number', range(1, len(export_df) + 1))

            # Rename columns to match the desired CSV header
            export_df.rename(columns={
                'Retention Time (min)': 'Retention Time (min)',
                'Height (mAU)': 'Height (mAU)',
                'Area (mAU*min)': 'Area (mAU*min)',
                'Width (min)': 'Width (min)',
                'Standard Match': 'Standard Match'
            }, inplace=True)

            # Select and reorder columns for export
            columns_to_export = [
                'Peak Number', 'Retention Time (min)', 'Height (mAU)',
                'Area (mAU*min)', 'Width (min)', 'Standard Match'
            ]
            
            # Fill NaN in 'Standard Match' with "Unknown" for export
            export_df['Standard Match'] = export_df['Standard Match'].fillna("Unknown")

            export_df[columns_to_export].to_csv(filename, index=False, float_format='%.2f')
            self.logger.info(f"Peak data exported to {filename}.")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting peak data: {str(e)}")
            return False

    def calculate_statistics(self):
        """Calculates basic run statistics."""
        stats = {
            'run_duration': self.time_data[-1] if self.time_data else 0,
            'max_uv': max(self.uv_data) if self.uv_data else 0,
            'max_pressure': max(self.pressure_data) if self.pressure_data else 0,
            'average_conductivity': np.mean(self.conductivity_data) if self.conductivity_data else 0
        }
        return stats

    def generate_report(self, method, run_info, filename):
        """Generates a comprehensive run report in HTML format."""
        try:
            stats = self.calculate_statistics()
            # Use the analyzed_peaks_df for the report
            peaks_for_report = self.analyzed_peaks_df.to_dict(orient='records')

            with open(filename, 'w') as f:
                f.write(f"""
                <html>
                <head>
                    <title>Chromatography Run Report - {run_info.get('run_id', 'N/A')}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                        h1, h2 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        .section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
                        .pass {{ color: green; font-weight: bold; }}
                        .fail {{ color: red; font-weight: bold; }}
                    </style>
                </head>
                <body>
                    <h1>Chromatography Run Report</h1>

                    <div class="section">
                        <h2>Run Information</h2>
                        <table>
                            <tr><td>Run ID</td><td>{run_info.get('run_id', 'N/A')}</td></tr>
                            <tr><td>Run Date</td><td>{run_info.get('start_time', datetime.now()).strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
                            <tr><td>Method Name</td><td>{method.name}</td></tr>
                            <tr><td>Author</td><td>{method.author}</td></tr>
                        </table>
                    </div>

                    <div class="section">
                        <h2>Method Details</h2>
                        <table>
                            <tr><td>Description</td><td>{method.description if method.description else 'N/A'}</td></tr>
                            <tr><td>Column Type</td><td>{method.column_type}</td></tr>
                            <tr><td>Column Volume</td><td>{method.column_volume:.1f} mL</td></tr>
                            <tr><td>Max Pressure</td><td>{method.max_pressure:.2f} MPa</td></tr>
                            <tr><td>Buffer A</td><td>{method.buffer_info['A']['name']} ({method.buffer_info['A']['composition']})</td></tr>
                            <tr><td>Buffer B</td><td>{method.buffer_info['B']['name']} ({method.buffer_info['B']['composition']})</td></tr>
                        </table>
                    </div>

                    <div class="section">
                        <h2>Run Statistics</h2>
                        <table>
                            <tr><td>Duration</td><td>{stats['run_duration']:.1f} min</td></tr>
                            <tr><td>Max UV</td><td>{stats['max_uv']:.1f} mAU</td></tr>
                            <tr><td>Max Pressure</td><td>{stats['max_pressure']:.2f} MPa</td></tr>
                            <tr><td>Average Conductivity</td>
                                <td>{stats['average_conductivity']:.1f} mS/cm</td></tr>
                        </table>
                    </div>

                    <div class="section">
                        <h2>Peak Analysis</h2>
                        <table>
                            <tr>
                                <th>Peak #</th>
                                <th>Retention Time (min)</th>
                                <th>Height (mAU)</th>
                                <th>Area (mAU*min)</th>
                                <th>Width (min)</th>
                                <th>Standard Match</th>
                            </tr>
                """)

                if not peaks_for_report:
                    f.write("<tr><td colspan='6'>No peaks detected.</td></tr>")
                else:
                    for i, peak in enumerate(peaks_for_report, 1):
                        f.write(f"""
                                <tr>
                                    <td>{i}</td>
                                    <td>{peak.get('Retention Time (min)', 0.0):.2f}</td>
                                    <td>{peak.get('Height (mAU)', 0.0):.1f}</td>
                                    <td>{peak.get('Area (mAU*min)', 0.0):.1f}</td>
                                    <td>{peak.get('Width (min)', 0.0):.2f}</td>
                                    <td>{peak.get('Standard Match', 'Unknown')}</td>
                                </tr>
                        """)

                f.write("""
                        </table>
                    </div>
                </body>
                </html>
                """)
            self.logger.info(f"Run report generated at {filename}.")
            return True
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return False
