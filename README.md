# FPLC_Software
This is a Python based FPLC software with in-built features. Please do use it and let me know your feedback. 
The FINAL_UPDATED_FPLC_WITH_PASSWORD_PROTECTION.py is a single shot run for the FPLC software interface. 
The FPLC_MODULAR_FINAL is a robust and much more efficient code with dependencies on other 16 code files given provided along with it. Please feel free to use it 


FPLC Control Software: README
This document provides an overview of the Fast Protein Liquid Chromatography (FPLC) Control Software.

Project Title
Fast Protein Liquid Chromatography (FPLC) Control Software

Author
Sagnik Mitra

License
IIT INDORE

Overview
This software is a comprehensive suite designed for managing and controlling protein purification workflows using FPLC systems. It integrates detailed (simulated) hardware communication protocols and incorporates a conceptual Model Predictive Control (MPC)-like logic for precise pressure and gradient control during chromatographic runs.

Key Features
User Interface (GUI): Built with PyQt6, providing an intuitive graphical interface for system control, monitoring, and data visualization.

Simulated Hardware Communication: Features simulated protocols for interacting with FPLC hardware components.

Method Editor: Allows users to create, load, and save FPLC experimental methods, defining various steps, durations, flow rates, and buffer compositions.

Real-time Chromatogram Plotting: Displays live UV absorbance and conductivity data during a run. It includes functionalities for auto-scaling, peak detection, and real-time data readouts (UV, pressure, conductivity, flow, buffer percentages, temperature, pH).

Fraction Collection: Offers enhanced control over fraction collection, supporting peak-based, time-based, and volume-based collection modes. It provides a table to track collected fractions.

Buffer Blending: Manages buffer recipes and simulates buffer blending calculations.

System State Management: Monitors and displays the current operational state of the FPLC system (e.g., IDLE, RUNNING, PAUSED).

User Management and Security: Features a login system with hashed password validation and role-based access control to ensure secure operation and restrict certain functionalities (e.g., diagnostics, log export) to administrators.

Error Handling and Logging: Implements a robust error handling system with different severity levels and comprehensive logging to a file and console for troubleshooting and auditing.

Data Integrity and Compliance: Includes components for managing data integrity and ensuring system compliance.

System Diagnostics and Maintenance Scheduling: Provides tools for running system diagnostics and scheduling maintenance tasks.

Column Qualification: Offers a dedicated interface and functionality for performing column qualification procedures.

Model Predictive Control (MPC) Logic: Incorporates conceptual MPC logic for advanced control of pressure and buffer gradients, aiming for optimal separation performance.

Data Export and Reporting: Allows exporting raw chromatogram data and peak analysis data to CSV, and generating comprehensive HTML reports of experimental runs.

Dependencies
The software relies on the following key Python libraries and custom modules:

sys

numpy

datetime

PyQt6 (QtWidgets, QtCore, QtGui) for the graphical user interface.

matplotlib (figure, gridspec, backend_qt5agg) for plotting chromatograms.

serial (for simulated serial communication)

json

csv

logging

hashlib

os

subprocess

shutil

random

enum

threading

time

collections

Custom FPLC-specific modules (e.g., fplc_system_state, fplc_method, fplc_serial_manager, fplc_controller, fplc_user_management, fplc_sample_manager, fplc_maintenance_scheduler, fplc_validation_rules, fplc_method_validator, fplc_data_processor, fplc_flow_diagram, fplc_column_qualification, fplc_peak_analysis, fplc_error_handler).

How to Run
Dependencies Installation: Ensure all required Python libraries listed above are installed.

Execution: The application can be launched by running the main Python script. A login dialog will appear first.

Login: Use the hardcoded credentials (username: "Sagnik", password: "password") to log in. After successful login, the main FPLC control system window will be displayed.

Project Structure
The code is modularized into several Python files (as indicated by the imports) to manage different aspects of the FPLC system, including UI components, hardware control, data processing, user management, and system configuration.
