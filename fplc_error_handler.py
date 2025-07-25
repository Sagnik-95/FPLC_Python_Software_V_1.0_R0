from enum import Enum
from typing import List, Optional, Callable, Any, Dict
import logging
import time
import traceback
import hashlib
import json 
import os
from dataclasses import dataclass
from datetime import datetime

class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class FPLCError(Exception):
    def __init__(self, message: str, severity: ErrorSeverity, details: Optional[dict] = None):
        super().__init__(message)
        self.severity = severity
        self.details = details or {}
        self.timestamp = time.time()

class ErrorHandler:
    def __init__(self, log_file: str = "fplc_errors.log"):
        self.log_file = log_file
        self.error_callbacks: List[Callable] = []
        self.setup_logging()
        
    def setup_logging(self) -> None:
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def handle_error(self, error: FPLCError) -> None:
        error_info = {
            'message': str(error),
            'severity': error.severity.value,
            'details': error.details,
            'timestamp': error.timestamp,
            'traceback': traceback.format_exc()
        }
        
        logging.error(f"FPLC Error: {error_info}")
        
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logging.error(f"Error in error callback: {e}")
                
    def add_error_callback(self, callback: Callable) -> None:
        self.error_callbacks.append(callback)
        
    def remove_error_callback(self, callback: Callable) -> None:
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)

class DataIntegrityManager:
    def __init__(self):
        self.checksums = {}
        
    def calculate_checksum(self, data: Any) -> str:
        """Calculates checksum for data integrity verification"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_data_integrity(self, data: Any, stored_checksum: str) -> bool:
        """Verifies data integrity using stored checksum"""
        current_checksum = self.calculate_checksum(data)
        return current_checksum == stored_checksum

@dataclass
class SystemConfig:
    com_port: str = "COM3"
    baud_rate: int = 9600
    max_pressure: float = 25.0
    min_pressure: float = 0.0
    max_flow_rate: float = 10.0
    sample_interval: int = 1000
    data_directory: str = "data"
    method_directory: str = "methods"
    log_directory: str = "logs"

@dataclass
class SystemConfig2:
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
            # ... (rest of from_json method) ...
            logging.warning(f"Configuration file not found at {config_path}. Using default system settings.")
            return cls()
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {config_path}. Using default system settings.")
            return cls()
        except TypeError as e:
            logging.error(f"Error in configuration data types from {config_path}: {e}. Using default system settings.")
            return cls()
    
    
class ConfigManager:
    def __init__(self, config_file: str = "system_config.json"):
        self.lims_config = {
            'base_url': 'https://lims.example.com',
            'api_key': 'your_api_key_here',
            'enabled': False
        }
        self.config_file = config_file
        self.config = SystemConfig()
        self.load_config()
        self.audit_directory = "audit_logs"
        self.secure_storage = "secure_storage"
        self.compliance_enabled = True
        self.password_expiry_days = 90
        self.session_timeout_minutes = 30
        
    def load_config(self) -> None:
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    setattr(self.config, key, value)
                    
    def save_config(self) -> None:
        config_dict = {
            'com_port': self.config.com_port,
            'baud_rate': self.config.baud_rate,
            'max_pressure': self.config.max_pressure,
            'min_pressure': self.config.min_pressure,
            'max_flow_rate': self.config.max_flow_rate,
            'sample_interval': self.config.sample_interval,
            'data_directory': self.config.data_directory,
            'method_directory': self.config.method_directory,
            'log_directory': self.config.log_directory
        }
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

class ComplianceManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.audit_trail = []
        self.electronic_signatures = {}
        self.current_user = None
        
    def log_audit_event(self, event_type: str, details: Dict, user: str) -> None:
        """Records an audit trail entry with digital signature"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user': user,
            'details': details,
            'signature': self._generate_signature(details)
        }
        self.audit_trail.append(audit_entry)
        self._save_audit_trail()
    
    def _generate_signature(self, data: Dict) -> str:
        """Generates a digital signature for data integrity"""
        data_string = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def verify_signature(self, user: str, password: str, reason: str) -> bool:
        """Verifies electronic signature"""
        if not self._verify_credentials(user, password):
            return False
        
        signature = {
            'user': user,
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        }
        self.electronic_signatures[signature['timestamp']] = signature
        return True
    
    def _save_audit_trail(self) -> None:
        """Saves audit trail to secure storage"""
        audit_file = os.path.join(self.config.config.audit_directory, 
                                 f'audit_trail_{datetime.now().strftime("%Y%m")}.json')
        with open(audit_file, 'w') as f:
            json.dump(self.audit_trail, f, indent=4)



