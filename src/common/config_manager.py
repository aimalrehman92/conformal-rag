import os
import yaml
import logging
import datetime
import json
import shutil
from pathlib import Path

class ConfigManager:
    """Utility class to manage configuration loading, saving and logging"""
    
    def __init__(self, config_path=None, path_config_path=None, dataset_config_path=None, run_id=None):
        """
        Initialize the ConfigManager with a config file path
        
        Args:
            config_path (str): Path to the YAML config file
            run_id (str): Optional identifier for the run
        """
        self.config = {}
        self.run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/{self.run_id}"
        
        if config_path:
            self.config = self.load_config(config_path)
        if path_config_path:
            self.path_config = self.load_config(path_config_path)
        if dataset_config_path:
            self.dataset_config = self.load_config(dataset_config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from a YAML file
        
        Args:
            config_path (str): Path to the YAML config file
            
        Returns:
            dict: The loaded configuration
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, output_path=None):
        """
        Save the current configuration to a YAML file
        
        Args:
            output_path (str): Path to save the config file, defaults to log directory
            
        Returns:
            str: Path to the saved config file
        """
        if output_path is None:
            os.makedirs(self.log_dir, exist_ok=True)
            output_path = os.path.join(self.log_dir, f"config_{self.run_id}.yaml")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        return output_path
    
    def setup_logging(self, log_level=logging.INFO):
        """
        Setup logging configuration
        
        Args:
            log_level: Logging level
            
        Returns:
            str: Path to the log file
        """
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, f"run_{self.run_id}.log")
        
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        # Disable httpx logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Log some initial information
        logging.info(f"Starting run with ID: {self.run_id}")
        logging.info(f"Log file: {log_file}")
        
        return log_file, self.run_id
    
    def log_config(self):
        """Log the important parts of the configuration"""
        if not self.config:
            logging.warning("No configuration loaded to log")
            return
            
        logging.info("=== Run Configuration ===")
        
        # Log dataset info
        if 'dataset' in self.config:
            logging.info(f"Dataset: {self.config['dataset']['name']}")
            logging.info(f"Query size: {self.config['dataset']['query_size']}")
        
        # Log index info
        if 'index' in self.config:
            logging.info(f"Embedding model: {self.config['index']['embedding_model']}")
            logging.info(f"Delete existing index: {self.config['index']['delete_existing']}")
            
        logging.info("========================")
    
    def update_config(self, updates):
        """
        Update the configuration with new values
        
        Args:
            updates (dict): Dictionary containing updates to apply
            
        Returns:
            dict: The updated configuration
        """
        # This is a simple implementation that only handles top-level keys
        for key, value in updates.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
        
        return self.config
    
    def copy_run_artifacts(self, result_dir):
        """
        Copy config and logs to a results directory for reproducibility
        
        Args:
            result_dir (str): Path to the results directory
            
        Returns:
            str: Path to the result run directory
        """
        result_run_dir = os.path.join(result_dir, "config")
        os.makedirs(result_run_dir, exist_ok=True)
        
        # Get the latest config and log files
        config_files = sorted(Path(self.log_dir).glob("config_*.yaml"))
        # log_files = sorted(Path(self.log_dir).glob("run_*.log"))
        
        if config_files:
            latest_config = str(config_files[-1])
            shutil.copy2(latest_config, os.path.join(result_run_dir, "config.yaml"))
        
        # if log_files:
        #     latest_log = str(log_files[-1])
        #     shutil.copy2(latest_log, os.path.join(result_run_dir, "run.log"))
        
        return result_run_dir