import yaml
import logging
import sys
from pathlib import Path

def load_config(config_path='../config/config.yaml'):
    """Load the YAML configuration file

    Args:
        config_path (str, optional): _description_. Defaults to '../config/config.yaml'.
    Returns:
        dict: The configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config

def setup_logging(config):
    """Set up the logging configuration foor the application

    Args:
        config (_type_): _description_
    """
    log_config = config.get("logging", {})
    log_filepath = config.get("filepaths", {}).get("log_file", "logs/pipeline.log")
    
    # Create logs folder if not exist
    Path(log_filepath).parent.mkdir(parents=True, exist_ok=True)
    
    log_level = log_config.get("level", 'INFO')
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    
    # Take the original log
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Delete old handler to avoid duplicate
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Create handler to write log in console
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    
    # Create handler to write log in file
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # Format log
    formatter = logging.Formatter(log_format)
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    
    logging.info("Logging has been configured")
