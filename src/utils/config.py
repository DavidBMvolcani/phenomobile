"""
Configuration management utilities for phenomobile project.
Handles environment file loading and configuration validation.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv


class ConfigManager:
    """Manages configuration from main config, project config, and environment files."""
    
    def __init__(self, env_file: str = '.env', project_config: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            env_file: Path to environment file
            project_config: Path to project-specific config file (optional)
        """

        # 1. CALCULATE PROJECT ROOT
        # This points to the 'phenomobile' directory regardless of where you run the script
        # __file__ is src/utils/config.py -> .parent is src/utils -> .parent is src -> .parent is root
        self.root_path = Path(__file__).resolve().parent.parent.parent

        self.env_file = env_file
        self.project_config = project_config 
        self.config = {}
        self.load_main_config()
        self.load_project_config()
        self.load_environment()
        self.setup_paths()
    
    def load_main_config(self) -> None:
        """Load main configuration using absolute path."""
        # Fix: Use self.root_path instead of relative Path('config/...')
        main_config_path = self.root_path / 'config' / 'main_config.json'
        
        if not main_config_path.exists():
            raise FileNotFoundError(f"Main config file not found: {main_config_path}")
        
        with open(main_config_path, 'r') as f:
            main_config = json.load(f)
        
        # Merge main config
        self._merge_config(main_config)
    
    def load_project_config(self) -> None:
        """Load project-specific configuration using absolute path."""
        # Only load project config if specified
        if not self.project_config:
            return
            
        # Fix: Use self.root_path
        project_config_path = self.root_path / 'config' / self.project_config
        
        if not project_config_path.exists():
            raise FileNotFoundError(f"Project config file not found: {project_config_path}")
        
        with open(project_config_path, 'r') as f:
            project_config = json.load(f)
        
        # Merge project config
        self._merge_config(project_config)
    
    def _merge_config(self, new_config: Dict) -> None:
        """Merge new configuration into existing config."""
        def deep_merge(base_dict, new_dict):
            for key, value in new_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_merge(self.config, new_config)
    
    def load_environment(self) -> None:
        """Load environment variables from .env file."""
        if not self.env_file:
            return
            
        env_path = self.root_path / self.env_file
        
        if not env_path.exists():
            print(f"Warning: Environment file not found at {env_path}")
            print("Continuing without environment variables - no server connection available")
            return
        
        load_dotenv(env_path)
        
        # Check for required environment variables
        required_vars = ['dataset_folder', 'home_directory_name']
        missing_vars = []
        
        for var in required_vars:
            if os.environ.get(var) is None:
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    def setup_paths(self) -> None:
        """Setup project paths relative to the calculated project root."""
        # We use self.root_path as our anchor
        self.config['home_path'] = str(self.root_path)
        
        # Get dataset folder with proper fallback
        experiment_settings = self.config.get('experiment_settings') or {}
        dataset_folder = experiment_settings.get('dataset_folder', 'datasets')
        self.config['datasets_path'] = self.root_path / dataset_folder
        
        # Get download folder with proper fallback
        paths_config = self.config.get('paths') or {}
        download_folder = paths_config.get('download_folder', 'images')
        self.config['download_path'] = str(self.root_path / download_folder)
        
        # Outputs and Src are also relative to root
        self.config['outputs_path'] = str(self.root_path / 'outputs')
        self.config['src_path'] = str(self.root_path / 'src')
        
        if self.config['src_path'] not in sys.path:
            sys.path.insert(0, self.config['src_path'])
    
    def get(self, key: str, default: Optional = None) -> Optional:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_flir_path(self) -> str:
        """Get FLIR image extractor path."""
        flir_path = self.get('FlirImageExtractor_path')
        if flir_path not in sys.path:
            sys.path.append(flir_path)
        return flir_path
    
    def get_config_path(self, config_name: Optional[str] = None) -> str:
        """
        Get configuration file path.
        
        Args:
            config_name: Configuration file name (optional)
            
        Returns:
            Full path to configuration file
        """
        if config_name is None:
            config_name = self.project_config
        
        config_folder = 'config'
        return str(self.root_path / config_folder / config_name)
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """
        Get full path to dataset file.
        
        Args:
            dataset_name: Dataset file name
            
        Returns:
            Full path to dataset file
        """
        datasets_path = self.config['datasets_path']
        if isinstance(datasets_path, str):
            return str(Path(datasets_path) / dataset_name)
        else:
            return str(datasets_path / dataset_name)
    
    def ensure_output_dir(self) -> str:
        """Ensure output directory exists and return path."""
        outputs_path = Path(self.config['outputs_path'])
        outputs_path.mkdir(exist_ok=True)
        return str(outputs_path)
    
    def update_from_args(self, args: Dict) -> None:
        """
        Update configuration with command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Override env file if specified
        if args.get('env_file') and args['env_file'] != '.env':
            self.env_file = args['env_file']
            self.load_environment()
        
        # Override project config if specified
        if args.get('config'):
            self.project_config = args['config']
            self.load_project_config()
            self.setup_paths()  # Re-setup paths in case project config changed them
        
        # Store command-specific arguments
        for key, value in args.items():
            if value is not None and key != 'command':
                self.config[f'arg_{key}'] = value


def load_config(env_file: str = '.env', project_config: Optional[str] = None) -> ConfigManager:
    """
    Convenience function to load configuration.
    
    Args:
        env_file: Path to environment file
        project_config: Path to project-specific config file (optional)
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(env_file, project_config)
