#!/usr/bin/env python3
"""
Phenomobile - Plant phenotyping data processing pipeline.

A command-line interface for processing hyperspectral, thermal, and RGB images
to extract plant phenotyping features and train machine learning models.

Usage:
    python main.py create --hs --th --ndi-tuple "583.85,507.56"
    python main.py merge --hp-dataset data.csv --ref-dataset ref.csv --project lettuce
    python main.py train --dataset complete.csv --features "ndvi,ndi,anthocyanin" --target anthocyanin
"""

import sys
import traceback
from typing import Dict

# Import utilities
from utils.arguments import parse_and_validate
from utils.config import ConfigManager
from utils.logger import setup_logger, get_logger
from cli.commands import execute_command


def main():
    """Main entry point for the phenomobile CLI."""
    
    # Parse command line arguments
    try:
        args = parse_and_validate()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)
    
    # Setup logging
    verbose = args.get('verbose', False)
    env_file = args.get('env_file', '.env')
    
    try:
        logger = setup_logger(
            name='phenomobile',
            level='INFO' if not verbose else 'DEBUG',
            verbose=verbose
        )
    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1)
    
    # Load configuration
    try:
        project_config = args.get('config')
        config = ConfigManager(env_file, project_config)
        config.update_from_args(args)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Execute command
    command = args.get('command')
    
    try:
        logger.info(f"Starting phenomobile with command: {command}")
        logger.info(f"Arguments: {args}")
        
        result = execute_command(command, args, config)
        
        logger.info(f"Command '{command}' completed successfully")
        
        # Return result if available (for testing)
        return result
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"Command '{command}' failed: {e}")
        
        if verbose:
            logger.debug("Full traceback:")
            logger.debug(traceback.format_exc())
        
        sys.exit(1)


if __name__ == '__main__':
    main()
