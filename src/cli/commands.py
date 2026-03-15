"""
Command handlers for phenomobile CLI.
Implements the actual command execution logic.
"""

import os
import sys
from typing import Dict

from utils.logger import get_logger, log_step
from utils.config import ConfigManager
from cli.workflows import get_workflow
from cli.workflows import DatasetCreationWorkflow


def handle_create_command(args: Dict, config: ConfigManager) -> None:
    """
    Handle dataset creation command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration manager instance
    """
    logger = get_logger('create_command')
    
    try:
        log_step("Starting dataset creation")
        
        # Create workflow
        workflow = get_workflow('create', config)
        
        # Execute dataset creation
        workflow.create_datasets(args)
        
        logger.info("Dataset creation command completed successfully")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        sys.exit(1)


def handle_merge_command(args: Dict, config: ConfigManager) -> None:
    """
    Handle dataset merge command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration manager instance
    """
    logger = get_logger('merge_command')
    
    try:
        log_step("Starting dataset merge")
        
        # Validate dataset paths
        params_dataset = config.get_dataset_path(args.get('params_dataset'))
        ref_dataset = config.get_dataset_path(args.get('ref_dataset'))
        
        if not os.path.exists(params_dataset):
            raise FileNotFoundError(f"parameters dataset not found: {params_dataset}")
        
        if not os.path.exists(ref_dataset):
            raise FileNotFoundError(f"Reference dataset not found: {ref_dataset}")
        
        # Create workflow
        workflow = get_workflow('merge', config)
        
        # Execute dataset merge
        workflow.merge_datasets(args)
        
        logger.info("Dataset merge command completed successfully")
        
    except Exception as e:
        logger.error(f"Dataset merge failed: {e}")
        sys.exit(1)


def handle_train_command(args: Dict, config: ConfigManager) -> None:
    """
    Handle ML training command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration manager instance
    """
    logger = get_logger('train_command')
    
    try:
        log_step("Starting ML training")
        
        # Validate dataset path
        dataset_path = config.get_dataset_path(args.get('dataset'))
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
        
        # Create workflow
        workflow = get_workflow('train', config)
        
        # Execute ML training
        results = workflow.train_models(args)
        
        logger.info("ML training command completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        sys.exit(1)


def handle_help_command(args: Dict, config: ConfigManager) -> None:
    """
    Handle help command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration manager instance
    """
    from utils.arguments import print_help
    print_help()


def handle_plot_command(args: Dict, config: ConfigManager) -> None:
    """
    Handle plot generation command.
    
    Args:
        args: Parsed command line arguments
        config: Configuration manager instance
    """
    logger = get_logger('plot_command')
    
    try:
        log_step("Starting plot generation")
        
        # Create workflow
        workflow = get_workflow('plot', config)
        
        # Generate plots
        workflow.generate_plots(args)
        
        log_step("Plot generation completed successfully")
        
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        sys.exit(1)


# Command registry
COMMAND_HANDLERS = {
    'create': handle_create_command,
    'merge': handle_merge_command,
    'train': handle_train_command,
    'plot': handle_plot_command,
    'help': handle_help_command
}


def execute_command(command: str, args: Dict, config: ConfigManager) -> None:
    """
    Execute the specified command.
    
    Args:
        command: Command name
        args: Parsed command line arguments
        config: Configuration manager instance
    """
    if command not in COMMAND_HANDLERS:
        logger = get_logger()
        logger.error(f"Unknown command: {command}")
        logger.info(f"Available commands: {list(COMMAND_HANDLERS.keys())}")
        sys.exit(1)
    
    handler = COMMAND_HANDLERS[command]
    return handler(args, config)
