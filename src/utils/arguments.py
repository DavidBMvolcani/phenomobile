"""
Argument parsing utilities for phenomobile CLI.
Handles command-line argument parsing and validation.
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple


def create_base_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Phenomobile - Plant phenotyping data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create --hs --th --ndi_tuple "583.85,507.56"
  %(prog)s merge --params_dataset data.csv --ref_dataset ref.csv --project lettuce
  %(prog)s train --dataset complete.csv --features "ndvi,ndi,anthocyanin" --target anthocyanin
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--env_file', 
        default='.env',
        help='Path to environment file (default: .env)'
    )
    
    parser.add_argument(
        '--config',
        default='anthocyanin_config.json',
        help='Path to project configuration file (default: anthocyanin_config.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Create subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )
    
    # Add subcommands
    _add_create_parser(subparsers)
    _add_merge_parser(subparsers)
    _add_train_parser(subparsers)
    _add_plot_parser(subparsers)
    
    return parser


def _add_create_parser(subparsers) -> None:
    """Add dataset creation subcommand parser."""
    create_parser = subparsers.add_parser(
        'create',
        help='Create datasets from raw data'
    )
    
    create_parser.add_argument(
        '--hs',
        action='store_true',
        help='Process hyperspectral images'
    )
    
    create_parser.add_argument(
        '--th',
        action='store_true', 
        help='Process thermal images'
    )
    
    create_parser.add_argument(
        '--rgb',
        action='store_true',
        help='Process RGB images'
    )
    
    create_parser.add_argument(
        '--ndi_tuple',
        type=str,
        help='NDI wavelength tuple as "wl1,wl2" (e.g., "583.85,507.56")'
    )
    
    create_parser.add_argument(
        '--create_ndi_table',
        action='store_true',
        help='Create NDI tables for hyperspectral images'
    )
    
    create_parser.add_argument(
        '--split_objects',
        action='store_true',
        help='Split images to objects using annotations'
    )


def _add_merge_parser(subparsers) -> None:
    """Add dataset merging subcommand parser."""
    merge_parser = subparsers.add_parser(
        'merge',
        help='Merge parameters and reference datasets'
    )
    
    merge_parser.add_argument(
        '--params_dataset',
        required=True,
        help='Path to parameters dataset CSV file'
    )
    
    merge_parser.add_argument(
        '--ref_dataset', 
        required=True,
        help='Path to reference dataset CSV file'
    )
   

def _add_train_parser(subparsers) -> None:
    """Add ML training subcommand parser."""
    train_parser = subparsers.add_parser(
        'train',
        help='Train and evaluate ML models'
    )
    
    train_parser.add_argument(
        '--dataset',
        required=True,
        help='Path to training dataset CSV file'
    )
    
    train_parser.add_argument(
        '--features',
        help='Comma-separated list of feature columns'
    )
    
    train_parser.add_argument(
        '--target',
        required=True,
        help='Target column for prediction'
    )
    
    train_parser.add_argument(
        '--task',
        choices=['regression', 'classification'],
        default='regression',
        help='ML task type (default: regression)'
    )
    train_parser.add_argument(
        '--compute_r2_score_for_ndi_tables',
        action='store_true',
        help='Compute R2 score for NDI tables'
    )
    
    train_parser.add_argument(
        '--filter',
        choices=['light'],
        help='Filter by light conditions (automatically enabled for Anthocyanin project)'
    )
    
    train_parser.add_argument(
        '--model',
        choices=['linear regression', 'ridge', 'random forest', 'xgboost'],
        help='Specific model to use (default: all models)'
    )
    
    train_parser.add_argument(
        '--show_plots',
        action='store_true',
        help='Display plots interactively (default: save only)'
    )
    
    train_parser.add_argument(
        '--plot_separate',
        action='store_true',
        help='Create separate plots for each condition'
    )


def _add_plot_parser(subparsers) -> None:
    """Add plotting subcommand parser."""
    plot_parser = subparsers.add_parser(
        'plot',
        help='Generate plots from datasets'
    )
    
    plot_parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset file'
    )
    
    plot_parser.add_argument(
        '--type',
        type=str,
        choices=['regression', 'anthocyanin', 'r2-score-of-ndi'],
        required=True,
        help='Type of plot to generate'
    )
    
    plot_parser.add_argument(
        '--features',
        type=str,
        help='Comma-separated list of feature columns'
    )
    
    plot_parser.add_argument(
        '--target',
        type=str,
        help='Target column name'
    )
    
    
    plot_parser.add_argument(
        '--condition',
        type=str,
        help='Filter condition for plots'
    )
    
    plot_parser.add_argument(
        '--model',
        type=str,
        help='Model name for NDI plots'
    )
    
    plot_parser.add_argument(
        '--show_plots',
        action='store_true',
        help='Display plots interactively (default: save only)'
    )
    
    plot_parser.add_argument(
        '--plot_separate',
        action='store_true',
        help='Create separate plots for each condition'
    )


def parse_and_validate(args: Optional[List[str]] = None) -> Dict:
    """
    Parse and validate command line arguments.
    
    Args:
        args: List of argument strings (defaults to sys.argv)
        
    Returns:
        Dictionary with parsed arguments
        
    Raises:
        SystemExit: If arguments are invalid
    """
    parser = create_base_parser()
    
    if args is None:
        args = sys.argv[1:]
    
    parsed_args = parser.parse_args(args)
    
    # Convert to dictionary for easier handling
    args_dict = vars(parsed_args)
    
    # Validate arguments
    _validate_arguments(args_dict)
    
    return args_dict


def _validate_arguments(args_dict: Dict) -> None:
    """Validate parsed arguments."""
    
    # Validate create command
    if args_dict.get('command') == 'create':
        if not any([args_dict.get('hs'), args_dict.get('th'), args_dict.get('rgb')]):
            raise ValueError("At least one data type (--hs, --th, --rgb) must be specified for create command")
        
        if args_dict.get('ndi_tuple'):
            try:
                # Parse "wl1,wl2" format
                wavelengths = args_dict['ndi_tuple'].split(',')
                if len(wavelengths) != 2:
                    raise ValueError()
                float(wavelengths[0])
                float(wavelengths[1])
                # Convert to tuple
                args_dict['ndi_tuple'] = (float(wavelengths[0]), float(wavelengths[1]))
            except (ValueError, IndexError):
                raise ValueError("--ndi_tuple must be in format 'wl1,wl2' with numeric values")
    
    # Validate train command
    elif args_dict.get('command') == 'train':
        # there two options of training :one with given features 
        # and another with ndi_tables that store in files (csv or hdf5)
        # and their path cofigured in config file
        if not args_dict.get('features') and not args_dict.get('compute_r2_score_for_ndi_tables'):
            raise ValueError("At least one of --features or --compute_r2_score_for_ndi_tables must be specified for train command")

        if args_dict.get('features'):
            # Parse comma-separated features, handling parentheses and quotes
            features_str = args_dict['features']
            
            # Handle quoted features or features with parentheses
            if '"' in features_str or '(' in features_str:
                # Manual parsing for features with parentheses
                features = []
                current_feature = ''
                paren_count = 0
                in_quotes = False
                
                for char in features_str:
                    if char == '"' and not paren_count:
                        in_quotes = not in_quotes
                    elif char == '(' and not in_quotes:
                        paren_count += 1
                        current_feature += char
                    elif char == ')' and not in_quotes:
                        paren_count -= 1
                        current_feature += char
                    elif char == ',' and not paren_count and not in_quotes:
                        features.append(current_feature.strip())
                        current_feature = ''
                    else:
                        current_feature += char
                
                # Add the last feature
                if current_feature.strip():
                    features.append(current_feature.strip())
            else:
                # Simple case: just split by comma
                features = [f.strip() for f in features_str.split(',')]
            
            args_dict['features'] = features
        
        # Validate that target is in features
        if args_dict.get('target') and args_dict.get('features'):
            if args_dict['target'] not in args_dict['features']:
                raise ValueError(f"Target '{args_dict['target']}' must be included in features list")


def print_help() -> None:
    """Print help message."""
    parser = create_base_parser()
    parser.print_help()
