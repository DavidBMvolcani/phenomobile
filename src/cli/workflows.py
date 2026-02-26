"""
Workflow orchestration for phenomobile CLI.
High-level workflows that wrap existing classes with CLI-friendly interfaces.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import existing modules
from core.dataset_creation import dataset_creation
from datasets.lettuce_dataset import lettuce_dataset
from ml.training import training as TrainingClass

# Import utilities
from utils.logger import get_logger, log_execution_time, log_step, log_data_info
from utils.config import ConfigManager


class DatasetCreationWorkflow:
    """Workflow for creating datasets from raw data."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize dataset creation workflow.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('dataset_creation')
        self.flir_path = config.get_flir_path()
    
    @log_execution_time
    def create_datasets(self, args: Dict) -> None:
        """
        Create datasets based on command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        log_step("Initializing dataset creation")
        
        # Extract NDI tuple if provided
        ndi_tuple = args.get('ndi_tuple')
        
        # Initialize dataset creation
        ds_creation = dataset_creation(
            FlirImageExtractor_path=self.flir_path,
            config=self.config,
            ENV_FILE=False,  # Use ConfigManager instead
            CREATE=True,
            HS=args.get('hs', False),
            TH=args.get('th', False),
            RGB=args.get('rgb', False),
            create_ndi_table=args.get('create_ndi_table', False),
            ndi_tuple=ndi_tuple
        )
        
        self.logger.info("Dataset creation completed successfully")
        
        # Log created datasets
        if hasattr(ds_creation, 'spectral_img_df') and not ds_creation.spectral_img_df.empty:
            log_data_info(ds_creation.spectral_img_df, "Hyperspectral dataset")
        
        if hasattr(ds_creation, 'thermal_df') and not ds_creation.thermal_df.empty:
            log_data_info(ds_creation.thermal_df, "Thermal dataset")
            
        if hasattr(ds_creation, 'merged_img_df') and not ds_creation.merged_img_df.empty:
            log_data_info(ds_creation.merged_img_df, "Merged dataset")


class DatasetMergeWorkflow:
    """Workflow for merging hyperparameter and reference datasets."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize dataset merge workflow.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('dataset_merge')
        self.flir_path = config.get_flir_path()
    
    @log_execution_time
    def merge_datasets(self, args: Dict) -> None:
        """
        Merge datasets based on command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        log_step("Initializing dataset merge")
        
        # Get dataset paths
        hp_dataset_path = self.config.get_dataset_path(args.get('hp_dataset'))
        ref_dataset_path = self.config.get_dataset_path(args.get('ref_dataset'))
        project_name = args.get('project')
        
        
       # Get config path (always resolve to full path)
        config_name = args.get('config')
        config_path = self.config.get_config_path(config_name)
        
        self.logger.info(f"Merging datasets for project: {project_name}")
        self.logger.info(f"Hyperparameter dataset: {hp_dataset_path}")
        self.logger.info(f"Reference dataset: {ref_dataset_path}")
        self.logger.info(f"Configuration: {config_path}")
        
        # Initialize dataset creation for merging
        ds_creation = dataset_creation(
            FlirImageExtractor_path=self.flir_path,
            config=self.config,
            ENV_FILE=False,  # Use ConfigManager instead
            CREATE=False
        )
        
        # Perform merge
        ds_creation.merge_datasets(
            hp_dataset_path,
            ref_dataset_path,
            project_name=project_name,
            config_file_path=config_path
        )
        
        # Log results
        if hasattr(ds_creation, 'completed_df'):
            log_data_info(ds_creation.completed_df, "Merged dataset")
        
        self.logger.info("Dataset merge completed successfully")


class MLTrainingWorkflow:
    """Workflow for ML model training and evaluation."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize ML training workflow.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('ml_training')
    
    @log_execution_time
    def train_models(self, args: Dict) -> None:
        """
        Train and evaluate ML models.
        
        Args:
            args: Parsed command line arguments
        """
        log_step("Initializing ML training")
        
        # Get training parameters
        dataset_path = self.config.get_dataset_path(args.get('dataset'))
        features = args.get('features')
        target = args.get('target')
        task = args.get('task', 'regression')
        model = args.get('model')
        
        self.logger.info(f"Training {task} models")
        self.logger.info(f"Dataset: {dataset_path}")
        self.logger.info(f"Features: {features}")
        self.logger.info(f"Target: {target}")
        
        # Initialize training
        ml = TrainingClass(
            ENV_FILE=False,
            dataset_name=dataset_path,
            task=task,
            model=model
        )
        
        # Log dataset info
        log_data_info(ml.df, "Training dataset")
        
        # Choose evaluation method based on task type
        if task == 'regression':
            results_df = self._handle_regression_task(ml, features, target, args)
        elif task == 'classification':
            results_df = self._handle_classification_task(ml, features, target, args)
        else:
            raise ValueError(f"Unsupported task type: {task}")
        
        # Log results
        self.logger.info("Model evaluation completed")
        self.logger.info(f"Results shape: {results_df.shape}")
        
        # Save results to CSV
        outputs_dir = self.config.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(outputs_dir, f"model_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to: {csv_path}")
        
        # Display results
        print("\n=== MODEL EVALUATION RESULTS ===")
        print(results_df.to_string(index=False))
        
        # Generate plots for regression tasks
        if task == 'regression':
            if args.get('plot_separate'):
                self._generate_plots(ml, features, target, args)
            else:
                self._generate_single_plot(ml, features, target, args)
        
        return results_df
    
    def _handle_regression_task(self, ml: TrainingClass, features: List[str], target: str, args: Dict) -> pd.DataFrame:
        """Handle regression task evaluation."""
        results_df = ml.evaluate_regression_models(features, target)
        
        # Apply filtering if specified (only for Anthocyanin projects)
        if args.get('filter') == 'light':
            # Get project name from config
            project_name = self.config.get('metadata', {}).get('project_name', '')
            
            if project_name == 'Anthocyanin_BENI_ATAROT':
                self.logger.info("Detected Anthocyanin project, applying light filtering")
                
                # Get filter indicator from config
                filter_indicator = self.config.get('parameters', {}).get('indicator', 'catalog id')
                
                # Get categories from config and map to light conditions
                categories = self.config.get('categories', {})
                
                light_conditions = [
                    ('White and Blue Led', categories.get('RED_white_blue_led_ids', []) + categories.get('GREEN_white_blue_led_ids', [])),
                    ('White Led', categories.get('RED_white_led_ids', []) + categories.get('GREEN_white_led_ids', [])),
                    ('Shade', categories.get('RED_Shade_ids', []) + categories.get('GREEN_Shade_ids', [])),
                    ('Control', categories.get('RED_Control_ids', []) + categories.get('GREEN_Control_ids', []))
                ]
                
                for condition_name, ids in light_conditions:
                    if ids:  # Only process if there are IDs for this condition
                        self.logger.info(f"Evaluating for light condition: {condition_name}")
                        filtered_results = ml.evaluate_regression_models(
                            features, target,
                            filter_df=True,
                            filter_cond=condition_name,
                            filter_indicator=filter_indicator
                        )
                        filtered_results[filter_indicator] = condition_name
                        results_df = pd.concat([results_df, filtered_results], ignore_index=True)
            else:
                self.logger.warning(f"Light filtering specified but project '{project_name}' is not supported for light filtering")
                self.logger.info("Skipping light filtering")
        
        return results_df
    
    def _handle_classification_task(self, ml: TrainingClass, features: List[str], target: str, args: Dict) -> pd.DataFrame:
        """Handle classification task evaluation."""
        # For classification, we need to use different evaluation methods
        # The training class has classification methods but they work differently
        self.logger.info("Classification task detected - using classification evaluation")
        
        # Get available classification models from training class
        if hasattr(ml, 'classification_models_names'):
            model_names = ml.classification_models_names
        else:
            # Fallback if not set
            model_names = ['RandomForestClassifier', 'XGBClassifier']
        
        results_data = []
        
        for model_name in model_names:
            try:
                self.logger.info(f"Evaluating classification model: {model_name}")
                
                # Use the classification method from training class
                # Note: This will need to be adapted based on the actual classification interface
                acc, cr, cm = ml.eval_rf_classifer(target, None, None)  # This is a placeholder
                
                results_data.append({
                    'target': target,
                    'predictable_features': str(features),
                    'model_name': model_name,
                    'accuracy': acc,
                    'classification_report': cr
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {model_name}: {e}")
        
        return pd.DataFrame(results_data)
    
    def _generate_single_plot(self, ml: TrainingClass, features: List[str], target: str, args: Dict) -> None:
        """Generate a single regression plot."""
        self.logger.info("Generating regression plot")
        
        # Check number of predictive features (excluding target)
        predictive_features = [f for f in features if f != target]
        
        if len(predictive_features) == 1:
            # Use existing single feature plotting
            plot_feature = [features[0]]  # First feature
            plt = ml.plot_linear_regression(
                plot_feature, target,
                indecator='catalog id',
                show=False
            )
        else:
            # Use new prediction vs actual plotting for multi-feature models
            plt = ml.plot_prediction_vs_actual(
                features, target,
                show=False
            )
        
        # Save plot
        outputs_dir = self.config.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(outputs_dir, f"regression_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to: {plot_path}")
        
        # Show plot only if requested
        if args.get('show_plots'):
            plt.show()
        else:
            plt.close()
    
    def _generate_plots(self, ml: TrainingClass, features: List[str], target: str, args: Dict) -> None:
        """Generate separate plots for each condition."""
        self.logger.info("Generating separate plots by condition")
        
        # Check number of predictive features (excluding target)
        predictive_features = [f for f in features if f != target]
        
        if len(predictive_features) == 1:
            # Use existing single feature plotting
            plot_feature = [features[0]]  # First feature
            plt = ml.plot_linear_regression(
                plot_feature, target,
                indecator='catalog id',
                plot_separate=True,
                show=False
            )
        else:
            # Use new prediction vs actual plotting for multi-feature models
            plt = ml.plot_prediction_vs_actual(
                features, target,
                show=False
            )
        
        # Save plot
        outputs_dir = self.config.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(outputs_dir, f"regression_plots_separate_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Separate plots saved to: {plot_path}")
        
        # Show plot only if requested
        if args.get('show_plots'):
            plt.show()
        else:
            plt.close()


def create_workflow(workflow_type: str, config: ConfigManager):
    """
    Factory function to create workflow instances.
    
    Args:
        workflow_type: Type of workflow ('create', 'merge', 'train')
        config: Configuration manager instance
        
    Returns:
        Workflow instance
    """
    workflows = {
        'create': DatasetCreationWorkflow,
        'merge': DatasetMergeWorkflow,
        'train': MLTrainingWorkflow
    }
    
    if workflow_type not in workflows:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    return workflows[workflow_type](config)
