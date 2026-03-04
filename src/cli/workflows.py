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


def _get_training_class(config: ConfigManager):
    """Get appropriate training class based on project."""
    project_name = config.get('metadata.project_name', '')
    
    if 'Anthocyanin' in project_name:
        from ml.anthocyanin_training import anthocyanin_training as TrainingClass
    else:
        from ml.training import training as TrainingClass
    
    return TrainingClass


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
        
        # Validate thermal processing requirements
        if args.get('th', False):
            self._validate_thermal_requirements()
        
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
    
    def _validate_thermal_requirements(self) -> None:
        """Validate that thermal processing requirements are met."""
        try:
            # Test if FLIR path is accessible
            import sys
            if self.flir_path not in sys.path:
                sys.path.append(self.flir_path)
            
            # Try to import the module
            import flir_image_extractor
            
            # Try to create an instance (basic validation)
            test_extractor = flir_image_extractor.FlirImageExtractor()
            
            self.logger.info("FLIR image extractor validation passed")
            
        except ImportError as e:
            error_msg = f"Thermal processing requested but FLIR image extractor is not available: {e}"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"FLIR image extractor validation failed: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)


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
        TrainingClass = _get_training_class(self.config)
        ml = TrainingClass(
            dataset_name=dataset_path,
            config=self.config,
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
        
        return results_df
    
    def train_models_without_plotting(self, args: Dict) -> None:
        """Train models without plotting (for inheritance)."""
        return self.train_models(args)
    
    def _handle_regression_task(self, ml: TrainingClass, features: List[str], target: str, args: Dict) -> pd.DataFrame:
        """Handle regression task evaluation."""
        results_df = ml.evaluate_regression_models(features, target)
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
            # Use plotting module for single feature plotting
            from plotting.anthocyanin_plots import plot_anthocyanin_linear_regression
            
            plot_feature = [features[0]]  # First feature
            plt = plot_anthocyanin_linear_regression(
                plot_feature, target, ml.df,
                indecator='catalog id',
                show=False,
                categories=ml.config.get('categories', {}) if hasattr(ml, 'config') and ml.config else {}
            )
        else:
            # Use plotting module for prediction vs actual plotting
            from plotting.base_plots import plot_prediction_vs_actual
            plt = plot_prediction_vs_actual(
                features, target, ml.df,
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
            # Use plotting module for single feature plotting
            from plotting.anthocyanin_plots import plot_anthocyanin_linear_regression
            
            plot_feature = [features[0]]  # First feature
            plt = plot_anthocyanin_linear_regression(
                plot_feature, target, ml.df,
                indecator='catalog id',
                plot_separate=True,
                show=False,
                categories=ml.config.get('categories', {}) if hasattr(ml, 'config') and ml.config else {}
            )
        else:
            # Use plotting module for prediction vs actual plotting
            from plotting.base_plots import plot_prediction_vs_actual
            plt = plot_prediction_vs_actual(
                features, target, ml.df,
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


class PlottingWorkflow:
    """Workflow for generating plots from datasets."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize plotting workflow.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def generate_plots(self, args: Dict) -> None:
        """
        Generate plots based on specified type.
        
        Args:
            args: Dictionary of command line arguments
        """
        plot_type = args.get('type')
        dataset_path = args.get('dataset')
        
        self.logger.info(f"Generating {plot_type} plots from {dataset_path}")
        
        # Load dataset
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Loaded dataset with shape: {df.shape}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Generate plots based on type
        if plot_type == 'regression':
            self._generate_regression_plots(df, args)
        elif plot_type == 'anthocyanin':
            self._generate_anthocyanin_plots(df, args)
        elif plot_type == 'r2-score-of-ndi':
            self._generate_ndi_heatmap_plots(df, args)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _generate_regression_plots(self, df: pd.DataFrame, args: Dict) -> None:
        """Generate regression plots."""
        from plotting.base_plots import plot_prediction_vs_actual
        
        features = args.get('features', '').split(',')
        target = args.get('target')
        
        if not features or not target:
            raise ValueError("Both --features and --target required for regression plots")
        
        plt = plot_prediction_vs_actual(
            features, target, df,
            condition=args.get('condition'),
            show=False
        )
        
        self._save_plot(plt, "regression_plot", args)
    
    def _generate_anthocyanin_plots(self, df: pd.DataFrame, args: Dict) -> None:
        """Generate anthocyanin-specific plots."""
        from plotting.anthocyanin_plots import plot_anthocyanin_linear_regression
        
        features = args.get('features', '').split(',')
        target = args.get('target')
        
        if not features or not target:
            raise ValueError("Both --features and --target required for anthocyanin plots")
        
        # Get categories from config if available
        categories = self.config.get('categories', {})
        
        plt = plot_anthocyanin_linear_regression(
            features, target, df,
            indicator=args.get('indicator', 'catalog id'),
            condition=args.get('condition'),
            plot_separate=args.get('plot_separate', False),
            show=False,
            categories=categories
        )
        
        self._save_plot(plt, "anthocyanin_plot", args)
    
    def _generate_ndi_heatmap_plots(self, df: pd.DataFrame, args: Dict) -> None:
        """Generate NDI R² score heatmap plots."""
        from plotting.base_plots import plot_heatmap_of_r2_score_of_ndi
        
        # For NDI plots, we need r2_ndi_df which should be precomputed
        # This is a placeholder - in practice, this data would need to be
        # generated from NDI calculations
        self.logger.warning("NDI heatmap plots require precomputed R² data")
        
        # Placeholder implementation
        model = args.get('model', 'linear_regression')
        target = args.get('target', 'anthocyanin')
        
        # Create dummy data for demonstration
        import numpy as np
        dummy_r2_df = pd.DataFrame(
            np.random.rand(10, 10),
            index=[f"band_{i}" for i in range(10)],
            columns=[f"band_{i}" for i in range(10)]
        )
        
        plt = plot_heatmap_of_r2_score_of_ndi(
            dummy_r2_df, model, target, show=False
        )
        
        self._save_plot(plt, "ndi_r2_heatmap", args)
    
    def _save_plot(self, plt, plot_name: str, args: Dict) -> None:
        """Save plot to file and optionally display."""
        outputs_dir = self.config.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(outputs_dir, f"{plot_name}_{timestamp}.png")
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to: {plot_path}")
        
        # Show plot only if requested
        if args.get('show_plots'):
            plt.show()
        else:
            plt.close()


def get_workflow(workflow_type: str, config: ConfigManager):
    """
    Factory function to get workflow instance.
    
    Args:
        workflow_type: Type of workflow ('create', 'merge', 'train', 'plot')
        config: Configuration manager instance
        
    Returns:
        Workflow instance
    """
    workflows = {
        'create': DatasetCreationWorkflow,
        'merge': DatasetMergeWorkflow,
        'train': _get_workflow_class(config),
        'plot': PlottingWorkflow
    }
    
    if workflow_type not in workflows:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    return workflows[workflow_type](config)


def _get_workflow_class(config: ConfigManager):
    """Get appropriate workflow class based on project."""
    project_name = config.get('metadata.project_name', '')
    
    if 'Anthocyanin' in project_name:
        from .anthocyanin_workflow import AnthocyaninMLTrainingWorkflow as WorkflowClass
    else:
        WorkflowClass = MLTrainingWorkflow
    
    return WorkflowClass
