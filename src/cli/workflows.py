"""
Workflow orchestration for phenomobile CLI.
High-level workflows that wrap existing classes with CLI-friendly interfaces.
"""

import os
import sys
import pandas as pd
import numpy as np
import h5py

from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import existing modules
from ml.training import Training as TrainingClass

# Import utilities
from utils.logger import get_logger, setup_logger, log_execution_time, log_step, log_data_info
from utils.config import ConfigManager

#import my core modules
from core.datasets_creation.hyper_spectral_ds_creation import HyperSpectralDsCreation
from core.datasets_creation.thermal_ds_creation import ThermalDsCreation
from core.datasets_creation.rgb_ds_creation import RgbDsCreation

from abc import ABC, abstractmethod

from dotenv import load_dotenv


class DatasetCreationWorkflow:
    """Workflow for creating datasets from raw data."""
    
    def __init__(self, configManager: ConfigManager):
        """
        Initialize dataset creation workflow.
        
        Args:
            configManager: Configuration manager instance
        """
        self.configManager = configManager
        self.logger = get_logger('dataset_creation')
        self.flir_path = configManager.get_flir_path()
    
  
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
        
        # Get config and experiment settings
      
    
        HS_dataset_creation_parameters = self.configManager.get('HS_dataset_creation_parameters', {})
        SPLIT=HS_dataset_creation_parameters.get('SPLIT_IMAGE_TO_OBJECTS')
        ROTATE_IMAGE=HS_dataset_creation_parameters.get('ROTATE_IMAGE')

        HS_ANNOTATION_FILE_NAME = HS_dataset_creation_parameters.get('ANNOTATION_FILE')
        HS_object_filter_method = HS_dataset_creation_parameters.get('object_filter_method')
        HS_ndvi_threshold = HS_dataset_creation_parameters.get('ndvi_threshold')
        HS_hsv_filter_thresholds = HS_dataset_creation_parameters.get('hsv_filter_thresholds',{})
        
        RGB_dataset_creation_parameters = self.configManager.get('RGB_dataset_creation_parameters', {})
        RGB_ANNOTATION_FILE_NAME = RGB_dataset_creation_parameters.get('ANNOTATION_FILE')
        
        home_dir = self.configManager.get('home_path')
        paths = self.configManager.get('paths', {})
        download_folder = self.configManager.get('download_path')
        dataset_folder = self.configManager.get('datasets_path')
        self.dataset_folder = dataset_folder
        self.formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.COMPUTE_NDI = args.get('create_ndi_table')
        self.ndi_table_directory_path=self.configManager.get('ndi_table_directory_path','{}')

        self.save_ndi_table_as=self.configManager.get('save_ndi_table_as','hdf5')
        
        
        FlirImageExtractor_path = self.configManager.get('FlirImageExtractor_path')

        # Get data source
        data_src=self.configManager.get('data_source')
        self.data_source = data_src['type']
        if self.data_source == 'server':
            self.logger.info("Loading environment variables from .env file")
            self.setup_paths_from_env_file()
        else:
            self.logger.info("Using local data source")
            self.setup_paths_from_config()
            self.RAW_DATA_FOLDER = self.configManager.get('download_folder')

        # Get dataset creation flags
        hs = args.get('hs', False)
        th = args.get('th', False)
        rgb = args.get('rgb', False)

        map_hs_and_th_ds= True if hs and th else False

        # create datasets by user choice
        if hs:
            # Create HS dataset
            hs_ds=HyperSpectralDsCreation(
                logger=self.logger,
                map_hs_and_th_ds=map_hs_and_th_ds,
                annotation_file_name=HS_ANNOTATION_FILE_NAME,
                split_image_to_objects=SPLIT,
                home_dir=home_dir,
                rotate_image=ROTATE_IMAGE,
                
                download_folder=download_folder,
                dataset_folder=self.dataset_folder,
                formatted_datetime=self.formatted_datetime,

                data_source=self.data_source,
                ndi_tuple=ndi_tuple,
                COMPUTE_NDI=self.COMPUTE_NDI,
                ndi_table_directory_path=self.ndi_table_directory_path,
                ndi_storage_method=self.save_ndi_table_as,

                SMB_USERNAME=self.SMB_USERNAME,
                SMB_PASSWORD=self.SMB_PASSWORD,
                SMB_SERVER=self.SMB_SERVER,
                SMB_SHARE=self.SMB_SHARE,
                RAW_DATA_FOLDER=self.RAW_DATA_FOLDER,
                YEAR_DIR_NAME=self.server_year_dir_name,
                DATE_DIR_NAME=self.server_date_dir_name,
               

                object_filter_method=HS_object_filter_method,
                ndvi_threshold=HS_ndvi_threshold,
                hsv_filter_thresholds=HS_hsv_filter_thresholds
            )
            self.spectral_img_df, self.gray_for_HS_imgs = hs_ds.create_dataset()
        
        if th:
            self.logger.info("Creating TH dataset")
            th_ds = ThermalDsCreation(
                logger=self.logger,
                map_hs_and_th_ds=map_hs_and_th_ds,
                home_dir=home_dir,
                download_folder=download_folder,
                dataset_folder=self.dataset_folder,
                formatted_datetime=self.formatted_datetime,
                data_source=self.data_source,
                FlirImageExtractor_path=FlirImageExtractor_path,
                SMB_USERNAME=self.SMB_USERNAME,
                SMB_PASSWORD=self.SMB_PASSWORD,
                SMB_SERVER=self.SMB_SERVER,
                SMB_SHARE=self.SMB_SHARE,
                RAW_DATA_FOLDER=self.RAW_DATA_FOLDER,
                YEAR_DIR_NAME=self.server_year_dir_name,
                DATE_DIR_NAME=self.server_date_dir_name
            )
            self.thermal_img_df, self.gray_for_TH_imgs = th_ds.create_dataset()
        if rgb:
            self.logger.info("Creating RGB dataset")
            rgb_ds = RgbDsCreation(
                logger=self.logger,
                home_dir=home_dir,
                download_folder=download_folder,
                dataset_folder=self.dataset_folder,
                formatted_datetime=self.formatted_datetime,
                data_source=self.data_source,
                annotation_file_name=RGB_ANNOTATION_FILE_NAME,
                SMB_USERNAME=self.SMB_USERNAME,
                SMB_PASSWORD=self.SMB_PASSWORD,
                SMB_SERVER=self.SMB_SERVER,
                SMB_SHARE=self.SMB_SHARE,
                RAW_DATA_FOLDER=self.RAW_DATA_FOLDER,
                server_year_dir_name=self.server_year_dir_name,
                server_date_dir_name=self.server_date_dir_name,
                config=self.configManager  
            )
            self.rgb_img_df = rgb_ds.create_dataset()

        if map_hs_and_th_ds:
            self.logger.info("Mapping hyperspectral and thermal datasets")
            merge_ds = MergeParameterDs(
                logger=self.logger,
                dataset_folder=self.dataset_folder,
                RAW_DATA_FOLDER=self.RAW_DATA_FOLDER
            )
            merge_ds.map_hyper_spectral_and_thermal_datasets(
                gray_for_HS_imgs=self.gray_for_HS_imgs,
                gray_for_Th_imgs=self.gray_for_TH_imgs,
                gray_for_HS_imgs_directory_path=self.gray_for_HS_imgs_directory_path,
                gray_for_Th_imgs_directory_path=self.gray_for_TH_imgs_directory_path,
                spectral_img_df=self.spectral_img_df,
                thermal_df=self.thermal_img_df,
            )
        
        self.logger.info("Dataset creation completed successfully")
        
        # Log created datasets
        if 'hs_ds' in locals() and hasattr(hs_ds, 'spectral_img_df') and not hs_ds.spectral_img_df.empty:
            log_data_info(hs_ds.spectral_img_df, "Hyperspectral dataset")
        
        if 'th_ds' in locals() and hasattr(th_ds, 'thermal_df') and not th_ds.thermal_df.empty:
            log_data_info(th_ds.thermal_df, "Thermal dataset")
            
        if 'rgb_ds' in locals() and hasattr(rgb_ds, 'rgb_img_df') and not rgb_ds.rgb_img_df.empty:
            log_data_info(rgb_ds.rgb_img_df, "RGB dataset")

    def setup_paths_from_env_file(self):
        # Fallback to environment variables (legacy support)
        load_dotenv(override=True, dotenv_path='.env')
        self.SMB_USERNAME = os.environ.get("SMB_USERNAME")
        self.SMB_PASSWORD = os.environ.get("SMB_PASSWORD")
        self.SMB_SERVER = os.environ.get("SMB_SERVER")
        self.SMB_SHARE = os.environ.get("SMB_SHARE")
        
        self.RAW_DATA_FOLDER = os.environ.get("REMOTE_FOLDER")
        self.server_year_dir_name = os.environ.get("year")
        self.server_date_dir_name = os.environ.get("date")  
    
    def setup_paths_from_config(self):
        self.RAW_DATA_FOLDER = self.configManager.get('download_folder')

        self.SMB_USERNAME = None
        self.SMB_PASSWORD = None
        self.SMB_SERVER = None
        self.SMB_SHARE = None
        self.server_year_dir_name = None
        self.server_date_dir_name = None  
          

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


# Abstract class 
class DatasetMergeWorkflow(ABC):
    """Workflow for merging hyperparameter and reference datasets."""
    
    def __init__(self, configManager: ConfigManager):
        pass
    
    @abstractmethod
    def merge_datasets(self, args: Dict) -> None:
        pass


class MLTrainingWorkflow:
    """Workflow for ML model training and evaluation."""
    
    def __init__(self, configManger: ConfigManager):
        """
        Initialize ML training workflow.
        
        Args:
            configManager: Configuration manager instance
        """
        self.configManager = configManger
        self.logger = setup_logger('ml_training', level='INFO')  # Explicitly set INFO level
    
    @log_execution_time
    def train_models(self, args: Dict) -> None:
        """
        Train and evaluate ML models.
        
        Args:
            args: Parsed command line arguments
        """
        log_step("Initializing ML training")
        
        # Get training parameters
        dataset_path = self.configManager.get_dataset_path(args.get('dataset'))
        features = args.get('features')
        target = args.get('target')
        task = args.get('task', 'regression')
        model = args.get('model')
        
        self.logger.info(f"Training {task} models")
        self.logger.info(f"Dataset: {dataset_path}")
        self.logger.info(f"Features: {features}")
        self.logger.info(f"Target: {target}")
        
        # Initialize training
        from ml.training import Training as TrainingClass
        ml = TrainingClass(
            dataset_name=dataset_path,
            config=self.configManager,
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
        outputs_dir = self.configManager.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(outputs_dir, f"model_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to: {csv_path}")
          
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
    
 
class PlottingWorkflow:
    """Workflow for generating plots from datasets."""
    
    def __init__(self, configManager: ConfigManager):
        """
        Initialize plotting workflow.
        
        Args:
            config: Configuration manager instance
        """
        self.configManager = configManager
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
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Loaded dataset with shape: {df.shape}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
       
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
    
    
    def _save_plot(self, plt, plot_name: str, args: Dict) -> None:
        """Save plot to file and optionally display."""
        outputs_dir = self.configManager.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(outputs_dir, f"{plot_name}_{timestamp}.png")
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plot saved to: {plot_path}")
        
        # Show plot only if requested
        if args.get('show_plots'):
            plt.show()
        else:
            plt.close()



def get_workflow(workflow_type: str, configManager: ConfigManager):
    """
    Factory function to get workflow instance.
    
    Args:
        workflow_type: Type of workflow ('create', 'merge', 'train', 'plot')
        configManager: Configuration manager instance
        
    Returns:
        Workflow instance
    """
    workflows = {
        'create': DatasetCreationWorkflow,
        'merge': _get_workflow_class('merge', configManager),
        'train': _get_workflow_class('train', configManager),
        'plot': _get_workflow_class('plot', configManager)
    }
    
    if workflow_type not in workflows:
        raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    return workflows[workflow_type](configManager)


def _get_workflow_class(workflow_type: str,configManager: ConfigManager):
    """Get appropriate workflow class based on project."""
    metadata = configManager.get('metadata', {})
    project_name = metadata.get('project_name', '')
    
    if workflow_type == 'train':
        if 'Anthocyanin' in project_name:
            from .anthocyanin_workflow import AnthocyaninMLTrainingWorkflow as WorkflowClass
        else:
            WorkflowClass = MLTrainingWorkflow
        
    if workflow_type == 'merge':
        if 'Anthocyanin' in project_name:
            from .anthocyanin_workflow import AnthocyaninDatasetMergeWorkflow as WorkflowClass
        else:
             raise NotImplementedError("Merge workflow not implemented for this project type")
        
    if workflow_type == 'plot':
        if 'Anthocyanin' in project_name:
            from .anthocyanin_workflow import AnthocyaninPlottingWorkflow as WorkflowClass
        else:
            WorkflowClass = PlottingWorkflow
    
    return WorkflowClass
