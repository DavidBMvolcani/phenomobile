"""
Anthocyanin-specific workflow for phenomobile project.

This module contains specialized workflow classes for anthocyanin projects,
inheriting from the base MLTrainingWorkflow and adding anthocyanin-specific
functionality.
"""

from .workflows import MLTrainingWorkflow, DatasetMergeWorkflow, PlottingWorkflow
from ml.training import Training as TrainingClass
from ml.train_on_ndi_tables import TrainOnNdiTables
from utils.config import ConfigManager
from typing import List, Dict
import pandas as pd
import os
from datetime import datetime
from utils.logger import setup_logger
from core.datasets_merge.merge_parameter_ds_with_ref_ds import MergeParameterDsWithRefDs
from core.datasets_merge.merge_lettuce_dataset import MergeLettuceDataset

from plotting.anthocyanin_plots import AnthocyaninPlot



class AnthocyaninDatasetMergeWorkflow(DatasetMergeWorkflow):
    """Specialized workflow for anthocyanin dataset merging."""
    
    def __init__(self, config: ConfigManager):
        """Initialize anthocyanin dataset merge workflow."""
        self.config = config
        self.logger = setup_logger('anthocyanin_merge', level='INFO')
        self.dt=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def merge_datasets(self, args: Dict) -> None:
        """Merge hyperparameter and reference datasets for anthocyanin projects."""
        self.logger.info("Starting anthocyanin dataset merge")
        
        # Get dataset paths from config
        params_dataset = self.config.get_dataset_path(args.get('params_dataset'))
        ref_dataset = self.config.get_dataset_path(args.get('ref_dataset'))
        
        # Create dataset merger
        merger = MergeLettuceDataset(dataset_folder=self.config.get('datasets_path'))
        
        # Merge datasets and save it in the datasets folder
        merger.merge_datasets(
            params_dataset,
            ref_dataset,
            self.config)
        
        
        self.logger.info("Anthocyanin dataset merge completed successfully")

class AnthocyaninMLTrainingWorkflow(MLTrainingWorkflow):
    """Specialized workflow for anthocyanin ML training."""
    
    def __init__(self, config: ConfigManager):
        """Initialize anthocyanin ML training workflow."""
        super().__init__(config)  # This initializes self.logger from parent class
        self.logger.info("AnthocyaninMLTrainingWorkflow initialized successfully")

        self.categories = self.configManager.get('categories', {})
        self.plot_cls = AnthocyaninPlot(self.categories)
        self.split_dataset_to_train_and_test = self.configManager.get('split_dataset_to_train_and_test', False)
        self.test_size = self.configManager.get('test_size', 0.2)
    
    def _handle_regression_task(self,
         ml: TrainingClass,
         features: List[str],
         target: str, 
         args: Dict,
         split: bool,
         test_size: float) -> pd.DataFrame:
        """Handle regression task with anthocyanin-specific filtering."""
        results_df = pd.DataFrame()
        
        cfg_parameters = self.configManager.get('parameters', {})
        transform_target = cfg_parameters.get('target_transform', False)
        transform_target_method = cfg_parameters.get('target_transform_method', 'sqrt')
        if transform_target:
            self.logger.info("Target transformation enabled by the method in config")

        # Apply filtering if specified (only for Anthocyanin projects)
        if args.get('filter') == 'light':
        
            self.logger.info(" applying light filtering")
            
            # Get filter indicator from config
            df_column_for_filtering = cfg_parameters.get('column_to_filter_by', 'Illumination')
            
           
            light_conditions = ml.df[df_column_for_filtering].unique()
            self.logger.info(f"Found light conditions: {light_conditions}")
            self.light_conditions = light_conditions
            
            for condition_name in light_conditions:
                self.logger.info(f"Evaluating for light condition: {condition_name}")
                filtered_results = ml.evaluate_regression_models(
                    features, target,
                    filter_df=True,
                    filter_cond=condition_name,
                    df_column_for_filtering=df_column_for_filtering,
                    split=split,
                    test_size=test_size,
                    transformed_target=transform_target,
                    transformed_target_method=transform_target_method

                )
                # Add condition name to each row
                filtered_results[df_column_for_filtering] = condition_name
                results_df = pd.concat([results_df, filtered_results], ignore_index=True)

        # No filtering
        else:
            results_df = ml.evaluate_regression_models(
                features,
                target, 
                split=split, 
                test_size=test_size,
                transformed_target=transform_target,
                transformed_target_method=transform_target_method 
            )
        return results_df
    
    def _train_models(self, args: Dict) -> pd.DataFrame:
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
        
        
        cfg_parameters = self.configManager.get('parameters', {})
        fix_method = cfg_parameters.get('fix_method', 'KEEP ROWS')
        task = cfg_parameters.get('task', 'regression')
        model = cfg_parameters.get('model', None)
        # Initialize training
        ml = TrainingClass(
            dataset_name=dataset_path,
            configManager=self.configManager,
            fix_method=fix_method,
            task=task,
            model=model,
            logger=self.logger
        )
        
        # Handle different tasks
        if task == 'regression':
            results_df = self._handle_regression_task(ml, features, target, args,
             self.split_dataset_to_train_and_test, self.test_size)
        elif task == 'classification':
            results_df = self._handle_classification_task(ml, features, target, args,
             self.split_dataset_to_train_and_test, self.test_size)
        else:
            raise ValueError(f"Unsupported task type: {task}")
        
        # Save results to CSV
        outputs_dir = self.configManager.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(outputs_dir, f"model_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to: {csv_path}")
        self.logger.info(f"About to start plot generation...")
        
        # Generate plots for regression tasks
        if task == 'regression':
            self._generate_plots(ml, features, target, args)
        
        return results_df

    def compute_r2_score_for_ndi_tables(self,args: Dict):
        dataset_path = args.get('dataset')
        target = args.get('target')
        task = args.get('task')
        config = self.configManager

        ml =TrainOnNdiTables (
            dataset_name=dataset_path,
            target=target,
            config=config,
            task=task,
            logger=self.logger,
        )
        r2_score_df=ml.compute_r2_score_for_ndi_cube()

        best_wl1, best_wl2, best_r2 = ml.get_best_ndi_combination_r2()
        self.logger.info(f"Best combination: {best_wl1} and {best_wl2} with R² = {best_r2:.4f}")
        
        # Save results to CSV
        outputs_dir = self.configManager.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(outputs_dir, f"model_results_{timestamp}.csv")
        r2_score_df.to_csv(csv_path)
        self.logger.info(f"R2 score results saved to: {csv_path}")

        #save plot
        plot = ml.plot_r2_results()
        plot_path = os.path.join(outputs_dir, f"r2_score_plot_{timestamp}.png")
        plot.savefig(plot_path)
        self.logger.info(f"R2 score plot saved to: {plot_path}")

        return r2_score_df

    def train_models(self, args: Dict) -> pd.DataFrame:
        compute_r2_scores = args.get('compute_r2_score_for_ndi_tables', False)
        if compute_r2_scores:
            self.logger.info("Computing R2 score for NDI tables")
            results_df=self.compute_r2_score_for_ndi_tables(args)
        else:
            """Train models with anthocyanin-specific plotting."""
            results_df = self._train_models(args) 
        return results_df
    

    def _handle_classification_task(self, ml: TrainingClass, features: List[str], target: str, args: Dict) -> pd.DataFrame:
        """Handle classification task evaluation."""
        # For classification, we need to use different evaluation methods
        # The training class has classification methods but they work differently
        results_df = ml.evaluate_classification_models(features, target)
        return results_df
    
    def _generate_plots(self, ml: TrainingClass, 
        features: List[str],
        target: str, 
        args: Dict,
        save_plot=True) -> None:
        """Generate plots for anthocyanin regression."""
        
        self.logger.info("Generating anthocyanin regression plots")
        
        if args.get('plot_separate'):
            self.logger.info("Generating separate plots by condition")
            plot_separate=True
        else:
            plot_separate=False
        
        # Check number of predictive features (excluding target)
        predictive_features = [f for f in features if f != target]
        
        if len(predictive_features) == 1:
            # Use plotting module for single feature plotting
            self.logger.info("Generating linear regression plot")
            
            plot_feature = predictive_features[0]  # First feature
            
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            outputs_dir = self.configManager.ensure_output_dir()
            plot_path = f'''{outputs_dir}/anthocyanin_regression_for_{plot_feature}_{ts}.png'''
            plt = self.plot_cls.plot_anthocyanin_linear_regression(
                plot_feature, 
                target, 
                ml.df, 
                'label_name', # 4th position maps to indicator
                None,         # 5th position maps to color_map 
                None,         # 6th position maps to condition
                plot_separate, 
                False, 
                self.categories,
                True,
                plot_path
            )
           
        # more than one feature
        else:
            outputs_dir = self.configManager.ensure_output_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use plotting module for prediction vs actual plotting
            self.logger.info("Generating prediction vs actual plot")
            
            if hasattr(self, 'light_conditions'):
                for cond in self.light_conditions:
                    plt = self.plot_cls.plot_prediction_vs_actual(
                        features, target, ml.df[ml.df['Illumination'] == cond],
                        show=False )
                    plot_path = os.path.join(outputs_dir,
                         f'''anthocyanin_regression_for_{",".join(predictive_features)}_filtered_by_{cond}_{ts}.png''')
                    # Save the plot
                    self.plot_cls.save_plot(plt, plot_path)
            else:
                plt = self.plot_cls.plot_prediction_vs_actual(
                    features, target, ml.df,
                    show=False
                )
                plot_path = f'''{outputs_dir}/anthocyanin_regression_for_{",".join(predictive_features)}_{ts}.png'''
                self.plot_cls.save_plot(plt, plot_path)
        
        # Close the plot to free memory
        plt.close()

# does this class neccssary ??                    
class AnthocyaninPlottingWorkflow(PlottingWorkflow):
    def __init__(self, config: ConfigManager):
        super().__init__(config)
    
    def _generate_anthocyanin_plots(self, df: pd.DataFrame, args: Dict) -> None:
        """Generate anthocyanin-specific plots."""
        
        features = args.get('features', '').split(',')
        target = args.get('target')
        
        if not features or not target:
            raise ValueError("Both --features and --target required for anthocyanin plots")
        
        # Get categories from config if available
        categories = self.configManager.get('categories', {})
        
        plt = self.plot_cls.plot_anthocyanin_linear_regression(
            features, target, df,
            indicator=args.get('indicator', 'label_name'),
            condition=args.get('condition'),
            plot_separate=args.get('plot_separate', False),
            show=False,
            categories=categories
        )
        
        self.plot_cls.save_plot(plt, "anthocyanin_plot")