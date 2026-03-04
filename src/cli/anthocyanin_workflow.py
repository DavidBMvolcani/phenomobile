"""
Anthocyanin-specific workflow for phenomobile project.

This module contains specialized workflow classes for anthocyanin projects,
inheriting from the base MLTrainingWorkflow and adding anthocyanin-specific
functionality.
"""

from .workflows import MLTrainingWorkflow, _get_training_class
from typing import List, Dict, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from ml.training import training


class AnthocyaninMLTrainingWorkflow(MLTrainingWorkflow):
    """Specialized workflow for anthocyanin ML training."""
    
    def _handle_regression_task(self, ml: 'training', features: List[str], target: str, args: Dict) -> pd.DataFrame:
        """Handle regression task with anthocyanin-specific filtering."""
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
    
    def train_models(self, args: Dict) -> None:
        """Train models with anthocyanin-specific plotting."""
        # Call parent method for generic training (without plotting)
        results_df = self.train_models_without_plotting(args)
        
        # Add anthocyanin-specific plotting
        features = args.get('features')
        target = args.get('target')
        
        # Get training class
        TrainingClass = _get_training_class(self.config)
        ml = TrainingClass(
            dataset_name=self.config.get_dataset_path(args.get('dataset')),
            config=self.config,
            task=args.get('task', 'regression'),
            model=args.get('model')
        )
        
        # Generate plots for regression tasks
        if args.get('task', 'regression') == 'regression':
            if args.get('plot_separate'):
                self._generate_plots(ml, features, target, args)
            else:
                self._generate_single_plot(ml, features, target, args)
        
        return results_df
