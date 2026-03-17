import pandas as pd
import json

from pathlib import Path

from core.datasets_merge.merge_parameter_ds_with_ref_ds import MergeParameterDsWithRefDs


class MergeLettuceDataset(MergeParameterDsWithRefDs):
    """
    Lettuce-specific dataset class that inherits from datasets.lettuce_dataset merging strategies.
    """
    
    def __init__(self,dataset_folder):
        super().__init__(dataset_folder)
    
    # OVERRIDE
    def merge_datasets(self, param_ds_path, ref_ds_path, config):
        """
        Merge parameters and reference datasets for lettuce project.
        
        PARAMETERS:
            - param_ds_path: path to parameters dataset
            - ref_ds_path: path to reference dataset
            - config: configuration dictionary
        """
        parameters = config.config['parameters']
        kind_of_merged = parameters['kind_of_merged']
        
        self.merge_lettuce_BENI_ATAROT_datasets(param_ds_path, ref_ds_path,
                                                kind_of_merged, config)

    def merge_lettuce_BENI_ATAROT_datasets(self, param_ds_path, ref_ds_path,
                                           kind_of_merged, config):
        """
        Merge parameters and reference datasets for Lettuce BENI_ATAROT project.
        
        PARAMETERS:
            - param_ds_path: path to parameters dataset CSV
            - ref_ds_path: path to reference dataset CSV
            - kind_of_merged: merge strategy ('sample_hp' or 'as_it')
            - config: configuration dictionary
        """
    
        params_df = pd.read_csv(param_ds_path)
        reference_df = pd.read_csv(ref_ds_path)

        reference_dataset_name = Path(ref_ds_path).stem
        
       

        parameters = config.config['parameters']
        target = parameters['target']
        group_by_col = parameters['group_by_col']
        categories = config.config['categories']

        # Map categories to hp_df
        def map_category(label):
            for category_name, list_of_labels in categories.items():
                if label in list_of_labels:
                    return category_name
            return None

        params_df['category'] = params_df['label_name'].apply(map_category)
        params_df[group_by_col] = params_df['label_name']
        
        # Get the stems (filenames without extensions)
        params_suffix = f"_{Path(param_ds_path).stem}"
        ref_suffix = f"_{Path(ref_ds_path).stem}"

        # Choose merge strategy
        if kind_of_merged == 'sample_params':
            r_df = reference_df.copy()
            completed_df = params_df.copy()
            completed_df[target] = None
            self.completed_df = self.merge_with_samples(params_df, r_df,
                categories, group_by_col, target, completed_df)
        else:  # kind_of_merged == 'as_it'
            # Merge dataframes by group_by_col
            # Merge with custom suffixes
            self.completed_df = pd.merge(
                params_df, 
                reference_df, 
                on=group_by_col, 
                how='inner',
                suffixes=(params_suffix, ref_suffix)
            )
        # Drop the column used for merging
        self.completed_df = self.completed_df.drop(columns=[group_by_col])

        # Save the merged dataset
        self.save_mergerd_reference_and_parameters_ds(
            reference_dataset_name,self.completed_df)