import pandas as pd
import json
from scipy.stats import gaussian_kde
from core.dataset_creation import dataset_creation


class lettuce_dataset(dataset_creation):
    """
    Lettuce-specific dataset class that inherits from dataset_creation.
    Implements lettuce dataset merging strategies.
    """
    
    def __init__(self, FlirImageExtractor_path, config=None, ENV_FILE=True, CREATE=True,
             HS=False, TH=False, RGB=False,
             create_ndi_table=False, ndi_table_directory_path=None,
             ndi_tuple=None):
        """Initialize lettuce dataset with parent class parameters"""
        super().__init__(FlirImageExtractor_path, config, ENV_FILE, CREATE,
                        HS, TH, RGB, create_ndi_table, 
                        ndi_table_directory_path, ndi_tuple)
    
    def merge_with_samples(self, hp_df, r_df, categories, indicator, target, completed_df):
        """
        Merge hyperparameter dataset with reference dataset using KDE sampling.
        
        PARAMETERS:
            - hp_df: hyperparameter dataframe
            - r_df: reference dataframe
            - categories: dict mapping category names to label lists
            - indicator: column name to group by
            - target: target column to populate with samples
            - completed_df: dataframe to fill with sampled values
            
        RETURNS:
            - completed_df with sampled target values
        """
        category_gb = hp_df.groupby('category')['label_name'].count()
        categories_names = list(categories.keys())

        for category_name in categories_names:
            category_labels = categories[category_name]
            size_of_samples = category_gb[category_name]
            
            # Get records belonging to this category
            sub_df = r_df[r_df[indicator].isin(category_labels)]
            target_values = sub_df[target].values

            if len(sub_df) > 0:
                # Create KDE for this category's values
                kde = gaussian_kde(target_values)
                
                # Generate new samples
                new_samples = kde.resample(size=size_of_samples).flatten()
                
                # Copy samples to completed dataframe
                indexes = completed_df[completed_df['category'] == category_name].index
                for i in range(len(indexes)): 
                    idx = indexes[i]
                    completed_df.loc[idx, target] = new_samples[i]
        
        return completed_df
    
    def merge_lettuce_BENI_ATAROT_datasets(self, hp_ds_path, ref_ds_path,
                                           kind_of_merged, config_file_path):
        """
        Merge hyperparameter and reference datasets for Lettuce BENI_ATAROT project.
        
        PARAMETERS:
            - hp_ds_path: path to hyperparameter dataset CSV
            - ref_ds_path: path to reference dataset CSV
            - kind_of_merged: merge strategy ('sample_hp' or 'as_it')
            - config_file_path: path to JSON configuration file
        """
        hyperParameter_df = pd.read_csv(hp_ds_path)
        reference_df = pd.read_csv(ref_ds_path)

        reference_dataset_name = ref_ds_path.split('/')[-1]
        reference_dataset_name = reference_dataset_name.split('.')[0]
        
        hp_df = hyperParameter_df.copy()

        # Read configuration from JSON file
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        parameters = config['parameters']
        target = parameters['target']
        indicator = parameters['indicator']
        categories = config['categories']

        # Map categories to hp_df
        def map_category(label):
            for category_name, list_of_labels in categories.items():
                if label in list_of_labels:
                    return category_name
            return None

        hp_df['category'] = hp_df['label_name'].apply(map_category)
        hp_df[indicator] = hyperParameter_df['label_name']
        
        # Choose merge strategy
        if kind_of_merged == 'sample_hp' and config_file_path:
            r_df = reference_df.copy()
            completed_df = hp_df.copy()
            completed_df[target] = None
            self.completed_df = self.merge_with_samples(hp_df, r_df,
                categories, indicator, target, completed_df)
        else:  # kind_of_merged == 'as_it'
            # Merge dataframes by indicator
            self.completed_df = pd.merge(hp_df, reference_df, 
                                        on=indicator, how='inner')
        
        self.save_mergerd_reference_and_hyperParameter_ds(reference_dataset_name)
    
    def merge_datasets(self, hp_ds_path, ref_ds_path, config_file_path):
        """
        Merge hyperparameter and reference datasets for lettuce project.
        
        PARAMETERS:
            - hp_ds_path: path to hyperparameter dataset
            - ref_ds_path: path to reference dataset
            - config_file_path: path to JSON configuration file
        """
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        
        parameters = config['parameters']
        kind_of_merged = parameters['kind_of_merged']
        
        self.merge_lettuce_BENI_ATAROT_datasets(hp_ds_path, ref_ds_path,
                                                kind_of_merged, config_file_path)