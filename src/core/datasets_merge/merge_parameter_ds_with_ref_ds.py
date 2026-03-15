
from abc import abstractmethod
from scipy.stats import gaussian_kde
import pandas as pd
from datetime import datetime

class MergeParameterDsWithRefDs:
    def __init__(
        self,
        dataset_folder
    ):
        self.dataset_folder = dataset_folder
        self.formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

     
    # GOAL:
        # merge two datasets. The coomon Use of this function 
        # is to merge the reference dataset with the parameters dataset
        # because the process of merging is vary by the kind of project
        # we apply for each project specefic merger function
    # PARAMETERS
        #   - param_ds_path : the path to the dataset that contain parameters 
        #     that proceced from the raw data like vegetation indices 
        #   - ref_ds_path: path to dataset that contain the traget object
        #   - config_file_path: path to json configuration file

    @abstractmethod
    def merge_datasets(self,param_ds_path,ref_ds_path,config_file_path):
       pass
    
    def merge_with_samples(self, 
        parameters_df, 
        ref_df, 
        categories, 
        group_by_col, 
        target, 
        completed_df):
        """
        Merge parameters dataset with reference dataset using KDE sampling.
        
        PARAMETERS:
            - parameters_df: parameters dataframe
            - ref_df: reference dataframe
            - categories: dict mapping category names to label lists
            - group_by_col: column name to group by
            - target: target column to populate with samples
            - completed_df: dataframe to fill with sampled values
            
        RETURNS:
            - completed_df with sampled target values
        """
        category_gb = parameters_df.groupby('category')['label_name'].count()
        categories_names = list(categories.keys())

        for category_name in categories_names:
            category_labels = categories[category_name]
            size_of_samples = category_gb[category_name]
            
            # Get records belonging to this category
            sub_df = ref_df[ref_df[group_by_col].isin(category_labels)]
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
        
    def save_mergerd_reference_and_parameters_ds(
        self,
        reference_dataset_name='',
        completed_df=None
    ):
        datasets_paths=self.dataset_folder
        dt=self.formatted_datetime
        completed_df.to_csv(f"{datasets_paths}/complete_ds_of_{reference_dataset_name}_{dt}.csv",index=False)
        print(f'new file created :{datasets_paths}/complete_ds_of_{reference_dataset_name}_{dt}.csv')
