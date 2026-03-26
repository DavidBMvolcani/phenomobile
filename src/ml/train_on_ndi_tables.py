import pandas as pd
import numpy as np
from typing import override
import h5py

# my files
from .training import Training
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression

import seaborn as sns
import matplotlib.pyplot as plt

# PARAMETERS
#   - READ_NDI_TABLES_METHOD = 'from_hdf5'  # 'from_hdf5' or 'from_csv'
#   - H5_FILE_PATH = None  # path to hdf5 file with NDI tables

class TrainOnNdiTables(Training):
    def __init__(
        self,
        dataset_name,
        target,
        config=None,
        fix_method='KEEP ROWS',
        task='regression',
        model=None,
        logger=None,
        READ_NDI_TABLES_METHOD ='from_hdf5',
        H5_FILE_PATH=None,
       ):
        super().__init__(dataset_name, config, fix_method, task, model, logger)

        if config is not None:
            HS_parameters=config.get("HS_dataset_creation_parameters")
            READ_NDI_TABLES_METHOD = HS_parameters.get('READ_NDI_TABLES_METHOD', READ_NDI_TABLES_METHOD)
            H5_FILE_PATH = config.get('H5_FILE_PATH', H5_FILE_PATH)
            self.split_dataset_to_train_and_test = config.get('split_dataset_to_train_and_test', False)
        
        else:
            self.split_dataset_to_train_and_test = False
        
        # get reference dataframe from parent class
        self.ref_df = self.df.copy() 
        if target not in self.ref_df.columns:
            raise ValueError(f"Target {target} not found in reference dataframe")
        self.target = target

        # read NDI tables
        if READ_NDI_TABLES_METHOD == 'from_hdf5':
            self.read_ndi_tables_from_hdf5(H5_FILE_PATH)
        elif READ_NDI_TABLES_METHOD == 'from_csv':
            self.read_ndi_tables_from_csv()
        else:
            raise ValueError('Invalid READ_NDI_TABLES_METHOD')
        
        


    ######
    #
    # FUNCTIONS
    #
    #####

    # region HDF5 operations
    def get_ndi_table_by_table_key(self,df_key,hdf5_path):
        try:
            with h5py.File(hdf5_path, 'r') as hf:
                if df_key not in list(hf.keys()):
                    raise ValueError(f"Key {df_key} not found in HDF5 file")
                data = hf[df_key][:]

                cols = hf[df_key].attrs['columns']
                idx = hf[df_key].attrs['index']
            
                # Recreate the exact same DataFrame
                restored_df = pd.DataFrame(data, columns=cols, index=idx)
                return restored_df
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Error reading HDF5 file: {e}")
            raise e

    def read_ndi_tables_from_hdf5(self, h5_file_path):
        try:
            df_keys=self.ref_df['table_key'].to_list()
            self.dict_of_ndi_tables={}
            for df_key in df_keys:
                restored_ndi_table_df = self.get_ndi_table_by_table_key(df_key, h5_file_path)
                self.dict_of_ndi_tables[df_key] = restored_ndi_table_df
            
            if self.logger is not None:
                self.logger.info(f"Loaded {len(self.dict_of_ndi_tables)} NDI tables from HDF5")

            self.tables_keys=list(self.dict_of_ndi_tables.keys())
            #our assumption is that all the ndi tables have the same wavelength columns
            self.wavelengths=list(self.dict_of_ndi_tables[self.tables_keys[0]].columns)

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Error reading NDI tables from HDF5: {e}")
            raise e

    # endregion
    
    # region Regression evaluation

    
    @override
    def evaluate_regression_models(
        self,
        X, 
        y,
        target,
        split=False,
        test_size=0.2):

        # set the model
        model= LinearRegression()
        
        #  Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if(split):
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X,y)
            y_pred = model.predict(X)
            y_test=y

        r2 = r2_score(y_test,y_pred)
        rmse = root_mean_squared_error(y_test,y_pred)
        
        return r2, rmse
    
    # Get X and y from the NDI tables
    def get_X_and_y(self):
        sample_keys = list(self.dict_of_ndi_tables.keys())
        first_key = sample_keys[0]
        wavelength_labels = self.dict_of_ndi_tables[first_key].columns

        # 2. Extract the values from each DataFrame and stack them
        # We use a list comprehension to get the underlying numpy values (.values)
        # np.stack creates the 3rd dimension (Samples, WL, WL)
        ndi_cube = np.stack([self.dict_of_ndi_tables[k].values for k in sample_keys])

        if self.logger is not None:
            self.logger.info(f"Cube Shape: {ndi_cube.shape}") 
        
        # 3. Get the target values (assuming they are in the same order as the keys)
        y_values = self.ref_df[self.target].values
        
        return ndi_cube, y_values, wavelength_labels
    
    # Compute R² and RMSE for all NDI combinations
    def compute_scores_for_ndi_cube(self):
        
        ndi_cube, y_values, wavelength_labels = self.get_X_and_y()
        _, num_wl, _ = ndi_cube.shape
        r2_matrix = np.full((num_wl, num_wl), np.nan)
        rmse_matrix = np.full((num_wl, num_wl), np.nan)

        split=self.split_dataset_to_train_and_test
        if self.logger is not None:
            if(split):
                self.logger.info(f"Fitting model with split data.")
            else:
                self.logger.info(f"Fitting model without split data.")

        for i in range(num_wl):
            for j in range(num_wl):
                if i >= j: continue  # Skip duplicates and diagonals
                
                # Slicing the cube: "Give me all samples for this specific WL pair"
                # X shape will be (num_samples, 1)
                X = ndi_cube[:, i, j].reshape(-1, 1)
                
                r2, rmse = self.evaluate_regression_models(
                    X,
                    y_values, 
                    target=self.target,
                    split=split
                    )
                r2_matrix[i, j] = r2
                rmse_matrix[i, j] = rmse
        
        return r2_matrix, rmse_matrix, wavelength_labels

    def compute_rmse_score_for_ndi_cube(self):
        _, rmse_matrix,wavelength_labels = self.compute_scores_for_ndi_cube()
        self.rmse_results_df = pd.DataFrame(
            rmse_matrix, 
            index=wavelength_labels, 
            columns=wavelength_labels
        )
        return self.rmse_results_df

    def compute_r2_score_for_ndi_cube(self):
        r2_matrix, _ , wavelength_labels= self.compute_scores_for_ndi_cube()

        self.r2_results_df = pd.DataFrame(
            r2_matrix, 
            index=wavelength_labels, 
            columns=wavelength_labels
        )
        return self.r2_results_df
        
    def get_best_ndi_combination_r2(self):
        if self.r2_results_df is None:
            raise ValueError("r2_results_df is not set. Run compute_r2_score_for_ndi_cube first.")
        # Find the maximum value in the matrix
        best_r2 = np.nanmax(self.r2_results_df.values)

        # Find the indices of that maximum value
        idx1, idx2 = np.unravel_index(np.nanargmax(self.r2_results_df.values), self.r2_results_df.values.shape)

        best_wl1 = self.r2_results_df.columns[idx1]
        best_wl2 = self.r2_results_df.index[idx2]

        return best_wl1, best_wl2, best_r2
    
    def get_best_ndi_combination_rmse(self):
        if self.rmse_results_df is None:
            raise ValueError("rmse_results_df is not set. Run compute_rmse_score_for_ndi_cube first.")
        # Find the minimum value in the matrix
        best_rmse = np.nanmin(self.rmse_results_df.values)

        # Find the indices of that minimum value
        idx1, idx2 = np.unravel_index(np.nanargmin(self.rmse_results_df.values), self.rmse_results_df.values.shape)

        best_wl1 = self.rmse_results_df.columns[idx1]
        best_wl2 = self.rmse_results_df.index[idx2]

        return best_wl1, best_wl2, best_rmse

    # endregion



    # region CSV operations
    def read_ndi_tables_from_csv(self):
        raise NotImplementedError("read_ndi_tables_from_csv method not implemented yet")
    # endregion

    # region plotting
    def plot_r2_results(self,show_plots: bool = False):
        if self.r2_results_df is None:
            raise ValueError("r2_results_df is not set. Run evaluate_ndi_combinations first.")
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.r2_results_df, cmap='viridis')
        plt.title("R² Scores for all NDI Combinations")
        plt.xlabel("Wavelength 2 (nm)")
        plt.ylabel("Wavelength 1 (nm)")
        if show_plots:
            plt.show()
        return plt
    
    def plot_rmse_results(self,show_plots: bool = False):
        if self.rmse_results_df is None:
            raise ValueError("rmse_results_df is not set. Run evaluate_ndi_combinations first.")
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.rmse_results_df, cmap='viridis')
        plt.title("RMSE Scores for all NDI Combinations")
        plt.xlabel("Wavelength 2 (nm)")
        plt.ylabel("Wavelength 1 (nm)")
        if show_plots:
            plt.show()
        return plt
    
    # endregion