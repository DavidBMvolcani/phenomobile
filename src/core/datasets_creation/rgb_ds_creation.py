import smbclient
from smbclient.shutil import copyfile

from pathlib import Path

import os
import shutil
import cv2
import pandas as pd
from spectral import *
import spectral as sp
import pickle

# my files
from image_processing.rgb_image import RgbImage
from core.datasets_creation.dataset_creation import DatasetCreation

class RgbDsCreation (DatasetCreation):
    def __init__(self,
        logger,
        home_dir, 
        download_folder, 
        formatted_datetime,
        data_source,
        annotation_file_name,
        dataset_folder,
        SMB_USERNAME=None, 
        SMB_PASSWORD=None, 
        SMB_SERVER=None, 
        SMB_SHARE=None, 
        RAW_DATA_FOLDER=None, 
        server_year_dir_name=None, 
        server_date_dir_name=None,
        config=None):

        # Initialize parent class
        super().__init__(logger)
        
        #locals assignments 
        self.home_dir=home_dir
        self.download_folder=download_folder
        self.formatted_datetime=formatted_datetime
        self.annotation_file_name=annotation_file_name
        self.dataset_folder=dataset_folder
        self.SMB_USERNAME=SMB_USERNAME
        self.SMB_PASSWORD=SMB_PASSWORD
        self.SMB_SERVER=SMB_SERVER
        self.SMB_SHARE=SMB_SHARE
        self.RAW_DATA_FOLDER=RAW_DATA_FOLDER
        self.server_year_dir_name=server_year_dir_name
        self.server_date_dir_name=server_date_dir_name
        self.data_source=data_source
        self.config=config

        #check if pickle mask is the method to get the mask
        rgb_params=self.config.get("RGB_dataset_creation_parameters", {})
        mask_config=rgb_params.get("mask_computation_method", {})
        method = mask_config.get("method")
        if method =='pickle_masks':
            #check if pickle mask path is provided
            if self.config.get('pickle_mask_path', None) is None:
                raise ValueError("pickle_mask_path must be provided")
           
    
    def create_dataset(self):
        if self.data_source == "server":
            return self._create_dataset_from_server()
        else:
            return self._create_dataset_from_local()
    

    def get_pickle_df(self, images_date):
        pickle_mask_path = self.config.get('pickle_mask_path', None)
        if pickle_mask_path is None:
            raise ValueError("pickle_mask_path must be provided")
        pickle_folder=pickle_mask_path
        
        if images_date is None:
            raise ValueError("the date of the images must be provided")
        pickle_files_names= os.listdir(pickle_folder)
        pickle_match_lst=[f for f in pickle_files_names if images_date in f]
        if len(pickle_match_lst)==0:
            raise ValueError("No pickle file found in the pickle mask folder")
        elif len(pickle_match_lst)>1:
            raise ValueError("Multiple pickle files found in the pickle mask folder")
        
        pickle_file_name=pickle_match_lst[0]
        self.logger.info(f"Loading full pickle file: {pickle_file_name}")
        full_pickle_df= pd.read_pickle(os.path.join(
            pickle_folder, pickle_file_name))
        return full_pickle_df
    
    def _compute_mask_from_pickle(self,bb_df, full_pickle_df,bb_df_idx):
        #check that the row in bb_df and pickle file match
        row_bb_df = bb_df.loc[bb_df_idx]
        row_pickle_df = full_pickle_df.loc[bb_df_idx]
        row_pickle_df_without_mask = row_pickle_df.drop('mask')

        try:
            assert row_bb_df.equals(row_pickle_df_without_mask), "Content of bb_df and pickle file do not match"
        except AssertionError as e:
            self.logger.error(f"Error: {e}")
            self.logger.error(f"Row in bb_df: {row_bb_df}")
            self.logger.error(f"Row in pickle file: {row_pickle_df_without_mask}")
            raise e

        #load mask
        pickle_obj = row_pickle_df['mask']
        mask = pickle.loads(pickle_obj)
        return mask


    def _compute_hsv_values(self,img_path, bb_df, bb_df_idx, images_date, mask=None,res_df=None):
        """
        Compute HSV values for each bounding box in the dataframe.
        
        Args:
            img_path: Path to the image file
            bb_df: DataFrame containing bounding box information
            bb_df_idx: Index of the bounding box in the dataframe
            images_date: Date of the images
            mask: Optional mask to use for the bounding box
            res_df: DataFrame to append the results to
        Returns:
            DataFrame with added HSV columns
        """

        
        try:
            
            index= bb_df_idx
            img_bgr=cv2.imread(img_path)
            if img_bgr is None:
                self.logger.error(f"Failed to load image: {img_path}")
                return None

            #create rgb_image object
            rgb_img=RgbImage(
                img_bgr=img_bgr,
                bb_df=bb_df,
                df_index=bb_df_idx,
                config=self.config,
                pickle_mask=mask,
                bb_date=images_date,
            )

            #compute hsv values
            hm,sm,vm=rgb_img.get_masked_hsv_obj_means()
            hstd,sstd,vstd=rgb_img.get_masked_hsv_obj_stds()
            hmed,smed,vmed=rgb_img.get_masked_hsv_obj_medians()

       

            #save hsv values 
            res_df.at[index,'huePlantMean']=hm
            res_df.at[index,'satPlantMean']=sm
            res_df.at[index,'valPlantMean']=vm
            res_df.at[index,'huePlantStd']=hstd
            res_df.at[index,'satPlantStd']=sstd
            res_df.at[index,'valPlantStd']=vstd
            res_df.at[index,'huePlantMedian']=hmed
            res_df.at[index,'satPlantMedian']=smed
            res_df.at[index,'valPlantMedian']=vmed
        
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while computing HSV values: {e}")
            raise
        
        return res_df
    
    '''
    Tree diagram of the rgb folder in the server
        <PROJECT NAME>
            <YEAR>
                 <DATE>
                     <RGB>
                        <NUMBER X1>
                        <NUMBER X2>
                        […]
                        <NUMBER Xn>
        <<csv file of the bounding boxes>> (optional)
    '''
    def _create_dataset_from_server(self):
        try:
            smbclient.ClientConfig(username=self.SMB_USERNAME, password=self.SMB_PASSWORD)
            # Construct the full remote path
            remote_dir_path = f"\\\\{self.SMB_SERVER}\\{self.SMB_SHARE}\\{self.RAW_DATA_FOLDER}"

            remote_path=f"{remote_dir_path}\\{self.server_year_dir_name}\\{self.server_date_dir_name}"
            annotation_path=f"{remote_path}\\{annotation_file_name}"
            path = fr"\\{annotation_path}"
            with smbclient.open_file(path, mode="rb") as f:
                bb_df=pd.read_csv(f)

            
            #configuration
            RGB="RGB"
            # 1. Define the remote path 
            remote_files_path = Path(remote_dir_path) / self.server_year_dir_name / self.server_date_dir_name / RGB

            # 2. Define the local path
            local_rgb_folder_path = Path(self.home_dir) / self.download_folder / 'local_rgb_folder'     
            os.makedirs(local_rgb_folder_path, exist_ok=True)
            
            #load pickle file
            self.full_pickle_df = self.get_pickle_df(self.server_date_dir_name)
            

            res_df = bb_df.copy()
            #iterate over the images and compute hsv values
            imgs_names=bb_df['image_name'].values.tolist()
            for i in range(len(imgs_names)):
                img_name= imgs_names[i]

                idx=bb_df.loc[bb_df['image_name'] == img_name].index.item()
                img_path = remote_files_path / img_name

                copyfile(img_path, local_rgb_folder_path / img_name)
                image_dates=self.server_date_dir_name
                rgb_params=self.config.get("RGB_dataset_creation_parameters", {})
                if rgb_params.get("mask_computation_method", {}).get("method") == 'pickle_masks': 
                    mask=self._compute_mask_from_pickle(bb_df, self.full_pickle_df, idx)
                    res_df=self._compute_hsv_values(img_path, bb_df, idx, image_dates, mask, res_df)
                else:
                    res_df=self._compute_hsv_values(img_path, bb_df, idx, image_dates, None, res_df) 

                
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while creating RGB dataset from server: {e}")
            self.logger.error(e.__traceback__())
        finally:
            self.logger.info("Saving RGB dataset to CSV...")
            self.save_rgb_ds_to_csv(res_df, img_folder_name)
            # Reset the connection cache if needed, especially in scripts
            smbclient.reset_connection_cache()
            self.logger.info("SMB connection cache reset.") 
            shutil.rmtree(local_rgb_folder_path)
            self.logger.info(f"Local RGB folder removed: {local_rgb_folder_path}")
            self.logger.info("RGB dataset creation completed successfully.")
            self.rgb_ds = res_df
            return self.rgb_ds

    '''
    Tree structure:
    local_rgb_folder/
        img_folder_name/ (e.g., "3.12.25")
            annotation_file_name (e.g., "annotations.csv")
            images/
                image1.jpg
                image2.jpg
                ...
    '''
    def _create_dataset_from_local(self):

        # read the annotation file
        rgb_imgs_root_path = Path(f"{self.home_dir}/{self.download_folder}/local_rgb_folder")

        p = Path(rgb_imgs_root_path)
        folders = [f.name for f in p.iterdir() if f.is_dir()]

        self.rgb_ds = pd.DataFrame()

        for img_folder_name in folders:
            images_date=img_folder_name # the convention is that the folder name is the date
            self.logger.info(f"Processing image folder: {img_folder_name}")
            images=os.listdir(f"{rgb_imgs_root_path}/{img_folder_name}/images")
            bb_df = pd.read_csv(f"{rgb_imgs_root_path}/{img_folder_name}/{self.annotation_file_name}")

            #load pickle file
            self.full_pickle_df = self.get_pickle_df(images_date)
            
            res_df = bb_df.copy()
            for i in range(len(images)):
                img_name= images[i]
                indices = bb_df.index[bb_df['image_name'] == img_name].tolist()
                if len(indices) > 0:
                   idx=bb_df.loc[bb_df['image_name'] == img_name].index.item()
                   img_path = rgb_imgs_root_path / img_folder_name / "images" / img_name
                   rgb_params=self.config.get("RGB_dataset_creation_parameters", {})
                   if rgb_params.get("mask_computation_method", {}).get("method") == 'pickle_masks':
                      
                       mask=self._compute_mask_from_pickle(bb_df, self.full_pickle_df, idx)
                       res_df = self._compute_hsv_values(img_path, bb_df, idx, images_date, mask, res_df)
                   else:
                       res_df = self._compute_hsv_values(img_path, bb_df, idx, images_date, None, res_df)
                else:
                   self.logger.warning(f"Image name {img_name} does not exist in the dataframe for folder {img_folder_name}.")
                     
            current_img_res_df = res_df.copy()

            if len(self.rgb_ds) == 0:
                self.rgb_ds = current_img_res_df
            else:
                self.rgb_ds = pd.concat([self.rgb_ds, current_img_res_df], ignore_index=True)
                
            # Save RGB dataset to CSV
            self.logger.info("Saving RGB dataset to CSV...")
            self.save_rgb_ds_to_csv(current_img_res_df, img_folder_name)
            self.logger.info("RGB dataset creation completed successfully.")
            
        return self.rgb_ds


    
    def save_rgb_ds_to_csv(self,df,date=None):
        datasets_paths=self.dataset_folder
        dt=self.formatted_datetime if date is None else date
        df.to_csv(f"{datasets_paths}/rgb_imgs_dataset_{dt}.csv",index=False)
        self.logger.info(f'new file created : {datasets_paths}/rgb_imgs_dataset_{dt}.csv')

    def load_rgb_df(self,rgb_df_path):
        self.rgb_ds = pd.read_csv(rgb_df_path)
        self.logger.info(f'loaded file : {rgb_df_path}')
        return self.rgb_ds