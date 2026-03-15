import smbclient
from smbclient.shutil import copyfile

from pathlib import Path

import os
import shutil
import cv2
import pandas as pd
from spectral import *
import spectral as sp

# my files
from image_processing.rgb_image import rgb_image
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
    
    def create_dataset(self):
        if self.data_source == "server":
            return self._create_dataset_from_server()
        else:
            return self._create_dataset_from_local()
    

    def _compute_hsv_values(self,img_path, bb_df, bb_df_idx):
        """
        Compute HSV values for each bounding box in the dataframe.
        
        Args:
            img_path: Path to the image file
            bb_df: DataFrame containing bounding box information
            idx: Index of the bounding box in the dataframe
            
        Returns:
            DataFrame with added HSV columns
        """
        try:
            

            index= bb_df_idx
            img_bgr=cv2.imread(img_path)
            if img_bgr is None:
                self.logger.error(f"Failed to load image: {img_path}")
                return bb_df

            #create rgb_image object
            rgb_img=rgb_image(img_bgr,bb_df,index,self.config)

            #compute hsv values
            hm,sm,vm=rgb_img.get_masked_hsv_obj_means()
            hstd,sstd,vstd=rgb_img.get_masked_hsv_obj_stds()
            hmed,smed,vmed=rgb_img.get_masked_hsv_obj_medians()


            #save hsv values 
            bb_df.at[index,'huePlantMean']=hm
            bb_df.at[index,'satPlantMean']=sm
            bb_df.at[index,'valPlantMean']=vm
            bb_df.at[index,'huePlantStd']=hstd
            bb_df.at[index,'satPlantStd']=sstd
            bb_df.at[index,'valPlantStd']=vstd
            bb_df.at[index,'huePlantMedian']=hmed
            bb_df.at[index,'satPlantMedian']=smed
            bb_df.at[index,'valPlantMedian']=vmed
        
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise
        
        return bb_df
    
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
            

            #iterate over the images and compute hsv values
            imgs_names=bb_df['image_name'].values.tolist()
            for i in range(len(imgs_names)):
                img_name= imgs_names[i]

                idx=bb_df.loc[bb_df['image_name'] == img_name].index.item()
                img_path = remote_files_path / img_name

                copyfile(img_path, local_rgb_folder_path / img_name)
                bb_df=self._compute_hsv_values(local_rgb_folder_path / img_name, bb_df, idx)
                
        except Exception as e:
                    self.logger.error(f"An unexpected error occurred: {e}")
                    self.logger.error(e.__traceback__())
        finally:
            self.logger.info("Saving RGB dataset to CSV...")
            self.save_rgb_ds_to_csv(bb_df)
            # Reset the connection cache if needed, especially in scripts
            smbclient.reset_connection_cache()
            self.logger.info("SMB connection cache reset.") 
            shutil.rmtree(local_rgb_folder_path)
            self.logger.info(f"Local RGB folder removed: {local_rgb_folder_path}")
            self.logger.info("RGB dataset creation completed successfully.")
            self.rgb_ds = bb_df
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
            self.logger.info(f"Processing image folder: {img_folder_name}")
            images=os.listdir(f"{rgb_imgs_root_path}/{img_folder_name}/images")
            bb_df = pd.read_csv(f"{rgb_imgs_root_path}/{img_folder_name}/{self.annotation_file_name}")


            for i in range(len(images)):
                img_name= images[i]

                indices = bb_df.index[bb_df['image_name'] == img_name].tolist()

                if len(indices) > 0:
                   idx=bb_df.loc[bb_df['image_name'] == img_name].index.item()
                   img_path = rgb_imgs_root_path / img_folder_name / "images" / img_name
                   bb_df=self._compute_hsv_values(img_path, bb_df, idx)
                else:
                   self.logger.warning(f"Image name {img_name} does not exist in the dataframe for folder {img_folder_name}.")

                     
            current_bb_df = bb_df.copy()

            if len(self.rgb_ds) == 0:
                self.rgb_ds = current_bb_df
            else:
                self.rgb_ds = pd.concat([self.rgb_ds, current_bb_df], ignore_index=True)
                
            # Save RGB dataset to CSV
            self.logger.info("Saving RGB dataset to CSV...")
            self.save_rgb_ds_to_csv(current_bb_df,img_folder_name)
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