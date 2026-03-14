import smbclient
from smbclient.shutil import copyfile

import os
import shutil
import cv2
import pandas as pd
from spectral import *
import spectral as sp

# my files
from image_processing.thermal_image import thermal_image
from core.datasets_creation.dataset_creation import DatasetCreation

  # PARAMETERS:
# FlirImageExtractor_path: path for flir image extractor for thermal images
class ThermalDsCreation(DatasetCreation):
    def __init__(self,
        logger,
        map_hs_and_th_ds, 
        home_dir, 
        download_folder, 
        formatted_datetime,
        data_source,
        FlirImageExtractor_path,
        SMB_USERNAME=None, 
        SMB_PASSWORD=None, 
        SMB_SERVER=None, 
        SMB_SHARE=None, 
        RAW_DATA_FOLDER=None, 
        YEAR_DIR_NAME=None, 
        DATE_DIR_NAME=None):
        
        self.map_hs_and_th_ds = map_hs_and_th_ds
        self.home_dir=home_dir
        self.download_folder=download_folder
        self.formatted_datetime=formatted_datetime
        self.data_source = data_source
        self.logger = logger
        
        self.SMB_USERNAME = SMB_USERNAME
        self.SMB_PASSWORD = SMB_PASSWORD
        self.SMB_SERVER = SMB_SERVER
        self.SMB_SHARE = SMB_SHARE
        self.RAW_DATA_FOLDER = RAW_DATA_FOLDER
        self.YEAR_DIR_NAME = YEAR_DIR_NAME
        self.DATE_DIR_NAME = DATE_DIR_NAME
        
        self.FlirImageExtractor_path = FlirImageExtractor_path
        try:
            sys.path.append(self.FlirImageExtractor_path)
            import flir_image_extractor
            self.fir = flir_image_extractor.FlirImageExtractor()
        except ImportError as e:
            raise ImportError(f"Failed to import flir_image_extractor from {self.FlirImageExtractor_path}: {e}")
        
        if self.map_hs_and_th_ds:
            gray_for_Th_imgs="Grays_imgs_attched_with_Th_imgs"
            gray_for_Th_imgs_directory_path=f"{self.home_dir}/{self.download_folder}/\
                {gray_for_Th_imgs}/{self.formatted_datetime}"
            os.makedirs(gray_for_Th_imgs_directory_path, exist_ok=True)
            self.gray_for_Th_imgs_directory_path=gray_for_Th_imgs_directory_path

        # Initialize empty DataFrame for thermal data
        self.thermal_df=pd.DataFrame()

        

    def create_dataset(self):
        if self.data_source == 'server':
            self._create_dataset_from_server()
        elif self.data_source == 'local':
            self._create_dataset_from_local()
        else:
            raise ValueError("data_source must be either 'server' or 'local'")
            
    def _create_dataset_from_server(self):
          
        try:
            smbclient.ClientConfig(username=self.SMB_USERNAME, password=self.SMB_PASSWORD)
            # Construct the full remote path
            remote_dir_path = f"\\\\{self.SMB_SERVER}\\{self.SMB_SHARE}\\{self.RAW_DATA_FOLDER}"
        
            FLIR="FLIR"
            remote_files_path = f"{remote_dir_path}\\{self.YEAR_DIR_NAME}\\{self.DATE_DIR_NAME}\\{FLIR}"
            
        
            files = smbclient.listdir(remote_files_path)
            #filter out files that are not images
            image_extensions = [
            # Raster formats (pixel-based)
            'jpg', 'jpeg', 'jfif', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'heic', 'heif', 'avif']
            images=[ f for f in files if f.split('.')[1] in image_extensions ]
                
            # go over the images and extract relevant parameters
            for i in range(len(images)):
                img_name=images[i]
                remote_img_path=remote_files_path+f"\\{img_name}" 
                self.logger.info(f"Attempting to read from folder: {remote_img_path}")
                
                # Copy a local file to image dirctory
                local_img_path=self.download_folder+"/"+img_name
                copyfile(remote_img_path,dst=local_img_path)
        
                #####################
                # extract from the flir image the rgb inner image
                # then, convert it to gray for future actions (sift algorithm)

                if self.map_hs_and_th_ds:
                    self.logger.info(f"Processing image: {local_img_path}")
                    self.fir.process_image(local_img_path)
                    rgb_image=self.fir.get_rgb_np()
                    img_gray_of_th = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                    gary_img_name=img_name.split(".")[0]+".png"
                    cv2.imwrite(gray_for_Th_imgs_directory_path+"/"+gary_img_name,img_gray_of_th)
        
                ###################
                
                # create object of thermal image and df of parameters for all the images
                th_img=thermal_image.thermal_image(img_name,local_img_path,self.FlirImageExtractor_path)
                self.thermal_df=th_img.compute_Theraml_parameters(self.thermal_df,img_name)
                
                #delete the local image
                os.remove(local_img_path)
                
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            self.logger.error(e.__traceback__())
        finally:
            self.logger.info("Saving thermal dataset to CSV...")
            self.save_th_ds_to_csv()
            # Reset the connection cache if needed, especially in scripts
            smbclient.reset_connection_cache()
            self.logger.info("SMB connection cache reset.")
            if self.map_hs_and_th_ds:
               self.gray_for_Th_imgs=os.listdir(self.gray_for_Th_imgs_directory_path)

               return self.thermal_df,self.gray_for_Th_imgs 
            else:
               return self.thermal_df,None
        
    def _create_dataset_from_local(self):
        raise NotImplementedError("Creating dataset from local is not implemented yet")

    def save_th_ds_to_csv(self):
        datasets_paths=self.dataset_folder
        rmt_folder=self.RAW_DATA_FOLDER
        dt=self.formatted_datetime
        self.thermal_df.to_csv(f"{datasets_paths}/thermal_imgs_dataset_{rmt_folder}_{dt}.csv",index=False)
        print(f'new file created : {datasets_paths}/thermal_imgs_dataset_{rmt_folder}_{dt}.csv')

    def load_th_df(self,th_df_path,
                  gray_for_Th_imgs_directory_path=None):
        self.thermal_df=pd.read_csv(th_df_path)
        if gray_for_Th_imgs_directory_path is not None:
            self.gray_for_Th_imgs_directory_path=gray_for_Th_imgs_directory_path
            self.gray_for_Th_imgs=os.listdir(gray_for_Th_imgs_directory_path)
        self.logger.info(f'loaded file : {th_df_path}')
        return self.thermal_df