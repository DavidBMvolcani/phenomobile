import smbclient
from smbclient.shutil import copyfile

import os
import shutil
import cv2
import pandas as pd
from spectral import *
import spectral as sp

# my files
from image_processing.hyper_spectral_image import HyperSpectralImage
from core.datasets_creation.dataset_creation import DatasetCreation


class HyperSpectralDsCreation(DatasetCreation):
    # PARAMETERS:
    #  -annotation_file_name: the name of the annotation file (e.g "bounding boexs of plants")
    #  -split: whether the images splited to objects
    #  -ndi_tuple: NDI wavelength tuple as "wl1,wl2" (e.g., "583.85,507.56")
    def __init__(self,
        logger,
        map_hs_and_th_ds,
        annotation_file_name,
        split_image_to_objects,
        home_dir,
        rotate_image,
        download_folder,
        formatted_datetime,
        data_source,
        ndi_tuple=None,
        COMPUTE_NDI=False,
        SMB_USERNAME=None,
        SMB_PASSWORD=None,
        SMB_SERVER=None,
        SMB_SHARE=None,
        RAW_DATA_FOLDER=None,
        YEAR_DIR_NAME=None,
        DATE_DIR_NAME=None):

        self.logger = logger
        self.home_dir = home_dir
        self.download_folder = download_folder
        self.formatted_datetime = formatted_datetime
        self.annotation_file_name = annotation_file_name
        self.split_image_to_objects = split_image_to_objects
        self.rotate_image = rotate_image
        self.COMPUTE_NDI = COMPUTE_NDI

        self.data_source = data_source
        if self.data_source=='server':
            self.SMB_USERNAME = SMB_USERNAME
            self.SMB_PASSWORD = SMB_PASSWORD
            self.SMB_SERVER = SMB_SERVER
            self.SMB_SHARE = SMB_SHARE
            self.RAW_DATA_FOLDER = RAW_DATA_FOLDER
            self.YEAR_DIR_NAME = YEAR_DIR_NAME
            self.DATE_DIR_NAME = DATE_DIR_NAME

        # Initialize empty DataFrame for spectral data
        self.spectral_img_df=pd.DataFrame()
       
        self.map_hs_and_th_ds = map_hs_and_th_ds
        if self.map_hs_and_th_ds:
            # we create folder for gray imgs of the hs imgs and use then in SIFT algorithm further 
            gray_for_Hs_imgs="Grays_imgs_attched_with_Hs_imgs"
            gray_for_HS_imgs_directory_path=f"{self.home_dir}/{self.download_folder}/{gray_for_Hs_imgs}/{self.formatted_datetime}"
            os.makedirs(gray_for_HS_imgs_directory_path, exist_ok=True)
            self.gray_for_HS_imgs_directory_path=gray_for_HS_imgs_directory_path
        
        self.ndi_tuple = ndi_tuple
        if self.COMPUTE_NDI:
            self.setup_ndi_table_path()


    def create_dataset(self):

        if self.data_source == 'server':
            self._create_dataset_from_server()
        elif self.data_source == 'local':
            self._create_dataset_from_local()
        else:
            raise ValueError("data_source must be either 'server' or 'local'")

    #######
    '''
     Tree diagram of the hyper spectral folder in the server
        <PROJECT NAME>
            <YEAR>
                 <DATE>
                     <HS>
                        <NUMBER X1>
                        <NUMBER X2>
                        […]
                        <NUMBER Xn>
        <<csv file of the bounding boxes>> (optional)
    '''
    def _create_dataset_from_server(self):
        #The implemention below based on the raw data store in the server
        try:
            smbclient.ClientConfig(username=self.SMB_USERNAME, password=self.SMB_PASSWORD)
            # Construct the full remote path
            remote_dir_path = f"\\\\{self.SMB_SERVER}\\{self.SMB_SHARE}\\{self.RAW_DATA_FOLDER}"
            
            # read a file from a path in the server 
            Hs='HS'
            HS_dir=f"{remote_dir_path}\\{self.YEAR_DIR_NAME}\\{self.DATE_DIR_NAME}\\{Hs}"
            if self.annotation_file_name is not None:
                remote_path=f"{remote_dir_path}\\{self.YEAR_DIR_NAME}\\{self.DATE_DIR_NAME}"
                annotation_path=f"{remote_path}\\{self.annotation_file_name}"
                
                local_annotation_folder=f'{self.download_folder}/annotation_folder'
                os.makedirs(local_annotation_folder, exist_ok=True)
                local_annotation_file_path=f'{local_annotation_folder}/{self.annotation_file_name}'

                # store the annotation remote file in local folder
                smbclient.register_session(
                    server=self.SMB_SERVER,
                    username=self.SMB_USERNAME,
                    password=self.SMB_PASSWORD,
                )
                path = fr"\\{annotation_path}"
                with smbclient.open_file(path, mode="rb") as f:
                    annot_df = pd.read_csv(f)
                annot_df.to_csv(local_annotation_file_path)

            #read the hs images
            HS_img_dirs = smbclient.listdir(HS_dir)
            for idx,img_dir in enumerate(HS_img_dirs):
                img_name=img_dir
                HS_img_dir=HS_dir+f'\\{img_dir}'
                # enter to the directory of the calibrated image (cal)
                cal="results"
                cal_HS_img_dir=HS_img_dir+f'\\{cal}'
                hdr_file_name="REFLECTANCE_"+img_dir+".hdr"
                
                self.logger.info(f"Attempting to read from folder: {cal_HS_img_dir}")

                
                #download the folder of the current hs image
                #(the header file need all the folder to get to the metadata parameter)
                local_hdr_img_path=self.download_folder+"/"+img_name+"/"+hdr_file_name
                hs_local_folder=self.download_folder+"/"+img_name
                smbclient.shutil.copytree(
                    src=cal_HS_img_dir,
                    dst=hs_local_folder,
                    username=self.SMB_USERNAME,
                    password=self.SMB_PASSWORD,
                )

                if self.map_hs_and_th_ds:
                    # get the rgb image atthced to hs image  in different path
                    # and save it in gray format for futured actions (sift algorithm)
                    rgb_file_name=f'RGBSCENE_{img_name}.png'
                    rgb_file_path=f'{hs_local_folder}/{rgb_file_name}'
                    image = cv2.imread(rgb_file_path)
                    img_gray_of_hs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    local_gray_img_path=f'{gray_for_HS_imgs_directory_path}/{img_name}.png'
                    cv2.imwrite(local_gray_img_path,img_gray_of_hs)
           
                #get metedate parameters
                hdr = sp.envi.open(local_hdr_img_path)
                img = hdr.load()
                meta = hdr.metadata
                RGB_bands=meta['default bands']
                wl=meta['wavelength']
                ad=meta['acquisition date']
        
                #create an object of the vegetation_indices class
                img_num=img_name

                #set argument to hs_img object ant create it
                hs_img=HyperSpectralImage(
                    img,
                    wl,
                    RGB_bands,
                    img_num,
                    ROTATE= self.rotate_image,
                    ANNOTATION_PATH=local_annotation_file_path,
                    SPLIT_IMAGE_TO_OBJECTS=self.split_image_to_objects,
                    acquisition_date=ad,
                    longitude=meta['longitude'],
                    latitude=meta['latitude'],
                    ndi_tuple=self.ndi_tuple,
                    create_ndi_table=self.COMPUTE_NDI,
                    ndi_table_directory_path=self.ndi_table_directory_path) 
                #compute vegetation indices and save it in dataframe
                self.spectral_img_df=hs_img.compute_vegeation_indices(self.spectral_img_df)
                
                # delete the download image folder 
                shutil.rmtree(self.download_folder+"/"+img_dir)
                
            # delete the annotation folder
            if self.ANNOTATION is not None: 
                shutil.rmtree(local_annotation_folder)
            
            
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            self.logger.error(e.__traceback__())
        finally:
            self.logger.info("Saving hyperspectral dataset to CSV...")
            self.save_hs_ds_to_csv()
            
            # Reset the connection cache if needed, especially in scripts
            smbclient.reset_connection_cache()
            self.logger.info("SMB connection cache reset.")

            if self.map_hs_and_th_ds:
                # get the gray images of the gray img related to the hs imgs
                gray_for_HS_imgs=os.listdir(gray_for_HS_imgs_directory_path)
                self.gray_for_HS_imgs=gray_for_HS_imgs

                return self.spectral_img_df,self.gray_for_HS_imgs 
            else:
                return self.spectral_img_df,None

    def _create_dataset_from_local(self):
        #The implemention below based on the raw data store in the local machine
        raise NotImplementedError("Local data source is not implemented yet")

    def setup_ndi_table_path(self):
        # Common initialization for both paths
        rmt_folder=self.RAW_DATA_FOLDER
        if self.COMPUTE_NDI and not self.ndi_table_directory_path:
            ndi_path = f'{self.dataset_folder}/{rmt_folder}_ndi_tables'
            os.makedirs(ndi_path, exist_ok=True)
            self.ndi_table_directory_path = ndi_path
        else:
            self.ndi_table_directory_path = self.ndi_table_directory_path


    def save_hs_ds_to_csv(self):
        datasets_paths=self.dataset_folder
        rmt_folder=self.RAW_DATA_FOLDER
        dt=self.formatted_datetime
        self.spectral_img_df.to_csv(f"{datasets_paths}/hyper_sp_imgs_dataset_created_at_{dt}_from_{rmt_folder}_{dt}.csv",index=False)
        self.logger.info(f'new file created at : {datasets_paths}/hyper_sp_imgs_dataset_created_at_{dt}_from_{rmt_folder}_{dt}.csv')

      #load hyper-spectral_df and set the path to the gray images assosiated with the hs images
    def load_hs_df(self,hs_df_path,
                   gray_for_HS_imgs_directory_path=None):
        self.spectral_img_df=pd.read_csv(hs_df_path)
        if gray_for_HS_imgs_directory_path is not None:
            self.gray_for_HS_imgs_directory_path=gray_for_HS_imgs_directory_path
            self.gray_for_HS_imgs=os.listdir(gray_for_HS_imgs_directory_path)
        self.logger.info(f'loaded file : {hs_df_path}')
        return self.spectral_img_df