import smbclient
from smbclient.shutil import copyfile
import concurrent.futures
from smbclient import listdir, open_file

import os
import shutil
import cv2
import pandas as pd
from spectral import *
import spectral as sp
import h5py
import time

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
        dataset_folder,
        formatted_datetime,
        data_source,
        ndi_tuple=None,
        COMPUTE_NDI=False,
        ndi_table_directory_path=None,
        ndi_storage_method=None,

        SMB_USERNAME=None,
        SMB_PASSWORD=None,
        SMB_SERVER=None,
        SMB_SHARE=None,
        RAW_DATA_FOLDER=None,
        YEAR_DIR_NAME=None,
        DATE_DIR_NAME=None,
        
        object_filter_method=None,
        ndvi_threshold=None,
        hsv_filter_thresholds=None,

        ):
       # self.dataset_folder

        self.logger = logger
        self.home_dir = home_dir
        self.download_folder = download_folder
        self.dataset_folder = dataset_folder
        self.formatted_datetime = formatted_datetime
        self.annotation_file_name = annotation_file_name
        self.split_image_to_objects = split_image_to_objects
        self.rotate_image = rotate_image
        self.COMPUTE_NDI = COMPUTE_NDI
        self.ndi_table_directory_path=ndi_table_directory_path
        self.ndi_storage_method=ndi_storage_method
        self.object_filter_method = object_filter_method
        self.ndvi_threshold = ndvi_threshold
        self.hsv_filter_thresholds = hsv_filter_thresholds
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
        #self.ndi_tables_df=pd.DataFrame()
       
        self.map_hs_and_th_ds = map_hs_and_th_ds
        if self.map_hs_and_th_ds:
            # we create folder for gray imgs of the hs imgs and use then in SIFT algorithm further 
            gray_for_Hs_imgs="Grays_imgs_attched_with_Hs_imgs"
            gray_for_HS_imgs_directory_path=f"{self.home_dir}/{self.download_folder}/{gray_for_Hs_imgs}/{self.formatted_datetime}"
            os.makedirs(gray_for_HS_imgs_directory_path, exist_ok=True)
            self.gray_for_HS_imgs_directory_path=gray_for_HS_imgs_directory_path
        
        self.ndi_tuple = ndi_tuple
        if self.COMPUTE_NDI:

            if self.ndi_storage_method==None:
                raise ValueError ("ndi table storage method cannot be None")
            if self.ndi_storage_method=='hdf5':
                #create hdf5 file
                dt = self.formatted_datetime
                ndi_table_directory_path = self.ndi_table_directory_path
                self.h5_file_path=os.path.join(ndi_table_directory_path, f'ndi_tables_{dt}.h5')

                # prepare it to read the dataframe 
                self.spectral_img_df['NDI_df'] = None
                self.spectral_img_df['NDI_df'] = self.spectral_img_df['NDI_df'].astype(object)
    

    def save_hs_ds_to_hdf5(self,hs_ds_file_name=None):
        if hs_ds_file_name is not None:
            ndi_table_directory_path = self.ndi_table_directory_path
            file_name=f'ndi_tables_for_{hs_ds_file_name}.h5'
            self.h5_file_path=os.path.join(ndi_table_directory_path, file_name)
        
        # Creates keys like 'table_0', 'table_1', 'table_2'...
        size=len(self.spectral_img_df)
        table_keys = [f"table_{i}_{time.time_ns()}" for i in range(size)]
        self.spectral_img_df['table_key'] = table_keys
        
        # Extract all NDI tables at once (much faster!)
        ndi_tables = []
        for _, row in self.spectral_img_df.iterrows():
            ndi_tables.append(row['NDI_df'][0])
        
        # Batch write to HDF5 (single operation)
        with h5py.File(self.h5_file_path, 'w') as hf:
            # Create all datasets at once
            for i, (table_key, ndi_table) in enumerate(zip(table_keys, ndi_tables)):
                col_names = ndi_table.columns.astype(str).tolist()
                row_names = ndi_table.index.astype(str).tolist()
                
                # Single HDF5 operation
                ds = hf.create_dataset(table_key,
                                 data=ndi_table.values, 
                                 compression="gzip",
                                 chunks=True)  # Enable chunking
                
                # Save metadata efficiently
                ds.attrs['columns'] = col_names
                ds.attrs['index'] = row_names
        
        self.logger.info(f"Saved {len(ndi_tables)} NDI tables to {self.h5_file_path}")
        
        # Clean up - drop the ndi_table from the dataframe
        self.spectral_img_df.drop('NDI_df', axis=1, inplace=True)
    
    def check_annotation_file_validation(self, annot_df):
        issues = []
    
        for image_name in annot_df['image_name'].unique():
            labels = annot_df[annot_df['image_name'] == image_name]['label_name']
            duplicate_labels = labels[labels.duplicated()]
            
            if len(duplicate_labels) > 0:
                issues.append({
                    'image_name': image_name,
                    'duplicate_labels': duplicate_labels.tolist()
                })
        
        if issues:
            for issue in issues:
                self.logger.error(f"  - {issue['image_name']}: {issue['duplicate_labels']}")
            raise ValueError (f"Found {len(issues)} images with duplicate labels:")   
        
       

    def create_dataset(self):

        if self.data_source == 'server':
            spectral_img_df,gray_for_HS_imgs = self._create_dataset_from_server()
        elif self.data_source == 'local':
            spectral_img_df,gray_for_HS_imgs = self._create_dataset_from_local()
        else:
            raise ValueError("data_source must be either 'server' or 'local'")
        
        return spectral_img_df,gray_for_HS_imgs

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
            
            # read a file from a path in server
            Hs='HS'
            HS_dir=f"{remote_dir_path}\\{self.YEAR_DIR_NAME}\\{self.DATE_DIR_NAME}\\{Hs}"
            if self.annotation_file_name is not None:
                remote_path=f"{remote_dir_path}\\{self.YEAR_DIR_NAME}\\{self.DATE_DIR_NAME}"
                annotation_path=f"{remote_path}\\{self.annotation_file_name}"
                
                local_annotation_folder=f'{self.download_folder}/annotation_folder'
                os.makedirs(local_annotation_folder, exist_ok=True)
                local_annotation_file_path=f'{local_annotation_folder}/{self.annotation_file_name}'
                
                # store annotation remote file in local folder
                try:
                    self.logger.info(f"Connecting to SMB server: {self.SMB_SERVER}")
                    smbclient.register_session(
                        server=self.SMB_SERVER,
                        username=self.SMB_USERNAME,
                        password=self.SMB_PASSWORD,
                    )
                    self.logger.info(f"Reading annotation file from: {annotation_path}")
                    with smbclient.open_file(annotation_path, mode="rb") as f:
                        annot_df = pd.read_csv(f)
                    self.logger.info(f"Saving annotation file to: {local_annotation_file_path}")
                    annot_df.to_csv(local_annotation_file_path)
                    self.check_annotation_file_validation(annot_df)
                    self.logger.info("Annotation file downloaded successfully")


                except Exception as e:
                    self.logger.error(f"Failed to download annotation file: {e}")
                    self.logger.error(f"Annotation error location: {e.__traceback__()}")

            #read the hs images
            try:
                self.logger.info(f"Listing HS image directories in: {HS_dir}")
                HS_img_dirs = smbclient.listdir(HS_dir)
                self.logger.info(f"Found {len(HS_img_dirs)} HS image directories")
            except Exception as e:
                self.logger.error(f"Failed to list HS directories: {e}")
                self.logger.error(f"HS listing error location: {e.__traceback__()}")
                return
            
            if not HS_img_dirs:
                self.logger.error("No HS image directories found")
                return
            
            for idx,img_dir in enumerate(HS_img_dirs):
                try:
                    img_name=img_dir
                    HS_img_dir=HS_dir+f'\\{img_dir}'
                    self.logger.info(f"Processing HS directory {idx+1}/{len(HS_img_dirs)}: {img_dir}")
                    
                    # enter to directory of calibrated image (cal)
                    cal="results"
                    cal_HS_img_dir=HS_img_dir+f'\\{cal}'
                    hdr_file_name="REFLECTANCE_"+img_dir+".hdr"
                except Exception as e:
                    self.logger.error(f"Failed to process HS directory {img_dir}: {e}")
                    self.logger.error(f"HS directory error location: {e.__traceback__()}")
                    continue
                
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
                #self.fast_smb_copytree(cal_HS_img_dir, hs_local_folder)

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
                    ndi_tuple=self.ndi_tuple,
                    create_ndi_table=self.COMPUTE_NDI,
                    ndi_table_directory_path=self.ndi_table_directory_path,
                    ndi_storage_method=self.ndi_storage_method,
                    ANNOTATION_PATH=local_annotation_file_path,
                    SPLIT_IMAGE_TO_OBJECTS=self.split_image_to_objects,
                    acquisition_date=ad,
                    longitude=meta['longitude'],
                    latitude=meta['latitude'],
                    object_filter_method=self.object_filter_method,
                    ndvi_threshold=self.ndvi_threshold,
                    hsv_filter_thresholds=self.hsv_filter_thresholds
                    ) 
                self.logger.info(f"Processing hyperspectral image {img_name}")
                #compute vegetation indices and save it in dataframe
                self.spectral_img_df=hs_img.compute_vegeation_indices(
                    self.spectral_img_df)
                
                
                
                # delete the download image folder 
                shutil.rmtree(self.download_folder+"/"+img_dir)
                
            # delete the annotation folder
            if self.annotation_file_name is not None:
                shutil.rmtree(local_annotation_folder)
            
            self.logger.info("Saving hyperspectral dataset to CSV...")

            # create keys to map between the spectral_img_df and h5 file
            if self.COMPUTE_NDI and self.ndi_storage_method=='hdf5':
                #save the hyperspectral dataset to hdf5
                hs_ds_file_name=self.set_hs_ds_file_name()
                self.save_hs_ds_to_hdf5(hs_ds_file_name)
                
            # save the hyperspectral dataset to csv
            self.save_hs_ds_to_csv(hs_ds_file_name)
            
            # map 
            if self.map_hs_and_th_ds:
                # get the gray images of the gray img related to the hs imgs
                gray_for_HS_imgs=os.listdir(gray_for_HS_imgs_directory_path)
                self.gray_for_HS_imgs=gray_for_HS_imgs

                return self.spectral_img_df,self.gray_for_HS_imgs 
            else:
                return self.spectral_img_df,None

        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            self.logger.error(f"Error location: {e.__traceback__}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error args: {e.args}")
            self.logger.error(f"Error in function: {sys._getframe().f_code.co_name}")
            self.logger.error(f"Error line: {sys._getframe().f_lineno}")
            self.logger.error(f"Error file: {__file__}")
            
        finally:
            
            # delete the download image folder 
            shutil.rmtree(self.download_folder)
            # Reset the connection cache if needed, especially in scripts
            smbclient.reset_connection_cache()
            self.logger.info("SMB connection cache reset.")

            

    def _create_dataset_from_local(self):
        #The implemention below based on the raw data store in the local machine
        raise NotImplementedError("Local data source is not implemented yet")

   
    def set_hs_ds_file_name(self):
        rmt_folder=f'{self.RAW_DATA_FOLDER}_{self.DATE_DIR_NAME}'
        dt=self.formatted_datetime
        hs_ds_file_name=f"hyper_sp_imgs_dataset_created_at_{dt}_from_{rmt_folder}"
        return hs_ds_file_name

    def save_hs_ds_to_csv(self,hs_ds_file_name=None):
        if hs_ds_file_name is None:
            hs_ds_file_name=self.set_hs_ds_file_name()
        datasets_paths=self.dataset_folder
        rmt_folder=f'{self.RAW_DATA_FOLDER}_{self.DATE_DIR_NAME}'
        dt=self.formatted_datetime
        self.spectral_img_df.to_csv(f"{datasets_paths}/{hs_ds_file_name}.csv",index=False)
        self.logger.info(f'new file created at : {datasets_paths}/{hs_ds_file_name}')

      #load hyper-spectral_df and set the path to the gray images assosiated with the hs images
    def load_hs_df(self,hs_df_path,
                   gray_for_HS_imgs_directory_path=None):
        self.spectral_img_df=pd.read_csv(hs_df_path)
        if gray_for_HS_imgs_directory_path is not None:
            self.gray_for_HS_imgs_directory_path=gray_for_HS_imgs_directory_path
            self.gray_for_HS_imgs=os.listdir(gray_for_HS_imgs_directory_path)
        self.logger.info(f'loaded file : {hs_df_path}')
        return self.spectral_img_df