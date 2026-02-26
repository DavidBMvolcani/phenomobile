import smbclient
from smbclient.shutil import copyfile

import os
import shutil

from io import BytesIO
from dotenv import load_dotenv

from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import pandas as pd
import datetime as dt
import time

from spectral import *
import spectral as sp
import math

from scipy.stats import gaussian_kde
from scipy import stats
import json
import pdb

# my files

from image_processing.hyper_spectral_image import hyper_spectral_image
from image_processing.image_processing import image_processing
from image_processing.thermal_image import thermal_image
from image_processing.rgb_image import rgb_image



class dataset_creation:
    # Constructor method, called when a new object is created
    
    # PARAMETERS:
    # FlirImageExtractor_path: path for flir image extractor for thermal images
    # config: ConfigManager instance for accessing configuration
    # CREATE : boolean parameter indicate whether to create dataset from raw data
    # HS: boolean parameter indicate whether source of dataset
    #     include hyper-spectral images.
    # TH: boolean parameter indicate whether source of dataset
    #     include Thearmal images.
    # compute_ndi: boolean parameter  releveant for HS images
    # ndi_tuple : tuple parameter  releveant for HS images
    def __init__(self, FlirImageExtractor_path, config=None, ENV_FILE=True, CREATE=True,
                 HS=False, TH=False, RGB=False,
                 create_ndi_table=False, ndi_table_directory_path=None,
                 ndi_tuple=None):
        
        #process FLIR thermal image with exiftool
        self.FlirImageExtractor_path=FlirImageExtractor_path
        sys.path.append(FlirImageExtractor_path)
        import flir_image_extractor
        self.fir = flir_image_extractor.FlirImageExtractor()
        
        # Initialize datetime attributes (always needed, regardless of CREATE flag)
        self.date = dt.datetime.now().strftime("%Y%m%d")
        self.formatted_datetime = dt.datetime.now().strftime("%Y-%m-%d %H.%M.%S")

        if config is not None:
            # Store the config object for later use
            self.config = config
            
            # Use ConfigManager for configuration
            self.SMB_USERNAME = config.get('SMB_USERNAME')
            self.SMB_PASSWORD = config.get('SMB_PASSWORD')
            self.SMB_SERVER = config.get('SMB_SERVER')
            self.SMB_SHARE = config.get('SMB_SHARE')
            
            # Get experiment settings from config
            experiment_settings = config.get('experiment_settings', {})
            self.REMOTE_FOLDER = experiment_settings.get('REMOTE_FOLDER')
            self.year = experiment_settings.get('year')
            self.date = experiment_settings.get('date')
            
            # Get paths from config
            paths = config.get('paths', {})
            self.download_folder = paths.get('download_folder')
            dataset_folder = experiment_settings.get('dataset_folder')

            self.dataset_folder = config.get('datasets_path')
           
            
            # Get other parameters
            self.SPLIT = experiment_settings.get('SPLIT_IMAGE_TO_OBJECTS')
            self.ANNOTATION = experiment_settings.get('ANNOTATION_FILE')
            
        elif ENV_FILE:
            # Fallback to environment variables (legacy support)
            load_dotenv(override=True, dotenv_path='.env')
            self.SMB_USERNAME = os.environ.get("SMB_USERNAME")
            self.SMB_PASSWORD = os.environ.get("SMB_PASSWORD")
            self.SMB_SERVER = os.environ.get("SMB_SERVER")
            self.SMB_SHARE = os.environ.get("SMB_SHARE")
            
            self.REMOTE_FOLDER = os.environ.get("REMOTE_FOLDER")
            self.year = os.environ.get("year")
            self.date = os.environ.get("date")
            
            self.download_folder = os.environ.get("download_folder")
            self.dataset_folder = os.environ.get("dataset_folder")
            
            self.SPLIT = os.environ.get("SPLIT_IMAGE_TO_OBJECTS")
            self.ANNOTATION = os.environ.get("ANNOTATION_FILE")
        
        # Common initialization for both paths
        self.ndi_tuple = ndi_tuple
        self.COMPUTE_NDI = create_ndi_table
        if create_ndi_table and not ndi_table_directory_path:
            ndi_path = f'{self.dataset_folder}/ndi_tables'
            os.makedirs(ndi_path, exist_ok=True)
            self.ndi_table_directory_path = ndi_path
        else:
            self.ndi_table_directory_path = ndi_table_directory_path
        
        # Get current date and time
        current_datetime = dt.datetime.now()
        self.formatted_datetime = current_datetime.strftime("%Y-%m-%d %H")
        self.cd = os.getcwd()
        
        if CREATE:
            if HS:
                self.create_hs_ds()
            if TH:
                self.create_Theraml_ds()
            if RGB:
                self.create_RGB_ds() 
            if HS and TH:
                self.map_hyper_spectral_and_thermal_datasets()
    
    # create hyper-spectrl dataset based on the hs images store in the server
    #
    # PARAMETERS:
    #  -SPILT: whether the images splited to objects.
    def create_hs_ds(self):
        annotation_file_name=self.ANNOTATION
        SPLIT= self.SPLIT
        
        #create dataframe
        self.spectral_img_df=pd.DataFrame()
        spectral_img_df=self.spectral_img_df
        
        # create folder for the downloaded HS_imgs
        gray_for_Hs_imgs="Grays_imgs_attched_with_Hs_imgs"
        gray_for_HS_imgs_directory_path=f"{self.cd}/{self.download_folder}/{gray_for_Hs_imgs}/{self.formatted_datetime}"
        os.makedirs(gray_for_HS_imgs_directory_path, exist_ok=True)
        gray_for_HS_imgs=os.listdir(gray_for_HS_imgs_directory_path)
        self.gray_for_HS_imgs=gray_for_HS_imgs
        self.gray_for_HS_imgs_directory_path=gray_for_HS_imgs_directory_path

        #The implemention below based on the raw data store in the server
        try:
            smbclient.ClientConfig(username=self.SMB_USERNAME, password=self.SMB_PASSWORD)
            # Construct the full remote path
            remote_dir_path = f"\\\\{self.SMB_SERVER}\\{self.SMB_SHARE}\\{self.REMOTE_FOLDER}"
            
            # read a file from a path in the server 
            Hs='HS'
            HS_dir=f"{remote_dir_path}\\{self.year}\\{self.date}\\{Hs}"
            if annotation_file_name is not None:
                remote_path=f"{remote_dir_path}\\{self.year}\\{self.date}"
                annotation_path=f"{remote_path}\\{annotation_file_name}"
                
                local_annotation_folder=f'{self.download_folder}/annotation_folder'
                os.makedirs(local_annotation_folder, exist_ok=True)
                local_annotation_file_path=f'{local_annotation_folder}/{annotation_file_name}'

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
                print(f"Attempting to read from folder: {cal_HS_img_dir}")

                
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
                ANNOTATION_PATH=local_annotation_file_path
                hs_img=hyper_spectral_image.hyper_spectral_image(img,wl,RGB_bands,img_num,ANNOTATION_PATH=local_annotation_file_path,
                        SPLIT_IMAGE_TO_OBJECTS=SPLIT,acquisition_date=ad,longitude=meta['longitude'],
                        latitude=meta['latitude'],ndi_tuple=self.ndi_tuple,create_ndi_table=self.COMPUTE_NDI,
                        ndi_table_directory_path=self.ndi_table_directory_path) 
                #compute vegetation indices and save it in dataframe
                spectral_img_df=hs_img.compute_vegeation_indices(spectral_img_df)
                
                # delete the download image folder 
                shutil.rmtree(self.download_folder+"/"+img_dir)
                
            # delete the annotation folder
            if self.ANNOTATION is not None: 
                shutil.rmtree(local_annotation_folder)
            
            self.spectral_img_df=spectral_img_df
           
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print(e.__traceback__())
        finally:
            self.save_hs_ds_to_csv()
            # Reset the connection cache if needed, especially in scripts
            smbclient.reset_connection_cache()
            print("SMB connection cache reset.")

    def create_Theraml_ds(self):
        #locals assignments 
        cd=self.cd
        download_folder=self.download_folder
        formatted_datetime=self.formatted_datetime
        SMB_USERNAME=self.SMB_USERNAME
        SMB_PASSWORD=self.SMB_PASSWORD
        SMB_SERVER=self.SMB_SERVER
        SMB_SHARE=self.SMB_SHARE
        REMOTE_FOLDER=self.REMOTE_FOLDER
        year=self.year
        date=self.date
        
        gray_for_Th_imgs="Grays_imgs_attched_with_Th_imgs"
        gray_for_Th_imgs_directory_path=f"{cd}/{download_folder}/{gray_for_Th_imgs}/{formatted_datetime}"
        os.makedirs(gray_for_Th_imgs_directory_path, exist_ok=True)
        gray_for_Th_imgs=os.listdir(gray_for_Th_imgs_directory_path)
        self.gray_for_Th_imgs=gray_for_Th_imgs
        self.gray_for_Th_imgs_directory_path=gray_for_Th_imgs_directory_path

        self.thermal_df=pd.DataFrame()
        thermal_df=self.thermal_df

        try:
            smbclient.ClientConfig(username=SMB_USERNAME, password=SMB_PASSWORD)
            # Construct the full remote path
            remote_dir_path = f"\\\\{SMB_SERVER}\\{SMB_SHARE}\\{REMOTE_FOLDER}"
        
            
            FLIR="FLIR"
            remote_files_path = f"{remote_dir_path}\\{year}\\{date}\\{FLIR}"
            
        
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
                print(f"Attempting to read from folder: {remote_img_path}")
                
                # Copy a local file to image dirctory
                local_img_path=download_folder+"/"+img_name
                copyfile(remote_img_path,dst=local_img_path)
        
                #####################
                # extract from the flir image the rgb inner image
                # then, convert it to gray for future actions (sift algorithm)
        
                self.fir.process_image(local_img_path)
                rgb_image=self.fir.get_rgb_np()
                img_gray_of_th = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                gary_img_name=img_name.split(".")[0]+".png"
                cv2.imwrite(gray_for_Th_imgs_directory_path+"/"+gary_img_name,img_gray_of_th)
        
                ###################
                
                # create object of thermal image and df of parameters for all the images
                th_img=Thermal_img.Thermal_img(img_name,local_img_path,self.FlirImageExtractor_path)
                thermal_df=th_img.compute_Theraml_parameters(thermal_df,img_name)

                
                #delete the local image
                os.remove(local_img_path)
                
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print(e.__traceback__())
        finally:
            self.save_th_ds_to_csv()
            # Reset the connection cache if needed, especially in scripts
            smbclient.reset_connection_cache()
            print("SMB connection cache reset.")
    
    def create_RGB_ds(self):
        #locals assignments 
        cd=self.cd
        download_folder=self.download_folder
        formatted_datetime=self.formatted_datetime
        SMB_USERNAME=self.SMB_USERNAME
        SMB_PASSWORD=self.SMB_PASSWORD
        SMB_SERVER=self.SMB_SERVER
        SMB_SHARE=self.SMB_SHARE
        REMOTE_FOLDER=self.REMOTE_FOLDER
        year=self.year
        date=self.date

        annotation_file_name=self.ANNOTATION
        try:
            smbclient.ClientConfig(username=SMB_USERNAME, password=SMB_PASSWORD)
            # Construct the full remote path
            remote_dir_path = f"\\\\{SMB_SERVER}\\{SMB_SHARE}\\{REMOTE_FOLDER}"

            remote_path=f"{remote_dir_path}\\{year}\\{date}"
            annotation_path=f"{remote_path}\\{annotation_file_name}"
            path = fr"\\{annotation_path}"
            with smbclient.open_file(path, mode="rb") as f:
                bb_df=pd.read_csv(f)

            idx=len(bb_df.columns)
            
            #add relevant columns
            # bb_df.insert(idx,'plantBinImg',None)
            bb_df.insert(idx,'huePlantMean',None)
            bb_df.insert(idx,'satPlantMean',None)
            bb_df.insert(idx,'valPlantMean',None)

            #configuration
            RGB="RGB"
            remote_files_path = f"{remote_dir_path}\\{year}\\{date}\\{RGB}"
            local_rgb_folder_path=f'{cd}/{download_folder}/local_rgb_folder'
            os.makedirs(local_rgb_folder_path, exist_ok=True)
            

            #iterate over the images and compute hsv values
            imgs_names=bb_df['image_name'].values.tolist()
            for i in range(len(imgs_names)):
                img_name= imgs_names[i]
                remote_img_path=remote_files_path+f"\\{img_name}" 
                print(f"Attempting to read from folder: {remote_img_path}")
                
                # Copy a local file to image directory
                local_img_path=f'{local_rgb_folder_path}/{img_name}'
                copyfile(remote_img_path,dst=local_img_path)

                index=bb_df.index[i]
                img_bgr=cv2.imread(f'{local_img_path}')
                rgb_img=rgb_image.rgb_image(img_bgr,bb_df,index)
                hm,sm,vm=rgb_img.get_hsv_obj_means()

                #save hsv values 
                #bb_df.at[index,'plantBinImg']= rgb_img.get_binary_obj_img()
                bb_df.at[index,'huePlantMean']=hm
                bb_df.at[index,'satPlantMean']=sm
                bb_df.at[index,'valPlantMean']=vm
        
        except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    print(e.__traceback__())
        finally:
            self.save_rgb_ds_to_csv(bb_df)
            # Reset the connection cache if needed, especially in scripts
            smbclient.reset_connection_cache()
            print("SMB connection cache reset.") 
        shutil.rmtree(local_rgb_folder_path)
        
    ################
    #
    # MERGE DATASETS - merge hyperspectral and thermal images
    #
    ################
    
    # sometime mistakenly the photograhpher take more then one image for
    # the same sence - so we filterd out those images
    def remove_duplicated_images(self,data_,flann,threshold=0.2):
        imgs_names=list(data_.keys()).copy()
        data_copy= data_.copy()
        img_name=imgs_names.pop()
        suspected_images=[]
        cnt,lenght=0,len(imgs_names)
        while len(imgs_names)>0:
            cnt=cnt+1
            print(f"go over {cnt/lenght :.2f} from the images")
            kp1, des1=data_copy[img_name]
            #remove the current img_name from the dict
            data_copy.pop(img_name) 
            scores=[]
            for b, (kp2, des2) in data_copy.items():
                matches = flann.knnMatch(des1, des2, k=2)
                good_matches=[]
                for m, n in matches:
                    if m.distance < 0.75 * n.distance: 
                        good_matches.append(m)
                #the score in normalized in the lenght of des1
                cur_score = len(good_matches)/len(des1)
                if cur_score>threshold:
                    suspected_images.append(b)
                    imgs_names.remove(b)
            #pop another image
            img_name=imgs_names.pop()
        return suspected_images
        
    # we map hyper-spectral and thermal images by using SIFT algorithm. 
    # The SIFT algorithm applied on the gray images created for HS, and the 
    # gary images for thermal images erlier.
    def map_hyper_spectral_and_thermal_datasets(self,score_th=0.02):
               
        sift = cv2.SIFT_create()

        # match between hs and theraml images by SIFT algorithm
        # for each gray image attched to hs image - we found the correspond gray image attached to thermal image
        # that have more 'good matches' form the other rgb images.
        
        Grays_for_HS_imgs=self.gray_for_HS_imgs 
        Grays_for_Th_imgs=self.gray_for_Th_imgs
        gray_for_HS_imgs_directory_path=self.gray_for_HS_imgs_directory_path
        gray_for_Th_imgs_directory_path=self.gray_for_Th_imgs_directory_path
        
        # set two grops: A for the set with the more images and B to the other
        A=Grays_for_HS_imgs if len(Grays_for_HS_imgs)>len(Grays_for_Th_imgs) else Grays_for_Th_imgs
        A_path=gray_for_HS_imgs_directory_path if A==Grays_for_HS_imgs  else gray_for_Th_imgs_directory_path
        B=Grays_for_Th_imgs if A==Grays_for_HS_imgs else Grays_for_HS_imgs
        B_path=gray_for_Th_imgs_directory_path  if A==Grays_for_HS_imgs else gray_for_HS_imgs_directory_path
        
        # set df for the matches
        if(A==Grays_for_HS_imgs):
            df_columns=['gray_img_name_for_hs','gray_img_name_for_th']
        else:
            df_columns=['gray_img_name_for_th','gray_img_name_for_hs']

        # FLANN matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # compute descriptors of A
        A_data = {}
        for A_img_name in A:
            A_img_path=f'{A_path}/{A_img_name}'
            A_img=cv2.imread(A_img_path)
            kpA, desA = sift.detectAndCompute(A_img, None)
            A_data[A_img_name] = (kpA, desA)
        # compute descriptors of B
        B_data = {}
        for B_img_name in B:
            B_img_path=f'{B_path}/{B_img_name}'
            B_img=cv2.imread(B_img_path)
            kpB, desB = sift.detectAndCompute(B_img, None)
            B_data[B_img_name] = (kpB, desB)

        #remove duplicated
        lst=self.remove_duplicated_images(B_data,flann)
        B_data={key: val for (key,val) in B_data.items() if key not in lst}
        lst=self.remove_duplicated_images(A_data,flann)
        A_data={key: val for (key,val) in A_data.items() if key not in lst}

        #match between A and B
        img_matches_df=pd.DataFrame(columns=df_columns+['score'])
        cnt,lenght=0,len(A_data.items())
        for  img_a, (kpA, desA) in A_data.items():
            cnt=cnt+1
            print(f"go over {cnt/lenght :.2f} from the images") 
            max_score,corr_img=0,''
            for b, (kpB, desB) in B_data.items():  
                matches = flann.knnMatch(desA, desB, k=2)
                #ratio-test: if the best match (m) is much better (significantly smaller distance)
                # than the second-best match (n), the match is likely distinctive and reliable.
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance: # Adjust ratio as needed
                        good_matches.append(m)
                #the score in normalized in the lenght of des1
                score = len(good_matches)/len(desA)
                if max_score<score:
                    max_score=score
                    best_match_image=b
            img_matches_df.loc[len(img_matches_df)]=[img_a,best_match_image,max_score]

        # we set here a score therhshold for matching the images
        # only image that have matching with max-score higher the thershold
        # will consider as matching. otherwise it attached by herself with no matching.
        filter_df=img_matches_df[img_matches_df['score']>score_th].reset_index(drop=True)
        filter_df=filter_df.sort_values(by="gray_img_name_for_hs").reset_index(drop=True)
        filter_df=filter_df.drop('score',axis=1)

        #compute image without matching
        matches_df_imgs_for_hs=list(img_matches_df['gray_img_name_for_hs'].values)
        matches_df_imgs_for_th=list(img_matches_df['gray_img_name_for_th'].values)
        filter_df_imgs_for_hs=list(filter_df['gray_img_name_for_hs'].values)
        filter_df_imgs_for_th=list(filter_df['gray_img_name_for_th'].values)
        
        remained_hs_imgs=list(set(matches_df_imgs_for_hs)-set(filter_df_imgs_for_hs))
        remained_th_imgs=list(set(matches_df_imgs_for_th)-set(filter_df_imgs_for_th))
        
        
        #append the remaing images
        for img in remained_hs_imgs:
            filter_df.loc[len(filter_df)]=[img,None]
        for img in remained_th_imgs:
            filter_df.loc[len(filter_df)]=[None,img]
        
        
        filter_df=filter_df.sort_values(by='gray_img_name_for_hs')
        filter_df=filter_df[['gray_img_name_for_hs','gray_img_name_for_th']]
        
        # Replace all NaN values with an empty string
        filter_df = filter_df.fillna('')
        
        # rename the columns name 
        sp_df=self.spectral_img_df.rename(columns=lambda col: f'hs_{col}')
        th_df=self.thermal_df.rename(columns=lambda col: f'th_{col}')
        
        #remove images extenstions 
        filter_df['gray_img_name_for_hs']=filter_df['gray_img_name_for_hs'].apply(lambda x: x.split('.')[0] if x is not None else None)
        filter_df['gray_img_name_for_th']=filter_df['gray_img_name_for_th'].apply(lambda x: x.split('.')[0] if x is not None else None)
        th_df['th_image_name']=th_df['th_image_name'].apply(lambda x: x.split('.')[0])
        sp_df['hs_img_num']=sp_df['hs_img_num'].apply(lambda x: str(x))
        
        #merge the sp_df with the sorted_df
        sp_df_merged_with_filter_df=sp_df.merge(filter_df,left_on="hs_img_num",
                                                     right_on="gray_img_name_for_hs", how="left")
        
        # merge the above created df with thermal df 
        merged_img_df=sp_df_merged_with_filter_df.merge(th_df,left_on="gray_img_name_for_th",
                                                                       right_on="th_image_name",how="left")
        
        self.merged_img_df=merged_img_df.drop(['gray_img_name_for_hs','gray_img_name_for_th'], axis=1)

        self.save_mergerd_img_df()

    ##################
    #
    # MERGE DATASETS -merge reference and hyperparameter datasets
    #
    ##################

    # GOAL:
        # merge two datasets. The coomon Use of this function 
        # is for merge the reference dataset with the hyper parameter dataset
        # because the process of mergeing is vary by the kind of project
        # we apply for each project specefic merger function
    # PARAMETERS
        #   - hp_ds_path : the path to the dataset that contain parameters 
        #     that proceced from the raw data like vegetation indices 
        #   - ref_ds_path: path to dataset that contain the traget object
        #   - project_name : string parameter that indictate the kind of project
        #   -config_file_path: path to json configuration file

    def merge_datasets(self,hp_ds_path,ref_ds_path,project_name,config_file_path):
        """
        Merge datasets based on project type.
        For lettuce projects, delegate to lettuce_dataset class.
        """
        if project_name=='lettuce_BENI_ATAROT':
            from datasets.lettuce_dataset import lettuce_dataset
            lettuce_ds = lettuce_dataset(
                 FlirImageExtractor_path=self.FlirImageExtractor_path,
                 config=self.config,  # Pass the ConfigManager instance
                 ENV_FILE=False,  # # Use ConfigManager instead of env file
                 CREATE=False
            )
                

            lettuce_ds.merge_datasets(hp_ds_path, ref_ds_path, config_file_path)
            self.completed_df = lettuce_ds.completed_df
        
        
    #########################
    #
    # SAVE and LOAD functions 
    #
    #########################
    def save_hs_ds_to_csv(self):
        datasets_paths=self.dataset_folder
        rmt_folder=self.REMOTE_FOLDER
        date=self.date
        dt=self.formatted_datetime
        self.spectral_img_df.to_csv(f"{datasets_paths}/hyper_sp_imgs_dataset_created_at_{date}_from_{rmt_folder}_{dt}.csv",index=False)
        print(f'new file created at : {datasets_paths}/hyper_sp_imgs_dataset_created_at_{date}_from_{rmt_folder}_{dt}.csv')

    def save_th_ds_to_csv(self):
        datasets_paths=self.dataset_folder
        rmt_folder=self.REMOTE_FOLDER
        date=self.date
        dt=self.formatted_datetime
        self.thermal_df.to_csv(f"{datasets_paths}/thermal_imgs_dataset_{rmt_folder}_{dt}.csv",index=False)
        print(f'new file created : {datasets_paths}/thermal_imgs_dataset_{rmt_folder}_{dt}.csv')
    
    def save_rgb_ds_to_csv(self,rgb_df):
        datasets_paths=self.dataset_folder
        rmt_folder=self.REMOTE_FOLDER
        date=self.date
        dt=self.formatted_datetime
        rgb_df.to_csv(f"{datasets_paths}/rgb_imgs_dataset_{rmt_folder}_{dt}.csv",index=False)
        print(f'new file created : {datasets_paths}/rgb_imgs_dataset_{rmt_folder}_{dt}.csv')

    def save_mergerd_img_df(self):
        datasets_paths=self.dataset_folder
        rmt_folder=self.REMOTE_FOLDER
        date=self.date
        dt=self.formatted_datetime
        self.merged_img_df.to_csv(f"{datasets_paths}/hs_and_th_imgs_dataset_{rmt_folder}_{dt}.csv",index=False)
        print(f'new file created : {datasets_paths}/hs_and_th_imgs_dataset_{rmt_folder}_{dt}.csv')

    def save_mergerd_reference_and_hyperParameter_ds(self,reference_dataset_name=''):
        datasets_paths=self.dataset_folder
        dt=self.formatted_datetime
        self.completed_df.to_csv(f"{datasets_paths}/complete_ds_of_{reference_dataset_name}_{dt}.csv",index=False)
        print(f'new file created :{datasets_paths}/complete_ds_of_{reference_dataset_name}_{dt}.csv')

    #load hyper-spectral_df and set the path to the gray images assosiated with the hs images
    def load_hs_df(self,hs_df_path,
                   gray_for_HS_imgs_directory_path=None):
        self.spectral_img_df=pd.read_csv(hs_df_path)
        if gray_for_HS_imgs_directory_path is not None:
            self.gray_for_HS_imgs_directory_path=gray_for_HS_imgs_directory_path
            self.gray_for_HS_imgs=os.listdir(gray_for_HS_imgs_directory_path)
        return self.spectral_img_df

    def load_th_df(self,th_df_path,
                  gray_for_Th_imgs_directory_path=None):
        self.thermal_df=pd.read_csv(th_df_path)
        if gray_for_Th_imgs_directory_path is not None:
            self.gray_for_Th_imgs_directory_path=gray_for_Th_imgs_directory_path
            self.gray_for_Th_imgs=os.listdir(gray_for_Th_imgs_directory_path)
        return self.thermal_df
    
    # DELETE floders functions
    def delete_locals_grays_images_folders(self):
        shutil.rmtree(self.gray_for_HS_imgs_directory_path)
        shutil.rmtree(self.gray_for_Th_imgs_directory_path)
        