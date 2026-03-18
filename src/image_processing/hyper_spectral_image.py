import spectral.graphics as graphics
from skimage.color import rgb2hsv
import numpy as np
import pandas as pd
import time
import cv2

#my files
from .image_processing import ImageProcessing 


#######
#
# the indices are taken from: Index DataBase (https://www.indexdatabase.de/)
#
#######

class HyperSpectralImage:
    # PARAMETERS:
    #     - img : hyper-spectral calibrated image
    #     - wl : list of the wavelenghts of the hyper-spectral image
    #     - RGB_bands: defualts bands of the hyper-spectral image
    #     - img_name: string- the number of the image
    #     - ROTATE : boolean variable indicate whether the original image is rotate 90 degree
    #     - save_lbi : boolean flag for saving leaves(object) binary image
    #     - lbi_directory_path: the local path for saving the binary img
    #     - ndi_tuple: tuple of (wl1,wl2) that used for compute specific ndi of wl1,wl2
    #     - create_ndi_table: boolean value indicate whether to save the ndi table as csv file
    #     - ndi_table_directory_path: path to the ndi table
    #     - ANNOTATION_PATH: path to annotation file, by defualt there are no need in annotation
    #     - SPLIT_IMAGE_TO_OBJECTS: boolean parameter
    #     - object_filter_method: one of this options: {'by _hsv','by_ndvi'}
    #     - hsv_filter_thresholds :
    
    def __init__(self,
            img,
            wl,
            RGB_bands,
            img_name,
            ROTATE=True,
            save_lbi=False,
            lbi_directory_path='/',
            ndi_tuple=None,
            create_ndi_table=False,
            ndi_table_directory_path=None,
            ANNOTATION_PATH="",
            SPLIT_IMAGE_TO_OBJECTS=False,
            acquisition_date="",
            longitude="",
            latitude="",
            object_filter_method=None,
            ndvi_threshold=None,
            hsv_filter_thresholds=None):

        self.img=img

        self.wl=wl
        self.img_name=img_name
        self.ROTATE=ROTATE
        self.acquisition_date=acquisition_date
        self.longitude=longitude
        self.latitude=latitude
        self.ndi_tuple=ndi_tuple
        self.COMPUTE_NDI=create_ndi_table
        self.ndi_table_directory_path=ndi_table_directory_path

        if object_filter_method=='by_hsv':
            hue_threshold=hsv_filter_thresholds['hue_threshold']
            saturation_threshold=hsv_filter_thresholds['saturation_threshold']
            self.leaves_binary_img=self.detect_leaves_by_hsv(
                RGB_bands,
                hue_threshold,
                saturation_threshold)
        elif object_filter_method=='by_ndvi':
            self.leaves_binary_img=self.detect_leaves_by_ndvi(
                ndvi_threshold)
        else:
            raise ValueError(f"Invalid object_filter_method: {object_filter_method}")

        if SPLIT_IMAGE_TO_OBJECTS and ANNOTATION_PATH:
            self.split_image_to_objects(img_name,ANNOTATION_PATH)
        self.image_splitted=SPLIT_IMAGE_TO_OBJECTS
       
        self.fvc=self.FVC()
        if save_lbi:
            self.save_leaves_binary_img(lbi_directory_path)
       

     
    
    # # FUNCTION GOAL: filter out the leaves from the image
    # explaination: our assumption is that the defualt  hue threshold in the function
    # is seperate the leaves from the soil in all (or at least most) the image.
    # The staturation thershold  eliminate the calibaration panel from the image
    #
    # RETRUN VALUE: leaves binary image: leaves pixel are denoted as "1"
    def detect_leaves_by_hsv(self,
        RGB_bands,
    hue_threshold,
    saturation_threshold):
        rgb_img=graphics.get_rgb(self.img,(RGB_bands))
        hsv_img = rgb2hsv(rgb_img)
        hue_img = hsv_img[:, :, 0]
        saturation_img=hsv_img[:, :, 1]
        binary_img = (hue_img > hue_threshold) & (saturation_img > saturation_threshold)
        return  binary_img

    def detect_leaves_by_ndvi(self,ndvi_threshold):
        ndvi_ = self.get_ndvi()
        return ndvi_>ndvi_threshold

    def split_image_to_objects(self,image_num,ANNOTATION_PATH):
        self.img_proc=ImageProcessing(
            binary_img=self.leaves_binary_img,
            img_num=image_num,
            annot_path=ANNOTATION_PATH,
            rotate=self.ROTATE)
        self.objs_df=self.img_proc.get_objs()

    #### fractional vegetation cover (FVC)
    # fractional vegetation cover (FVC) or leaf cover, represents the proportion of a given area (like a plot of land or a field of view) 
    #  that is covered by leaves
    #   parameter:
    #       - binary img : pixels with "1" are the pixels that higher the the hue threshold. 
    def FVC(self):
        # compute FCV from the entire image size
        return self.leaves_binary_img.sum()/self.leaves_binary_img.ravel().shape[0]
    

    # auxiliary func that find the bands numbers of the RGB signals
    def auxiliary_func(self,red_wl= 670,nir_wl=800,blue_wl=450):
        x_ticks=list(range(len(self.wl)))
        wl_dic=dict(zip(x_ticks,self.wl))
        wl_dic={item[0]:float(item[1]) for item in wl_dic.items()}
    
        # we find in the wavelenghts of the image the nearst wavelength for each of them.
        key_blue,val_blue=min(wl_dic.items(),key=lambda item:abs(item[1]-blue_wl))
        key_red,val_red = min(wl_dic.items(),key=lambda item:abs(item[1]-red_wl))
        key_nir,val_nir=min(wl_dic.items(),key=lambda item:abs(item[1]-nir_wl))
       
        return key_blue,key_red,key_nir
    
        #PARAMETERS:
        #
        # -SPILT: whether the images splited to objects.
        #        in this case we compute the ndvi for each objcted 
        # -leaves_bi_of_obj : the binary image of object. 
    def NDVI(self,red_wl= 670,nir_wl=800,SPLIT=False,leaves_bi_of_obj=""):
        self.ndvi_ = self.get_ndvi()
    
        # statistic measurement of the the ndvi of the leaves pixels
        if not SPLIT:
            leaves_pixels_ndvi=self.ndvi_[self.leaves_binary_img==True].ravel()
        else:
            leaves_pixels_ndvi=self.ndvi_[leaves_bi_of_obj==True].ravel()
            
        self.ndvi_mean=leaves_pixels_ndvi.mean()
        self.ndvi_std=leaves_pixels_ndvi.std()
        self.ndvi_median=np.median(leaves_pixels_ndvi)
        
        return self.ndvi_,self.ndvi_mean,self.ndvi_std,self.ndvi_median
    
    #Thus the GNDVI tends to be more sensitive to chlorophyll content and 
    #less sensitive to soil background than the NDVI.
    def GNDVI(self,green_wl= 550,nir_wl=800,SPLIT=False,leaves_bi_of_obj=""):
        return  self.NDVI(green_wl,nir_wl,SPLIT,leaves_bi_of_obj) 
        
    #designed to be more sensitive than the NDVI in dense vegetation (the NDVI tends to saturate), 
    #correct for atmospheric and soil background effects, 
    #and be more useful for high-biomass regions (e.g., forests)
    def EVI(self,SPLIT=False,leaves_bi_of_obj=""):
        key_blue,key_red,key_nir=self.auxiliary_func()
        blue=self.img[:,:,key_blue]
        red=self.img[:,:,key_red]
        nir=self.img[:,:,key_nir]

        numerator=2.5*(nir-red)
        denominator=nir+6*red -7.5*blue +1
        evi = numerator/denominator
    
        # statistic measurement of the  evi of the leaves pixels
        if not SPLIT:
            leaves_pixels=evi[self.leaves_binary_img==True].ravel()
        else:
            leaves_pixels=evi[leaves_bi_of_obj==True].ravel()
        
        mask = ~np.isinf(leaves_pixels)
        evi_mean=leaves_pixels[mask].mean()
            
        return evi,evi_mean
    
    # The Leaf Area Index (LAI) is a fundamental biophysical parameter used to describe vegetation.
    # It quantifies the total leaf area per unit of ground surface area. 
    # However, in more recent works, this has been estimated via the Enhanced Vegetation Index (EVI),
    # as it has been shown that they are empirically correlated
    
    def LAI(self,SPLIT=False,leaves_bi_of_obj=""):
        evi,evi_mean=self.EVI()
        lai=3.618*evi-0.118
        # statistic measurement of the leaves pixels
        if not SPLIT:
            leaves_pixels=lai[self.leaves_binary_img==True].ravel()
        else:
            leaves_pixels=lai[leaves_bi_of_obj==True].ravel()
        
        lai_mean=leaves_pixels.mean()
        return lai,lai_mean
    
    # the wavelengths in the spectral image bands are float numbers
    # so we find the nearest wavelenth value to the given wavelength  
    def means_ndi(self,wl1,wl2,wl_dic,SPLIT=False,leaves_bi_of_obj=""):
        key_wl1,val_wl1 = min(wl_dic.items(),key=lambda item:abs(item[1]-wl1))
        key_wl2,val_wl2=min(wl_dic.items(),key=lambda item:abs(item[1]-wl2))
    
        band1=self.img[:,:,key_wl1]
        band2=self.img[:,:,key_wl2]
    
        ndi= (band1 - band2) / (band1 + band2)
    
        #get the statistic metric of the leaves only. 
        if not SPLIT:
            lp_ndi=ndi[self.leaves_binary_img==True].ravel() # lp= leaves pixel
        else:
            lp_ndi=ndi[leaves_bi_of_obj==True].ravel() # lp= leaves pixel
            
        mean=lp_ndi.mean() #,lp_ndi.std(),np.median(lp_ndi)
        return mean
    
    #  compute NDI for all the bands in the image
    def NDI(self,SPLIT=False,leaves_bi_of_obj=""):
        x_ticks=list(range(len(self.wl)))
        wl_dic=dict(zip(x_ticks,self.wl))
        wl_dic={item[0]:float(item[1]) for item in wl_dic.items()}
        
        #create a df that behave like 2-dim matrix of the wavelength
        # each cell in the df[wl1,wl2] represent the ndi index of wl1 and wl2
        self.ndi_df=pd.DataFrame(columns=self.wl,index=self.wl)
        for wl1 in self.wl:
            for wl2 in self.wl:
                self.ndi_df.loc[wl1,wl2]=self.means_ndi(float(wl1),
                    float(wl2),wl_dic,SPLIT,leaves_bi_of_obj)
        return self.ndi_df

    def vegeation_indices(self,spectral_img_df,SPLIT,leaves_bi_of_obj,label=""):

        dvi_,ndvi_mean,ndvi_std,ndvi_median=self.NDVI(SPLIT=SPLIT,leaves_bi_of_obj=leaves_bi_of_obj)
        _,gndvi_mean,_,_=self.GNDVI(SPLIT=SPLIT,leaves_bi_of_obj=leaves_bi_of_obj)
        evi,evi_mean=self.EVI(SPLIT=SPLIT,leaves_bi_of_obj=leaves_bi_of_obj)
        lai,lai_mean=self.LAI(SPLIT=SPLIT,leaves_bi_of_obj=leaves_bi_of_obj)

        if self.ndi_tuple:
            wl1,wl2=self.ndi_tuple[0],self.ndi_tuple[1]
            specific_ndi=self.get_specific_ndi(wl1,wl2,SPLIT=SPLIT,leaves_bi_of_obj=leaves_bi_of_obj)

        if self.COMPUTE_NDI:
            ndi_df=self.NDI(SPLIT=SPLIT,leaves_bi_of_obj=leaves_bi_of_obj)
            ndi_df_path=self.save_ndi_table(ndi_df)
        
        #add relveant info and the indices to spectral_img_df
        idx=len(spectral_img_df)
        
        spectral_img_df.loc[idx,'img_num']=self.img_name
        if SPLIT:
            spectral_img_df.loc[idx,'label_name']=label
        spectral_img_df.loc[idx,'longitude']=self.longitude
        spectral_img_df.loc[idx,'latitude']=self.latitude
        spectral_img_df.loc[idx,'acquisition date']=self.acquisition_date
        
        #assignment of vegetation_idices to df
        spectral_img_df.loc[idx,'FVC']=self.fvc
        spectral_img_df.loc[idx,'ndvi_mean'],spectral_img_df.loc[idx,'ndvi_std'],spectral_img_df.loc[idx,'ndvi_median']=ndvi_mean,ndvi_std,ndvi_median
        spectral_img_df.loc[idx,'gndvi_mean']=gndvi_mean
        spectral_img_df.loc[idx,'evi_mean']=evi_mean
        spectral_img_df.loc[idx,'lai_mean']=lai_mean

        if self.ndi_tuple:
            ndi_str=f'NDI_{str(self.ndi_tuple)}'
            spectral_img_df.loc[idx, ndi_str]=specific_ndi
        if self.COMPUTE_NDI:
            spectral_img_df.loc[idx,'NDI_df']=ndi_df_path
          

        return spectral_img_df
        
    def compute_vegeation_indices(self,spectral_img_df):
        SPLIT=self.image_splitted
        if not SPLIT:
            leaves_bi_of_obj=self.leaves_binary_img
            spectral_img_df=self.vegeation_indices(spectral_img_df=spectral_img_df,
                                              SPLIT=SPLIT,
                                              leaves_bi_of_obj=leaves_bi_of_obj)
        else:
            for i in range(len(self.objs_df)):
                row=self.objs_df.iloc[i]
                binary_leaves_of_obj=row['obj_pixels']
                #_,ndvi_mean,_,_=self.NDVI(SPLIT=True,leaves_bi_of_obj=binary_leaves_of_obj)
                
                spectral_img_df=self.vegeation_indices(spectral_img_df=spectral_img_df,
                                              SPLIT=SPLIT,
                                              leaves_bi_of_obj=binary_leaves_of_obj,
                                              label=row['label_name'])
        return spectral_img_df

    def save_ndi_table(self,ndi_df=None):
        if ndi_df is None:
            ndi_df=self.NDI()
        ts = time.time()
        ndi_table_name=f'ndi_table_of_img_{self.img_name}_{ts}'
        ndi_table_path=f"{self.ndi_table_directory_path}"
        ndi_df_path=f"{ndi_table_path}/{ndi_table_name}.csv"
        ndi_df.to_csv(ndi_df_path,index=False)
        return ndi_df_path
        
    def save_leaves_binary_img(self,img_directory_path):
        image_to_save = self.leaves_binary_img.astype(np.uint8) * 255
        img_name_with_extension=f'{self.img_name}.png'
        img_path=f"{img_directory_path}/{img_name_with_extension}"
        cv2.imwrite(img_path, image_to_save)
    
    ######
    #
    # GETTER FUNCTION
    #
    ####
    
    # Normalized Difference Vegetation Index,quantifies vegetation health and density.
    # NDVI values:
    # - Values close to +1 indicate dense, healthy vegetation like forests. 
    # - Values around 0 indicate barren areas like rock, sand, or snow. 
    # - Negative values typically represent water bodies. 
    # - Moderate values (e.g., 0.2-0.5) can indicate sparse or stressed vegetation like grasslands or crops. 
    
    # the convention is the represented wavelength of red and nir is 670 and 800 corrspondly (Haboudane 2004)
    def get_ndvi(self,red_wl= 670,nir_wl=800):
        _,key_red,key_nir=self.auxiliary_func(red_wl,nir_wl)
        #calculate ndvi
        red=self.img[:,:,key_red]
        nir=self.img[:,:,key_nir]
        self.ndvi_ = (nir - red) / (nir + red)
        return self.ndvi_
    
    # get the ndi of given wavelenghts 
    def get_specific_ndi(self,wl1,wl2,SPLIT=False,leaves_bi_of_obj=""):
        x_ticks=list(range(len(self.wl)))
        wl_dic=dict(zip(x_ticks,self.wl))
        wl_dic={item[0]:float(item[1]) for item in wl_dic.items()}
        
        #create a df that behave like 2-dim matrix of the wavelength
        # each cell in the df[wl1,wl2] represent the ndi index of wl1 and wl2
        ndi=self.means_ndi(float(wl1),
                    float(wl2),wl_dic,SPLIT,leaves_bi_of_obj)
        return  ndi