import sys

import seaborn as sns
from matplotlib import pyplot as plt

import subprocess
import json
from skimage.color import rgb2hsv
import numpy as np
from PIL import Image

from scipy.stats import skew
import scipy
from scipy.ndimage import uniform_filter,median_filter,gaussian_filter

import cv2


class thermal_image:
    
    # Constructor method, called when a new object is created
    def __init__(self,img_name,img_path,FlirImageExtractor_path):

        #process FLIR thermal image with exiftool - only import when needed
        try:
            sys.path.append(FlirImageExtractor_path)
            import flir_image_extractor
            self.img_name=img_name
            self.fir = flir_image_extractor.FlirImageExtractor()
            self.fir.process_image(img_path)
        except ImportError as e:
            raise ImportError(f"Failed to import flir_image_extractor from {FlirImageExtractor_path}: {e}")
        
        self.exiftool_path="exiftool"
        self.img_path=img_path

        #set thermal and rgb images
        self.set_thermal_img()
        self.set_rgb_img()

        #set hsv channels and filter the rgb images by hsv channels
        self.rgb_to_hsv()
        self.filtered_rbg_img_by_hsv_channels()

        # remove the calibartion panel from the image
        self.remove_calibration_panel_from_img()

        # apply the boolean filter on the thermal image
        # First we find the fillterd crop imaged from rbg image
        # then we apply the fillterd crop imaged on the thermal image
        self.get_crop_of_thermal_img_from_rgb_image()
        self.filter_leaves_from_thermal_img()

        #compute the temperatures of the leaves and other statistic parameters
        self.get_temperatures_of_leaves()
        fvc=self.get_FVC()
        self.get_statistic_temperatures_parameters_of_the_leaves()

    ###########
    #
    # FUNCTIONS
    #
    ##########

    #DMS format = degrees / minutes / seconds, human-readable.
    #Decimal degrees = float values, convenient for calculations, mapping, and GIS.
    def dms_to_dd(self,dms_str):
        # Example: '31 deg 54\' 28.26" N'
        parts = dms_str.replace('deg','').replace('"','').replace("'",'').split()
        deg, minutes, seconds, direction = float(parts[0]), float(parts[1]), float(parts[2]), parts[3]
        dd = deg + minutes/60 + seconds/3600
        if direction in ['S','W']:
            dd *= -1
        return dd
    
    def get_metedata_parameters(self):
        meta_bytes = subprocess.check_output(
            [self.exiftool_path,self.img_path,'-j'])
        meta_string = meta_bytes.decode('utf-8')
        meta_json = json.loads(meta_string)[0]
        return meta_json

    def get_metedata_parameter(self,tag):
        meta = self.get_metedata_parameters()
        return meta[tag]
    
    # set the thermal data and image
    def set_thermal_img(self):
        self.thermal_img_np=self.fir.get_thermal_np()
        self.thermal_img=Image.fromarray(self.thermal_img_np)

    def set_rgb_img(self):
        self.rgb_img_np=self.fir.get_rgb_np()

    def rgb_to_hsv(self):
        # identify the leaves pixels by converting the rgb image to hsv
        self.hsv_img = rgb2hsv(self.rgb_img_np)
        self.hue_img =   self.hsv_img[:, :, 0]
        self.saturation_img=  self.hsv_img[:, :, 1]
        self.value_img =   self.hsv_img[:, :, 2]

    def filtered_rbg_img_by_hsv_channels(self,shadow_lower_th=0.3,
                                         cal_panel_upper_th=1,sat_upper_th=0.2):
        remove_shadow= (self.value_img>shadow_lower_th)
        remove_reflectance_object= (self.value_img<cal_panel_upper_th)
        self.boolean_img_filtered_by_hsv = (self.saturation_img<sat_upper_th) & remove_reflectance_object & remove_shadow

    def remove_calibration_panel_from_img(self):
        # Convert to grayscale
        gray = cv2.cvtColor(self.rgb_img_np, cv2.COLOR_BGR2GRAY)
        thresh = cv2.Canny(gray, 50, 120)

        # Find contours
        # cv.CHAIN_APPROX_SIMPLE It will store number of end points (eg.In case of rectangle it will store 4)
        # cv.RETR_EXTERNAL retrieves only the extreme outer contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_perimeter,max_contour=0,[]

        # Iterate through the detected contours
        for contour in contours:
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, closed=True) 
            #If closed =True, the function calculates the perimeter of a closed contour 
            approx = cv2.approxPolyDP(contour, 0.15 * perimeter, True)
            
            # Check if the polygon has four vertices
            if len(approx) == 4:
                # we ignore tiny squares or rectangles and focus only on the big calibration square panel
                if(perimeter>max_perimeter):
                    max_perimeter=perimeter
                    max_contour=contour
                    x, y, w, h = cv2.boundingRect(approx)
                    roi = self.rgb_img_np[y:y+h, x:x+w]
        
        approx = cv2.approxPolyDP(max_contour, 0.15 * max_perimeter, True)

        # mark the calibration square panel as zeros and the background as 1
        gray_img = cv2.cvtColor(self.rgb_img_np, cv2.COLOR_BGR2GRAY) 
        new_img=np.ones(gray_img.shape)
        new_img[y:y+h, x:x+w]=0
        reflect_obj_binary_img=new_img

        # now apply logical AND for the filtered img  with hsv channel to the reflect_obj_binary_img
        reflect_obj_bool_img=reflect_obj_binary_img.astype(bool)
        self.boolean_img_filtered_by_hsv_without_calibration_panel=self.boolean_img_filtered_by_hsv & reflect_obj_bool_img
       

    def get_crop_of_thermal_img_from_rgb_image(self):
        filtered_bool_img=self.boolean_img_filtered_by_hsv_without_calibration_panel
        img = Image.fromarray(filtered_bool_img)
        rgb_w, rgb_h = img.size
        data=self.thermal_img_np
        thermal_h,thermal_w=data.shape
        
        x1 = round((rgb_w-thermal_w) / 2)
        y1 = round((rgb_h-thermal_h) / 2 )
        x2 = round(x1+thermal_w)
        y2 = round(y1+thermal_h)
        
        box=(x1,y1,x2,y2)
        self.crop_img=img.crop(box)

    def get_AirTemperature(self,tag='AtmosphericTemperature'):
        air_temp=self.get_metedata_parameter(tag)
        self.air_temp=float(air_temp.split()[0]) # get rid from the celsuis sign ('C')
        return self.air_temp

    def filter_leaves_from_thermal_img(self):
        air_temp=self.get_AirTemperature()
        data=self.thermal_img_np
        # our assumption is that leaves can exist up to 7 celsius degree avove air temperature
        self.binary_temp_above_th_img=data<air_temp+7 

        # now we filter only the leaves without the shadow of the leaves 
        binary_crop_img = np.asarray(self.crop_img)
        self.binary_temp_above_th_img_without_shadow=self.binary_temp_above_th_img & binary_crop_img

    def get_temperatures_of_leaves(self):
        mask=self.binary_temp_above_th_img_without_shadow.astype(np.uint8)
        data=self.thermal_img_np
        self.leaves_thermal_np=data[mask==1]
        return self.leaves_thermal_np

    #fractional vegetation cover (FVC) or leaf cover
    def get_FVC(self):
        mask=self.binary_temp_above_th_img_without_shadow.astype(np.uint8)
        data=self.thermal_img_np
        data_size=data.shape[0]*data.shape[1]
        #fractional vegetation cover (FVC) or leaf cover
        self.fvc=self.leaves_thermal_np.shape[0]/data_size
        return self.fvc
        
    def get_statistic_temperatures_parameters_of_the_leaves(self):
        self.leaves_thermal_mean=self.leaves_thermal_np.mean()
        self.leaves_thermal_median=np.median(self.leaves_thermal_np)
        self.leaves_thermal_std=self.leaves_thermal_np.std()
        # Calculate skewness
        self.leaves_thermal_skewness = skew(self.leaves_thermal_np)

    def compute_Theraml_parameters(self,thermal_df,img_name):
        i=len(thermal_df)
        thermal_df.loc[i,'image_name']=img_name
        thermal_df.loc[i,'fvc']=self.fvc
        thermal_df.loc[i,'leaves_mean_temp']=self.leaves_thermal_mean
        thermal_df.loc[i,'leaves_median_temp']=self.leaves_thermal_median
        thermal_df.loc[i,'leaves_std_temp']=self.leaves_thermal_std
        thermal_df.loc[i,'leaves_skewness_tmp']=self.leaves_thermal_skewness
        
        #metadata_parameters
        thermal_df.loc[i,'CreateDate']=self.get_metedata_parameter('CreateDate')
        thermal_df.loc[i,'SubjectDistance']=self.get_metedata_parameter('SubjectDistance')
        thermal_df.loc[i,'FocalLength']=self.get_metedata_parameter('FocalLength')
        thermal_df.loc[i,'RelativeHumidity']=self.get_metedata_parameter('RelativeHumidity')
        thermal_df.loc[i,'GPSLatitude']=self.dms_to_dd(self.get_metedata_parameter('GPSLatitude'))
        thermal_df.loc[i,'GPSLongitude']=self.dms_to_dd(self.get_metedata_parameter('GPSLongitude'))
        
        return thermal_df
        
       
                    
    #########
    #
    # ploting and print
    #
    ########
    
    # plot RGB and Thermal img
    def fir_plot(self):
        self.fir.plot()

    def plot_thermal_img(self):
        plt.figure()
        plt.imshow(self.thermal_img_np, cmap="afmhot",aspect="auto" ) # set aspect ratio
        plt.colorbar(format='%.2f')  # add color bar to image
        plt.show()

    def plot_rgb_img(self):
        fig=plt.figure()
        plt.imshow(self.rgb_img_np)
        plt.show()

    def plot_hsv_channels(self):
        fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(8, 2))
        ax0.imshow(self.rgb_img_np)
        ax0.set_title("RGB image")
        ax0.axis('off')
        ax1.imshow(self.hue_img, cmap='hsv')
        ax1.set_title("Hue channel")
        ax1.axis('off')
        ax2.imshow(self.saturation_img)
        ax2.set_title("Saturation channel")
        ax2.axis('off')
        ax3.imshow(self.value_img)
        ax3.set_title("Value channel")
        ax3.axis('off')

    def plot_boolean_img_filtered_by_hsv(self):
        plt.figure()
        plt.title("filtered image by hsv channels")
        plt.imshow(self.boolean_img_filtered_by_hsv)

    def plot_boolean_img_filtered_by_hsv_without_calibration_panel(self):
        plt.figure()
        plt.imshow(self.boolean_img_filtered_by_hsv_without_calibration_panel)

    def plot_gray_crop_of_thermal_image(self):
        plt.figure(figsize=(7,4))
        plt.title("The crop of the Thermal image in the RGB image")
        plt.imshow(self.crop_img)

    def plot_fillterd_leaves_from_thermal_image(self):
        plt.figure()
        plt.subplot(121)
        plt.title("with shadow")
        plt.imshow(self.binary_temp_above_th_img)
        plt.subplot(122)
        plt.title("without_shadow")
        plt.imshow(self.binary_temp_above_th_img_without_shadow)
        plt.show()

    def print_statistic_temperatures_parameters_of_the_leaves(self):
        print('mean temp of the leaves',self.leaves_thermal_mean,sep='\n')
        print('median temp of the leaves',self.leaves_thermal_median,sep='\n')
        print('std temp of the leaves',self.leaves_thermal_std,sep='\n')
        print('skewness of the leaves',self.leaves_thermal_skewness,sep='\n')

    def histogram_of_temperatures_of_the_leaves(self):
        fig = plt.figure(figsize =(8, 5))
        sns.histplot(data=self.leaves_thermal_np, kde=True, color='skyblue', edgecolor='black', bins=30)
        plt.show()
    
        