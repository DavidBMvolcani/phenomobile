from time import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from datetime import datetime

from skimage.color import rgb2hsv

from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy import stats


class rgb_image:

     # PARAMETERS:
    #     - img_bgr : img_bgr image 
    #     - bb_df: the dataframe of rgb images and their object (like plant) bounding boxes
    #     - df_index: the index of the image in bb_df
    #     - 
    def __init__(self,img_bgr,bb_df,df_index):
        self.img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        blank_img =np.zeros(self.img_rgb.shape[:2])
        
        #compute the croped img by the bb
        cropped_img,x1,y1,x2,y2=self.compute_croped_img(bb_df,df_index)
        
        #compute hsv values
        self.compute_hsv_values(cropped_img)
        
        #compute binary image of the object
        hue_lt,value_ut=self.compute_thersholds( self.hue_img,self.value_img)
        obj_binary_img = (self.hue_img > hue_lt)  & ( self.value_img<value_ut)
        blank_img[y1:y2,x1:x2]=obj_binary_img
        objBinImg=blank_img.copy()

        self.obj_binary_img= obj_binary_img
        
        #compute hsv means of the object
        self.hue_plant_mean=self.hue_img[obj_binary_img].mean()
        self.saturation_plant_mean=self.saturation_img[obj_binary_img].mean()
        self.value_plant_mean=self.value_img[obj_binary_img].mean()

    def compute_croped_img(self,bb_df,df_index):
        index=bb_df.index[df_index]
        x1, y1=bb_df.at[index,'bbox_x'],bb_df.at[index,'bbox_y']
        w,h=bb_df.at[index,'bbox_width'],bb_df.at[index,'bbox_height']
        x2=x1+w
        y2=y1+h
        cropped_image=self.img_rgb[y1:y2,x1:x2]
        return cropped_image,x1,y1,x2,y2

    def compute_hsv_values(self,croped_img,plot=False):
        hsv_img = rgb2hsv(croped_img)
        self.hue_img = hsv_img[:, :, 0]
        self.saturation_img=hsv_img[:, :, 1]
        self.value_img = hsv_img[:, :, 2]
        if plot:
            self.plot_hsv_images()

    def compute_thersholds(self,hue_img,value_img):
        small_hue_img = cv2.resize(hue_img, (512, 512), interpolation=cv2.INTER_AREA)
        small_value_img = cv2.resize(value_img, (512, 512), interpolation=cv2.INTER_AREA)
        hue_lt = self.find_x_vally(small_hue_img)
        value_ut = self.find_x_vally(small_value_img)
        return hue_lt,value_ut

   

    ##########
    #
    # AUXILIARY FUNCTIONS
    #
    ##########

    # GOAL:
    #    find the theroshold number for binary filter of the object
    # explanation:
    #    the hsv images are differ in their values
    #    so we can't peaks constant value to filter the hsv image
    #    (like hue image) with this. Instead we assume the distrbution
    #    of the hsv values are bimodel with two dominant peaks
    #    one model is for the object and the other is the background
    #    so we filter out the vally point in each image and use it for filter.
    # 
    # PARAMETERS:
    #     - img : one of the hsv images (for example: hue image)
    def find_x_vally(self,img,plot=False):

        data=img.ravel()
        # Calculate Z-scores for each data point
        z_scores = np.abs(stats.zscore(data))
        
        # Define a threshold (e.g., 3 standard deviations)
        threshold = 3
        
        # Create a boolean mask of data points that are NOT outliers
        not_outliers_mask = z_scores <= threshold
        
        # Filter the data to keep only non-outliers
        cleaned_data = data[not_outliers_mask]
        
        
        # 2. Perform Kernel Density Estimation (KDE)
        # Create a smooth representation of the distribution
        kde = gaussian_kde(cleaned_data)
        # Define the range over which to evaluate the KDE
        num_points=200
        x_range = np.linspace(min(cleaned_data), max(cleaned_data),num_points)
        # Get the density values for the x_range
        y_density = kde(x_range)
        
        
        # Find all peaks; distance helps filter noise
        skip_distance=num_points*0.1
        peak_indices, _ = find_peaks(y_density, distance=skip_distance)
        
        # Identify the TWO highest peaks specifically
        # Sort by density (y values) and take the top two
        top_two_peaks = peak_indices[np.argsort(y_density[peak_indices])[-2:]]
        peak_start, peak_end = np.sort(top_two_peaks) # Ensure they are in x-axis order
        
        # --- 4. FIND THE VALLEY ---
        # Slice the density array between the two peak indices
        valley_slice = y_density[peak_start:peak_end]
        # Find the index of the minimum value in that slice
        relative_valley_index = np.argmin(valley_slice)
        valley_index = peak_start + relative_valley_index
        
        # Final coordinates
        valley_x = x_range[valley_index]
        valley_y = y_density[valley_index]
        # print(f"Valley found at X: {valley_x:.2f}, Y: {valley_y:.4f}")

        if plot:
            self.plot_Bimodal_Distribution(x_range, y_density,top_two_peaks,
                                           valley_x, valley_y)    
        return valley_x

    ##########
    #
    # GETTER FUNCTIONS
    #
    ##########
    
    def get_hsv_obj_means(self):
        hm=self.hue_plant_mean
        sm=self.saturation_plant_mean
        vm=self.value_plant_mean
        return hm,sm,vm

    def get_binary_obj_img(self):
        return self.obj_binary_img
    ##########
    #
    # PLOT FUNCTIONS
    #
    ##########

    def plot_hsv_images(self):
        img_rgb=self.img_rgb
        hue_img=self.hue_img
        saturation_img=self.saturation_img
        value_img=self.value_img 
        
        fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(8, 2))
        ax0.imshow(img_rgb)
        ax0.set_title("RGB image")
        ax0.axis('off')
        ax1.imshow(hue_img, cmap='hsv')
        ax1.set_title("Hue channel")
        ax1.axis('off')
        ax2.imshow(saturation_img)
        ax2.set_title("Saturation channel")
        ax2.axis('off')
        ax3.imshow(value_img)
        ax3.set_title("Value channel")
        ax3.axis('off')

    def plot_Bimodal_Distribution(self,x_range, y_density,top_two_peaks,
                                 valley_x, valley_y):
        plt.figure(figsize=(10, 5))
        plt.plot(x_range, y_density, label='KDE Density', color='black')
        plt.plot(x_range[top_two_peaks], y_density[top_two_peaks], "x", color='red', markersize=10, label='Top 2 Peaks')
        plt.plot(valley_x, valley_y, "o", color='blue', label='Valley (Minimum)')
        plt.fill_between(x_range, y_density, alpha=0.1)
        plt.title("Bimodal Distribution: Peaks and Intervening Valley")
        plt.legend()
        plt.show()