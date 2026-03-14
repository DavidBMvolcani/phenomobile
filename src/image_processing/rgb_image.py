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
    #       (!) it must include the fields:
    #          - bbox_x, 
    #          - bbox_y, 
    #          - bbox_width, 
    #          - bbox_height
    #     - df_index: the index of the image in bb_df
    #     - config : project configuration used to set the strategy for computing the mask of the object
    def __init__(self,
            img_bgr=None,
            bb_df=None,
            df_index=None,
            config=None):
        
        self.bb_df=bb_df
        self.df_index=df_index
        self.config=config

        if bb_df is None or df_index is None:
            raise ValueError("bb_df and df_index must be provided")
        if img_bgr is None:
            raise ValueError("img_bgr must be provided")

        if ('bbox_x' not in bb_df.columns or \
            'bbox_y' not in bb_df.columns or \
            'bbox_width' not in bb_df.columns or \
            'bbox_height' not in bb_df.columns):
            raise ValueError("bb_df must include the fields: bbox_x, bbox_y, bbox_width, bbox_height")
        
        self.img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  
        
        #compute the croped img by the bb
        self.cropped_img=self.compute_croped_img()

        #compute the mask of the object
        mask=self.compute_mask()
        if mask is not None:
            self.mask=mask

        self.masked_img=self.get_masked_img()
        # compute hsv values of the  masked image
        self.compute_hsv_values(self.masked_img)
        

    def compute_croped_img(self):
        index=self.bb_df.index[self.df_index]
        x1, y1=self.bb_df.at[index,'bbox_x'],self.bb_df.at[index,'bbox_y']
        w,h=self.bb_df.at[index,'bbox_width'],self.bb_df.at[index,'bbox_height']
        x2=x1+w
        y2=y1+h
        cropped_image=self.img_rgb[y1:y2,x1:x2]
        return cropped_image

    def compute_hsv_values(self,masked_img,plot=False):
        hsv_img = rgb2hsv(masked_img)
        hue_img = hsv_img[:, :, 0]
        saturation_img = hsv_img[:, :, 1]
        value_img = hsv_img[:, :, 2]
        if plot:
            self.plot_hsv_images()
        #mask the hsv images
        self.masked_hue = hue_img[hue_img > 0]
        self.masked_saturation = saturation_img[saturation_img > 0]
        self.masked_value = value_img[value_img > 0]
        

    def compute_mask(self):
        if self.config is None:
            raise ValueError("config for mask computation must be provided")
        mask_config=self.config.get("mask_computation_method", {})
        method = mask_config.get("method", "hsv_threshold")
        if method == "hsv_threshold":
            self.compute_hsv_threshold_mask()
        elif method == "object_contours":
            mask=self.compute_object_contours_mask()
            return mask
        else:
            raise ValueError(f"Unknown mask computation method: {method}")
        return None

    def compute_hsv_threshold_mask(self):
        raise NotImplementedError("compute_hsv_threshold_mask method is not implemented yet")
    
    # (!) this function parameter and strategy should be configurable based on
    # (!) the specific use case and image characteristics of the scene and the realtion of 
    # (!) the object of interest to the background 
    def compute_object_contours_mask(self,lt=50,ht=150):

        # helper function to compute the largest contour mask
        def _compute_largest_contour_mask(lt=lt,ht=ht):
            # 1. Load the image
            img = self.cropped_img.copy()
            # Convert to grayscale (Canny requires single channel)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 3. Apply Canny Edge Detection
            edges = cv2.Canny(gray, threshold1=lt, threshold2=ht)

            # Create a kernel (the size determines how far the 'reach' is to connect lines)
            kernel = np.ones((15,15), np.uint8)

            # 4. Dilate: Makes the white lines thicker to bridge the gaps
            thick_edges = cv2.dilate(edges, kernel, iterations=1)

            # 5. Closing: Fills small holes inside the boundary
            closed_edges = cv2.morphologyEx(thick_edges, cv2.MORPH_CLOSE, kernel)

            # 6. THE FILLING STEP: Find and fill the largest hole
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a blank black canvas for our result
            solid_mask = np.zeros_like(edges)

            if len(contours) == 0:
                return 0,gray,solid_mask,None
            
            if len(contours) > 0:
                # Pick the largest shape (the lettuce)
                largest = max(contours, key=cv2.contourArea)

                # Compute the area in pixels
                area_pixels = cv2.contourArea(largest)

                # Total pixels in the image
                total_pixels = img.shape[0] * img.shape[1]

                # Percentage of image covered by lettuce
                percentage = (area_pixels / total_pixels) * 100

                #print(f"Lettuce coverage: {percentage:.2f}%")
            
                percentage=percentage/100
                return percentage,gray,solid_mask,largest
        
        percentage,gray,solid_mask,largest = _compute_largest_contour_mask(lt=lt,ht=ht)
        while  percentage<0.5 and lt>1 and ht>5:
            # print("change thresholds")
            ht=ht/2
            lt=lt/2
            percentage,gray,solid_mask,largest=_compute_largest_contour_mask(lt=lt,ht=ht)

        # DRAW and FILL the shape
        cv2.drawContours(solid_mask, [largest], -1, 255, thickness=cv2.FILLED)

        # perform erosion on solid mask to remove edges in the corner of the image that
        #  are not part of the objects
        kernel = np.ones((50,50),np.uint8)
        solid_mask = cv2.morphologyEx(solid_mask, cv2.MORPH_ERODE, kernel)

        return solid_mask

       
        
    ##########
    #
    # GETTER FUNCTIONS
    #
    ##########
    
    def get_masked_hsv_obj_means(self):
       
        return self.masked_hue.mean(),self.masked_saturation.mean(),self.masked_value.mean()

    def get_masked_hsv_obj_stds(self):
       
        return self.masked_hue.std(),self.masked_saturation.std(),self.masked_value.std()
    
    def get_masked_hsv_obj_medians(self):
       
       return np.median(self.masked_hue),np.median(self.masked_saturation),np.median(self.masked_value)
        
    def get_masked_img(self):
        if self.mask is None:
            raise ValueError("Mask is not computed .")

        mask=self.mask
        # 1. Ensure the mask is 8-bit (OpenCV requires CV_8U for masks)
        mask = mask.astype(np.uint8)

        # 2. Force the mask to the exact size of the image if it's off
        if mask.shape[:2] != self.cropped_img.shape[:2]:
            #print(f"Resizing mask from {mask.shape} to {cropped_img.shape[:2]}")
            mask = cv2.resize(mask, (self.cropped_img.shape[1], self.cropped_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 3. Now apply the bitwise_and
        mask_img_color = cv2.bitwise_and(self.cropped_img, self.cropped_img, mask=mask)

        return mask_img_color
        
    ##########
    #
    # PLOT FUNCTIONS
    #
    ##########

    def plot_hsv_images(self):
        img_rgb=self.img_rgb
        hue_img=self.masked_hue
        saturation_img=self.masked_saturation
        value_img=self.masked_value 
        
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

    