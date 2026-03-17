import sys

from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
from PIL import Image
import cv2

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import pdb

class ImageProcessing:

    # PARAMETERS:
    # img_num : the number of the image
    def __init__(self,binary_img,img_num,
                annot_path,rotate=True):
       
        self.binary_img=binary_img
        self.img_num=img_num
        img_erosion=self.remove_noise()
        self.num_of_labels=self.compute_num_of_labels(annot_path)
        self.labels_assignment(annot_path,rotate=True)

    ###########
    #
    # FUNCTIONS
    #
    ##########
    

    def compute_num_of_labels(self,annot_path):
        df_annot=pd.read_csv(annot_path)
        df_annot['img_num']=df_annot['image_name'].apply(lambda x: x.split('.')[0])
        return len(df_annot[df_annot['img_num']==self.img_num]['label_name'].unique())
    
    #remove noise from binary_img
    def remove_noise(self):
        # Ensure the image is truly binary (0 or 255)
        image_array = (self.binary_img * 255).astype(np.uint8)
        _, binary_img = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        
        #elimenate the noise from the image
        img_erosion = cv2.erode(binary_img, kernel, iterations=1)
        self.binary_img_without_noise=img_erosion
        return img_erosion

    # assign class_id for each object
    # PARAMETERS:
    # annot_path: path to the annotation file. the annotation file is csv file
    #             with bounding box for each object and class id
    #             the columns of csv file are: ['label_name','bbox_x', 'bbox_y','bbox_width',
    #                                          'bbox_height','image_name','image_width','image_height','image_num']
    #              image_num columns must be assoisate with self.img_num
    # rotate : boolean paramter indicate whather the image selected for the annotation 
    #         was rotate. if it set to True , then we rotate it from left to Right (90 degree)
    def labels_assignment(self,annot_path,rotate):
        img_num=self.img_num
        if(rotate):
            pil_image = Image.fromarray(self.binary_img_without_noise)
            pil_rotated_img = pil_image.transpose(Image.ROTATE_270)
            rotated_img = np.asarray(pil_rotated_img)

        # get the annotation of the current image 
        df_annot=pd.read_csv(annot_path)
        df_annot["image_num"]=df_annot['image_name'].apply(lambda x: x.split(".")[0])
        objs_df=df_annot[df_annot["image_num"]==img_num]
        idx=len(objs_df.columns)
        objs_df.insert(idx,'obj_pixels',None)
        objs_df_labels=objs_df['label_name'].unique().tolist()

        #assign  binary image for each object
        for l in objs_df_labels:
            obj_df=objs_df[objs_df['label_name']==l]
            index=obj_df.index[0]
            x1, y1=obj_df.at[index,'bbox_x'],obj_df.at[index,'bbox_y']
            w,h=obj_df.at[index,'bbox_width'],obj_df.at[index,'bbox_height']
            x2=x1+w
            y2=y1+h

            # we create binary image with the same size of 
            # the original image where the current object are mark as True
            one_obj = np.zeros_like(rotated_img)
            binary_cropped_image = rotated_img[y1:y2, x1:x2]
            one_obj[y1:y2, x1:x2]=binary_cropped_image

            # we rotate again the image to adapat it to the original rotate
            if(rotate):
                one_obj_pil_image = Image.fromarray(one_obj)
                rotated_90_img = one_obj_pil_image.transpose(Image.ROTATE_90)
                one_obj = np.asarray(rotated_90_img)
                
            objs_df.at[index,'obj_pixels']=one_obj.astype(bool)
            
        self.objs_df=objs_df
    
    ######
    #
    # GETTER FUNCTION
    #
    ####
    # The function return binary image of object by given label
    def get_binary_img_of_object(self,label):
        cond = self.objs_df['label_name']==label
        idx = self.objs_df[cond].index[0]
        arr = self.objs_df.at[idx,'obj_pixels']
        return arr
    
    def get_objs(self):
        return self.objs_df