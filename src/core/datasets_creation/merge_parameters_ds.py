
import cv2
import pandas as pd
import datetime as dt
from spectral import *

# explanation: this class is used to merge two sources of  parameters datasets
# for example: the hyper-spectral and thermal datasets

class MergeParameterDs:
    def __init__(
        self,
        logger,
        dataset_folder,
        RAW_DATA_FOLDER=None
        ):

        self.logger = logger
        self.dataset_folder=dataset_folder
        self.RAW_DATA_FOLDER=RAW_DATA_FOLDER
        self.formatted_datetime = dt.datetime.now().strftime("%Y-%m-%d %H.%M.%S")

        
    # we map hyper-spectral and thermal images by using SIFT algorithm. 
    # The SIFT algorithm applied on the gray images created for HS, and the 
    # gary images for thermal images erlier.
    def map_hyper_spectral_and_thermal_datasets(
        self,
        gray_for_HS_imgs,
        gray_for_Th_imgs,
        gray_for_HS_imgs_directory_path,
        gray_for_Th_imgs_directory_path,
        spectral_img_df,
        thermal_df,
        ratio_of_distance_between_matches=0.75,
        score_th=0.02
        ):
               
        sift = cv2.SIFT_create()

        # match between hs and theraml images by SIFT algorithm
        # for each gray image attched to hs image - we found the correspond gray image attached to thermal image
        # that have more 'good matches' form the other rgb images.
        
        Grays_for_HS_imgs=gray_for_HS_imgs 
        Grays_for_Th_imgs=gray_for_Th_imgs
        
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
            self.logger.info(f"go over {cnt/lenght :.2f} from the images") 
            max_score,corr_img=0,''
            for b, (kpB, desB) in B_data.items():  
                matches = flann.knnMatch(desA, desB, k=2)
                #ratio-test: if the best match (m) is much better (significantly smaller distance)
                # than the second-best match (n), the match is likely distinctive and reliable.
                good_matches = []
                for m, n in matches:
                    if m.distance < ratio_of_distance_between_matches * n.distance: 
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
        sp_df=spectral_img_df.rename(columns=lambda col: f'hs_{col}')
        th_df=thermal_df.rename(columns=lambda col: f'th_{col}')
        
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

        # save the merged dataset
        self.save_mergerd_img_df()

        # delete the local grays images folders
        self.delete_locals_grays_images_folders()

    ###
    # helper function
    #
    # explain: sometimes mistakenly the photograhpher take more then one image for
    # the same sence - so we filterd out those images
    def remove_duplicated_images(self,data_,flann,threshold=0.2):
        imgs_names=list(data_.keys()).copy()
        data_copy= data_.copy()
        img_name=imgs_names.pop()
        suspected_images=[]
        cnt,lenght=0,len(imgs_names)
        while len(imgs_names)>0:
            cnt=cnt+1
            self.logger.info(f"go over {cnt/lenght :.2f} from the images")
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

    def save_mergerd_img_df(self):
        datasets_paths=self.dataset_folder
        rmt_folder=self.RAW_DATA_FOLDER
        dt=self.formatted_datetime
        self.merged_img_df.to_csv(f"{datasets_paths}/hs_and_th_imgs_dataset_{rmt_folder}_{dt}.csv",index=False)
        self.logger.info(f'new file created : {datasets_paths}/hs_and_th_imgs_dataset_{rmt_folder}_{dt}.csv')
    
      
    # DELETE floders functions
    def delete_locals_grays_images_folders(self):
        self.logger.info("Deleting local grays images folders")
        shutil.rmtree(self.gray_for_HS_imgs_directory_path)
        shutil.rmtree(self.gray_for_Th_imgs_directory_path)