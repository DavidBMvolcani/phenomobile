import numpy as np
import itertools
import pprint
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
from heapq import nlargest
import statsmodels.api as sm
import math
import os
from dotenv import load_dotenv

from sklearn import datasets
from sklearn.model_selection import train_test_split,KFold, cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import RocCurveDisplay

from xgboost import XGBRegressor,XGBClassifier

import pdb
import colorsys
import webcolors

from scipy import stats
import statsmodels.api as sm



class training:

    # PARAMETERS:
    # - ENV_FILE: boolean parameter indicate whether the parameters
    #     will be taken from enviroment file.
    #    - dataset_name: the name of the dataset that will be used for training
    #    - fix_method: the method that will be chosse when dataset have null values 
    #    - tasks: string from one of this :['regression' ,'classifiction'] 
    #      the defualt value is 'regression'
    #    - model : you can set the model you want to fit to the data from
    #      the options below. if no option will be set - all the models be chossen.
    #      the options are ['linear regression']
    def __init__(self,ENV_FILE=True,dataset_name=None,
                 fix_method='KEEP ROWS',task='regression',model=None ):
                        
        if ENV_FILE :
            # Load environment variables from hidden .env file
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H")
            cd=os.getcwd()
            
            # Load environment variables from hidden .env file
            dataset_folder=os.environ.get("dataset_folder")
            if dataset_name is None:
                dataset_name=os.environ.get("dataset_name")
            home_directory_name=os.environ.get("home_directory_name")

            #find the path to the home directory
            splited_path=cd.split("/")
            hd_idx=splited_path.index(home_directory_name)
            separator = "/"
            home_path=separator.join(splited_path[:hd_idx+1])
            self.home_path= home_path
            
            #set the directory path to the output_dataset (csv files)
            datasets_paths=f"{home_path}/{dataset_folder}"
            dataset=f'{datasets_paths}/{dataset_name}'

            self.datasets_paths=datasets_paths

            self.df=pd.read_csv(dataset)
            self.original_df=self.df.copy()

            # fixed_df : if there are null values in the dataframe 
            # a predefined method will be apply to fix it
            if self.does_df_has_null_values(self.df):
                self.remove_null_from_df(fix_method)

        # Use provided dataset_name as direct path
        else:
            if dataset_name is None:
                raise ValueError("dataset_name must be provided when ENV_FILE=False")
            
            self.df = pd.read_csv(dataset_name)
            self.original_df = self.df.copy()
            
            # fixed_df : if there are null values in the dataframe 
            # a predefined method will be apply to fix it
            if self.does_df_has_null_values(self.df):
                self.remove_null_from_df(fix_method)

        # Set up models based on task (common to both ENV_FILE paths)
        if task=='regression':
            #set regression models
            self.set_regression_models(model)
        elif task=='classifiction':
            # set suported classifiction models
            self.set_calssification_model()

    #####
    #
    # PREPROCESSING
    #
    ####

    # region
    def does_df_has_null_values(self,df,PRINT=False):
        return df.isnull().any().any()

    def columns_with_null_values(self,df):
        cols_with_null_vals=[]
        for col in list(df.columns):
            if df[col].isnull().any()==True:
                cols_with_null_vals.append(col)
        return cols_with_null_vals
        

    # there are there options how to deal with null values
    # 1) remove the rows with null values
    # 2) remove the columns with null values
    # 3) impuation of values
    # comments: 
    # caution sholud be taken in the second method, becuase 
    # there passability to earse the target columns incidentally
    # the third option is vary by dataset discription
    def remove_null_from_df(self,method):
        if method=='KEEP ROWS':
            self.df = self.df[self.df.notnull().all(axis=1)]
        if method=='KEEP COLS':
            self.df  = self.df.dropna(axis=1, how='any')

        
     # endregion
    #######
    #
    # CLASSIFICATION MODELS
    #
    ######

    # region
    def set_calssification_model(self):
        classification_models_names=['LogisticRegression','DecisionTreeClassifier',
                                       'SVC','RandomForestClassifier',
                                        'XGBClassifier','GaussianNB']
        self.classification_models_names=classification_models_names

    def eval_rf_classifer(self,target,X,y):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc=accuracy_score(y_test, y_pred)
        cr=classification_report(y_test, y_pred,zero_division=0)
        cm=confusion_matrix(y_test, y_pred)
        return acc,cr,cm

    def eval_xgb_binary_classifer(target,X,y):
         raise NotImplementedError("not implemented function")
        
    def eval_xgb_classifer(self,target,X,y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        num_of_labels=len(label_encoder.classes_)
        if num_of_labels==2:
            acc,cr,cm=self.eval_xgb_binary_classifer(target,X,y)
        else:    
            model=XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_),
                                            n_estimators=100, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc=accuracy_score(y_test, y_pred)
            cr=classification_report(y_test, y_pred,zero_division=0)
            cm=confusion_matrix(y_test, y_pred)
        return acc,cr,cm
        
    def show_evaluation_metrics(self,acc,cr,cm,show_cofusion_matrix=True):
        print(f"Accuracy: {acc}")
        print("\nClassification Report:\n", cr)
        if show_cofusion_matrix:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm) 
            disp.plot()
            plt.title("Confusion Matrix")
            plt.show()
  
        
    def classification(self,classifier_name,target,trainable_features):
        df=self.df
        # make sure the target are intenger
        df[target]=df[target].astype(int)

        X = df[trainable_features].copy()
        X = X.drop(target, axis=1) 
        y = df[target]
        
        if classifier_name not in self.classification_models_names:
            print("model name not in supported models")
            return
        if classifier_name=='RandomForestClassifier':
            acc,cr,cm=self.eval_rf_classifer(target,X,y)
        elif classifier_name=='XGBClassifier':
            acc,cr,cm=self.eval_xgb_classifer(target,X,y)
            
        #show     
        self.show_evaluation_metrics(acc,cr,cm)
            
    # endregion
    ########
    #
    # REGRESSION MODELS
    #
    ######  

    # GOAL: set the regression models. there are 3 regression models: xgboost,randomforeset
    #     and linear_regression
    # PARAMETERS:
    #    - model : you can set the model you want to fit to the data from
    #      the options below. if no option will be set - all the models be chossen.
    #      the options are ['linear regression','Ridge']
    def set_regression_models(self,model=None):
        reg_models=[]
        if model=='linear regression':
             lr = LinearRegression()
             reg_models.append(lr)
        elif model=="Ridge":
            mod=Ridge(alpha=1.0)
            reg_models.append(mod)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1)
            lr = LinearRegression()
            xgb_model=XGBRegressor(objective='reg:squarederror',n_estimators=100,n_jobs=-1)
            reg_models.append(xgb_model)
            reg_models.append(rf)
            reg_models.append(lr)
       
        
        # Create pretty names map dynamically based on actual models
        pretty_names_map = {}
        for reg_model in reg_models:
            model_type = str(type(reg_model))
            if 'RandomForestRegressor' in model_type:
                pretty_names_map[model_type] = "Random Forest Regressor"
            elif 'XGBRegressor' in model_type:
                pretty_names_map[model_type] = "XGB Regressor"
            elif 'LinearRegression' in model_type:
                pretty_names_map[model_type] = "Linear Regression"
            elif 'Ridge' in model_type:
                pretty_names_map[model_type] = "Ridge"
            else:
                pretty_names_map[model_type] = str(type(reg_model))

        self.reg_models=reg_models
        self.pretty_names_map=pretty_names_map
        
    
     # PARAMETERS:
     #    - split: boolean parameter that set wheather to split the dataset to train and test groups
    def model_evaluation(self,X, y,target,results_df,pretty_names_map,reg_models,split=True):
        #  Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # iterate all over the models and predict the metric results
        for reg_model in reg_models:
            idx=len(results_df)
            results_df.loc[idx,'target']=target
            results_df.loc[idx,'predictable_features']=str(list(X.columns))
            model_name=pretty_names_map[str(type(reg_model))]
            results_df.loc[idx,'model_name']= model_name
    
            if(split):
                reg_model.fit(X_train,y_train)
                y_pred = reg_model.predict(X_test)
            else:
                reg_model.fit(X,y)
                y_pred = reg_model.predict(X)
                y_test=y
            
            results_df.loc[idx,'R2']= r2_score(y_test,y_pred)
            results_df.loc[idx,'RMSE']= root_mean_squared_error(y_test,y_pred)
        return results_df


    def split_dataset_to_X_and_y(self,trainable_features,target,df):
        #check if the trainable_features exist in the df columns
        df_cols=list(df.columns)
        if not(set(trainable_features).issubset(set(df_cols))):
            print(
                    "trainable features contain one or more parameter "
                    "that are not exist in the trainable dataset"
                )
            return
        if target not in df_cols:
            print("target variable not existed in the dataset columns")
            return

        X=df[trainable_features].copy()
        # Remove target from X if it's in the features
        if target in trainable_features:
            X=X.drop(target,axis=1)
        y=df[target].values.ravel()

        return X,y


    def plot_prediction_vs_actual(self, features, target, 
                                 condition=None, show=True):
        """Plot predicted vs actual values with metrics for multi-feature models."""
        # Get X, y data
        X, y = self.split_dataset_to_X_and_y(features, target, self.df)
        
        # Train model and predict
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y, y_pred, alpha=0.6, label='Predictions')
        
        # Add perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                 label='Perfect Prediction', linewidth=2)
        
        # Formatting
        plt.xlabel(f'True {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'Linear Regression: Predicted vs Actual (R²={r2:.4f}, RMSE={rmse:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if show:
            plt.show()
        
        return plt

    # GOAL: filter the df by a mask or condition
    #       the current implemention is used for Anthocyanin dataset
    #       but changes can be done for adapting it for other datasets.
    def filter_df_by_category(self, df, condition, indicator):
        if condition == 'White and Blue Led':
            values = [f"{letter}{i}" for letter in ['R','G'] for i in range(1, 6)]
        elif condition == 'White Led':
            values = [f"{letter}{i}" for letter in ['R','G'] for i in range(6, 11)]
        elif condition == 'Shade':
            values = [f"{letter}{i}" for letter in ['R','G'] for i in range(11, 16)]
        elif condition == 'Control':
            values = [f"C{i}" for i in range(1, 11)]
        else:
            values = []  # return empty if unknown condition

        # filter the dataframe
        res_df = df[df[indicator].isin(values)]
        return res_df
    
    # PARAMETERS:
    #  - trainable_features: list of trainable_features. our convention is the target
    #    variable is the last variable in this list
    #  - target: the traget variable for prediction
    #  - fitler_df: flag parameter indicate that we filter the dataframe by mask
    #  - filter_cond: if filter_df is True then we use this condition for filter the df.
    #  - filter_indicator: we use this for filter the df.
    def evaluate_regression_models(self,trainable_features,target,
                                   filter_df=None,filter_cond=None,filter_indicator=None):
        if filter_df and filter_cond and filter_indicator:
            df=self.df.copy()
            df=self.filter_df_by_category(df,filter_cond,filter_indicator)
        else:
            df=self.df
        X,y=self.split_dataset_to_X_and_y(trainable_features,target,df)
        results_df=pd.DataFrame()
        results_df=self.model_evaluation(X, y,target,results_df,self.pretty_names_map,
                              self.reg_models,split=False)
        if filter_df and filter_cond:
            results_df[filter_indicator]=filter_cond
        return results_df
    

    ########
    #
    # NDI TABLE - functions
    #
    ########
    # region

    def fix_ndi_df(self,ndi_df):
        col_idx=np.argmax(ndi_df.columns.str.contains("^Unnamed"))
        indexes=ndi_df.loc[col_idx].index
        indexes=indexes[~indexes.str.startswith("Unnamed")]
        ndi_df.index=indexes
        ndi_df=ndi_df.loc[:, ~ndi_df.columns.str.contains("^Unnamed")]
        return ndi_df

    def read_ndi_tables(self):
        df=self.df
        dict_ndi_tables={}
        for idx, row in df.iterrows():
            ndi_df_relative_path=df.loc[idx,'NDI_df']
            ndi_df_absolute_path=f'{self.datasets_paths}/{ndi_df_relative_path}'
            ndi_df=pd.read_csv(ndi_df_absolute_path)
            if ndi_df.columns.str.contains("^Unnamed").any():
                ndi_df=self.fix_ndi_df(ndi_df)
            dict_ndi_tables[idx]=[row,ndi_df]

        # our assumpution is the ndi_df is equal in in number of bands 
        # for all the ndi-tables assoicated to this df.
        # so we extract the last ndi table to get his shape (index,columns)
        first_ndi_df=dict_ndi_tables[0][1]
        col,idx=first_ndi_df.columns,first_ndi_df.index
        return dict_ndi_tables,col,idx
            
    
    # in some cases the original dataframe will contain references for each record
    # to ndi table- so we compute the r2 score for any cell in any one of the ndi-tables
    # PARAMETERS:
    # target_string: string name of the target
    def compute_r2_for_ndi_tables(self,target_string,
                                  model = LinearRegression()):                         
        df=self.df
        # read the ndi tables
        dict_ndi_tables,cols,indexes=self.read_ndi_tables()
        
        #now we compute the regression model for each pair of bands from the ndi table
        r2_ndi_df=pd.DataFrame(columns=cols, index=indexes)
        
        for band1 in list(r2_ndi_df.index):
            for band2 in list(r2_ndi_df.columns):
                if band1==band2:
                    r2_ndi_df.loc[band1,band2]=0
                else:# band1!=band2:
                    X_lst,y_lst=[],[]
                    #for every row in dataframe we read the the table ndi
                    for idx, row in df.iterrows():
                        ndi_df=dict_ndi_tables[idx][1]
                        X_lst.append(ndi_df.loc[band1,band2])
                        y_lst.append(row[target_string])
                        
                    # run the model on this bands
                    X,y=np.array(X_lst).reshape(-1, 1),np.array(y_lst)
                    model.fit(X,y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    r2_ndi_df.loc[band1,band2]=r2
                    
        self.r2_ndi_df=r2_ndi_df.astype('float')
        self.r2_ndi_df_model=str(model)
        self.r2_ndi_df_traget=target_string
        
        return r2_ndi_df       

    # endregion

    ###########
    #
    # PLOT
    #
    ##########

    def get_Anthocyanin_color_map(self):
        color_map = {f"R{i}": "red" for i in range(1, 16)}
        color_map.update({f"C{i}": "red" for i in range(1, 6)})
        color_map.update({f"G{i}": "green" for i in range(1, 16)})
        color_map.update ({f"C{i}": "green" for i in range(6, 11)})
        return color_map
    
    #GOAL:set the markers shapes for the points by their indector value
    def set_Anthocyanin_markers(self):
        letters = ["R", "G"]
        count = 5
        grp_A = [f"{letter}{i}" for letter in letters for i in range(1, count + 1)]
        grp_B = [f"{letter}{i}" for letter in letters for i in range(6, 2*count + 1)]
        grp_C = [f"{letter}{i}" for letter in letters for i in range(11, 3*count + 1)]
        grp_D = [f"C{i}"  for i in range(1, 2*count + 1)]

        # mark the groups . Common marker options:
        # 'o' → circle ,'s' → square, '^' → triangle up, # 'D' → diamond
        markers = {a: 'o' for a in grp_A}
        markers.update({b: 's' for b in grp_B})
        markers.update({c: '^' for c in grp_C})
        markers.update({d: 'D' for d in grp_D})

        labels={'o':'White and Blue Led','s':'White Led',
                '^':'Shade','D':'control'}
        return  markers,labels

    def set_Anthocyanin_categories_shapes_in_plot(self,plt,X,y,df,target,indecator):
        markers,labels=self.set_Anthocyanin_markers()

        y_to_mark={}
        for index, row in df.iterrows():
            point_shape = markers.get(row[indecator], 'v')  # default shape : triangle down
            y_to_mark[row[target]]=point_shape
        
        x_vals=X.values.ravel()
        points=list(zip(x_vals,y))
        #assign the shapes to the points 
        for mark in set(markers.values()):
            x_points=[p[0] for p in points if y_to_mark[p[1]]==mark]
            y_points=[p[1] for p in points if y_to_mark[p[1]]==mark]
            plt.scatter(x_points,y_points, marker=mark, label=labels[mark])

   

    ######################################################

    def find_closest_color_name(self,requested_color):
        min_distance = float('inf')
        closest_name = None
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            # Calculate Euclidean distance
            distance = ((r_c - requested_color[0]) ** 2 +
                        (g_c - requested_color[1]) ** 2 +
                        (b_c - requested_color[2]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        return closest_name
    
    def get_color_name(self,rgb_tuple):
        try:
            # Try to find an exact match in the CSS3 color list
            hex_value = webcolors.rgb_to_hex(rgb_tuple)
            return webcolors.hex_to_name(hex_value, spec='css3')
        except ValueError:
            # If no exact match, find the closest named color
            return self.find_closest_color_name(rgb_tuple)

   
    #GOAL: plot the values of the Anthocyanin in one line
    def plot_Anthocyanin_values_in_one_line_plot(self,plt,y_arr,df,target,indecator,
                                                 trainable_features=None, hp_order='vsh') :
        markers,labels=self.set_Anthocyanin_markers()

        y_to_mark={}
        for index, row in df.iterrows():
            point_shape = markers.get(row[indecator], 'v')  # default shape : triangle down
            y_to_mark[row[target]]=point_shape

        #assign the shapes to the points 
        for mark in set(markers.values()):
            y_points=[y for y in y_arr if y_to_mark[y]==mark]
            dummy_values = np.zeros_like(y_points)
            plt.scatter(dummy_values,y_points, marker=mark, label=labels[mark])
        
        #annoatate the indecator values and hyper-parameter values (as string) 
        ax = plt.gca()
        color_map=self.get_Anthocyanin_color_map()
        for idx, row in df.iterrows():
            point_color = color_map.get(row[indecator], 'black')
            ax.annotate(
                str(row[indecator]),
                (0,row[target]),
                textcoords='offset points',
                xytext=(15,0),
                ha='left',
                color=point_color
            )
            if trainable_features and  hp_order=='vsh': 
                hyperParameter_lst=row[trainable_features[:-1]].values.tolist()
                v,s,h=hyperParameter_lst
                
                # Convert to RGB (result is a tuple of floats in [0.0, 1.0])
                r, g, b= colorsys.hsv_to_rgb(h, s, v)
                # To get standard 8-bit RGB values (0-255 range), multiply by 255
                r_8bit = int(r * 255)
                g_8bit = int(g * 255)
                b_8bit = int(b * 255)
                rgb_color = (r, g, b)
                rgb_8_bit=(r_8bit,g_8bit, b_8bit )
                color_name=self.get_color_name(rgb_8_bit)
                ax.annotate(
                    color_name,
                    (0,row[target]),
                    textcoords='offset points',
                    xytext=(75,0),
                    ha='left',
                    color=rgb_color
                )
        plt.xlabel(target)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=3)


   

    # GOAL:
    #   - plot linear regression of one feature (the first features of 'trainable_features ) 
    # PARAMERTERS:
    #  - indecator: parameter to mark to the points (for example :'catalog id')
    #  - color_map : dictionary of colors for the points
    #  - condition : plot only one group (for example :'white led')
    #  - plot_seperate : plot every one from 4 catgories in different plot
    def plot_linear_regression(self, trainable_features, target,
                           indecator='', color_map={}, condition=None,
                           plot_separate=False,show=True):
        df = self.df.copy()
        
        # Helper function to plot one dataframe
        def plot_single(ax, sub_df, title_suffix=''):
            X, y = self.split_dataset_to_X_and_y(trainable_features, target, sub_df)
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            r2 = r2_score(y, y_pred)
            rmse=root_mean_squared_error(y,y_pred)

            # Plot regression line
            ax.plot(X, y_pred, color='red', label='Regression Line')
            
            # Plot points
            if target == 'Anthocyanin':
                color_map = self.get_Anthocyanin_color_map()
                self.set_Anthocyanin_categories_shapes_in_plot(ax, X, y, sub_df, target, indecator)
            else:
                ax.scatter(X, y, label='Data Points')
            
            # Annotate each point
            for idx, row in sub_df.iterrows():
                point_color = color_map.get(row[indecator], 'black')
                ax.annotate(
                    str(row[indecator]),
                    (row[trainable_features[0]], row[target]),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha='center',
                    color=point_color
                )
            
            # Fit the OLS model
            # Add a constant (intercept) to the independent variables
            X = sm.add_constant(X)

            # Fit the OLS model
            model = sm.OLS(y, X).fit()
            p_values=np.round(model.pvalues[trainable_features[0]],8)
           
            stats_text = f'$R^2$ = {r2:.3f}\n$RMSE$ = {rmse:.3f}\n$p_value$={p_values}'
            ax.annotate(stats_text, 
                        xy=(0.05, 0.97),              # Top-left of the box
                        xycoords='axes fraction', 
                        va='top',                    # Ensures the top of the text starts at 0.9
                        fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5))
            
            ax.set_xlabel(f'{trainable_features[0]} (x)')
            ax.set_ylabel(f'{target} (y)')
            ax.set_title(f'{title_suffix}')
            ax.grid(True)
            ax.legend()
        
        if plot_separate:
            # pdb.set_trace()
            # If plotting separate subplots, define categories
            categories = ['White and Blue Led', 'White Led', 'Shade', 'Control']
            n_rows, n_cols = 2, 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), sharey=True)
            axes = axes.flatten()  # flatten for easy iteration
            for ax, cat in zip(axes, categories):
                sub_df = self.filter_df_by_category(df, cat, indecator)
                plot_single(ax, sub_df, title_suffix=f'{cat}')
            # Hide any unused axes if categories < n_rows*n_cols
            for i in range(len(categories), len(axes)):
                axes[i].axis('off')
            plt.tight_layout()
            if show:
                plt.show()
        else:
            # Filter by condition if provided
            if condition:
                df = self.filter_df_by_category(df, condition, indecator)
            plt.figure(figsize=(10, 5))
            plot_single(plt.gca(), df, title_suffix='Linear Regression')
            if show:
                plt.show()

        return plt

    def plot_heatmap_of_r2_score_of_ndi(self):
        if self.r2_ndi_df is not None:
            ndi_df=self.r2_ndi_df
            model= self.r2_ndi_df_model
            target=self.r2_ndi_df_traget
            
            if ndi_df.columns.str.contains("^Unnamed").any():
                ndi_df=self.fix_ndi_df(ndi_df)
            # 2. Plot the heatmap using Seaborn
            plt.figure(figsize=(10, 6)) # Optional: Adjusts the size of the plot
            sns.heatmap(ndi_df )
            # Use 'df' for raw data heatmap, or 'corr_matrix' for correlation heatmap
            
            # 3. Add titles and labels (optional)
            msg=f'''R2 scores of {model}
                for NDI(band_1,band_2) to {target} for all the records in the in the dataset '''
            plt.title(msg)
            plt.xlabel('Columns: band 1')
            plt.ylabel('Index: band2')
            
            # 4. Display the plot
            plt.show()
        
                    