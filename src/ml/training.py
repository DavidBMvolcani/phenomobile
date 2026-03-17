import numpy as np
import itertools
import pprint
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from heapq import nlargest
import math
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod

from sklearn import datasets
from sklearn.model_selection import train_test_split,KFold, cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor,XGBClassifier

import pdb

from scipy import stats
import statsmodels.api as sm



class Training():

    # PARAMETERS:
    #    - dataset_name: name of the dataset that will be used for training
    #    - config: ConfigManager instance for path resolution and configuration
    #    - fix_method: method that will be chosen when dataset has null values 
    #    - task: string from one of this: ['regression', 'classification'] 
    #      default value is 'regression'
    #    - model: you can set the model you want to fit to data from
    #      options below. if no option will be set - all models will be chosen.
    #      options are ['linear regression']
    def __init__(self, 
        dataset_name, 
        config=None, 
        fix_method='KEEP ROWS', 
        task='regression', 
        model=None,
        logger=None
        ):
        """
        Initialize training class with dataset and configuration.
        
        Args:
            dataset_name: Name of the dataset to use for training
            config: ConfigManager instance for path resolution and configuration
            fix_method: Method to handle null values in dataset (default: 'KEEP ROWS')
            task: Task type - 'regression' or 'classification' (default: 'regression')
            model: Specific model to use (default: all models)
        """
        # Use ConfigManager for all path resolution
        if config is None:
            raise ValueError("config parameter is required")
        
        self.logger = logger
        
        self.config = config
        self.home_path = config.config.get('home_path')
        self.datasets_paths = config.config.get('datasets_path')
       
        # Load dataset using ConfigManager path resolution
        dataset_path = config.get_dataset_path(dataset_name)
        self.df = pd.read_csv(dataset_path)
        self.original_df = self.df.copy()
        
        # Handle null values
        if self.does_df_has_null_values(self.df):
            self.remove_null_from_df(fix_method)
        
        # Set up models based on task
        if task == 'regression':
            # set regression models
            self.set_regression_models(model)
        elif task == 'classification':
            # set supported classification models
            self.set_classification_model()

    #####
    #
    # PREPROCESSING
    #
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
            print("\nConfusion Matrix:\n", cm)
            # Note: For visual confusion matrix, use plotting.plot_confusion_matrix(cm)
  
        
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
    def model_evaluation(self,
        X, 
        y, 
        target, 
        results_df, 
        pretty_names_map, 
        reg_models, 
        split=False,
        test_size=0.2):

        self.logger.info(f"Split dataset to train and test: {split}")
        if split:
            self.logger.info(f"Test size: {test_size}")
        else:
            self.logger.info("No split to train and test")
        #  Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # iterate all over the models and predict the metric results
        for reg_model in reg_models:
            idx=len(results_df)
            results_df.loc[idx,'target']=target
            results_df.loc[idx,'predictable_features']=str(list(X.columns))
            model_name=pretty_names_map[str(type(reg_model))]
            results_df.loc[idx,'model_name']= model_name

            if(split):
                self.logger.info(f"Fitting model {model_name} with split data. test size: {test_size}")
                reg_model.fit(X_train,y_train)
                y_pred = reg_model.predict(X_test)
            else:
                self.logger.info(f"Fitting model {model_name} without split data")
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


    # GOAL: filter the df by a mask or condition
    def filter_df_by_category(self, df, filter_cond,df_column_for_filtering):
        df = df[df[df_column_for_filtering] == filter_cond]
        return df
    
    
    # PARAMETERS:
    #  - trainable_features: list of trainable_features. our convention is the target
    #    variable is the last variable in this list
    #  - target: the traget variable for prediction
    #  - filter_df: flag parameter indicate that we filter the dataframe by mask
    #  - filter_cond: if filter_df is True then we use this condition for filter the df.
    #  - df_column_for_filtering: we use this for filter the df.
    def evaluate_regression_models(self,
        trainable_features,
        target,
        filter_df=None,
        filter_cond=None,
        df_column_for_filtering=None,
        split=False,
        test_size=0.2):
        if filter_df and filter_cond and df_column_for_filtering:
            df=self.df.copy()
            df=self.filter_df_by_category(df,filter_cond,df_column_for_filtering)
        else:
            df=self.df
        X,y=self.split_dataset_to_X_and_y(trainable_features,target,df)
        results_df=pd.DataFrame()
        results_df=self.model_evaluation(X, y,target,results_df,self.pretty_names_map,
                              self.reg_models,split=split,test_size=test_size)
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
        self.r2_ndi_df_target=target_string
        
        return r2_ndi_df       

    # endregion

