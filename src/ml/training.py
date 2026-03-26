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
    #    - configManager: ConfigManager instance for path resolution and configuration
    #    - fix_method: method that will be chosen when dataset has null values 
    #    - task: string from one of this: ['regression', 'classification'] 
    #      default value is 'regression'
    #    - model: you can set the model you want to fit to data from
    #      options below. if no option will be set - all models will be chosen.
    #      options are ['linear regression']
    def __init__(self, 
        dataset_name, 
        configManager=None, 
        fix_method='KEEP ROWS', 
        task='regression', 
        model=None,
        logger=None,
        ):

        self.logger = logger

        # Use ConfigManager for all path resolution
        if configManager is None:
            try:
                dataset_path =dataset_name
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error loading dataset: {e}")
                raise
        else:
            self.configManager = configManager
            # Load dataset using ConfigManager path resolution
            dataset_path = configManager.get_dataset_path(dataset_name)

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
    
    def set_transformation_for_the_target(self, method,target):
        if method == 'sqrt':
            self.df[target] = np.sqrt(self.df[target])
        elif method == 'log':
            epsilon = 1e-6
            self.df[target] = np.log(self.df[target]+ epsilon)
        elif method == 'none':
            pass
        else:
            raise ValueError(f"Unknown transformation method: {method}")
    
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
        test_size=0.2
        ):

        if self.logger is not None:
            self.logger.info(f"Split dataset to train and test: {split}")
        if split:
            if self.logger is not None:
                self.logger.info(f"Test size: {test_size}")
        else:
            if self.logger is not None:
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
                if self.logger is not None:
                    self.logger.info(f"Fitting model {model_name} with split data. test size: {test_size}")
                reg_model.fit(X_train,y_train)
                y_pred = reg_model.predict(X_test)
            else:
                if self.logger is not None:
                    self.logger.info(f"Fitting model {model_name} without split data")
                reg_model.fit(X,y)
                y_pred = reg_model.predict(X)
                y_test=y
            
            self.y_test = y_test
            self.y_pred = y_pred
            if self.transform_target:
                if self.target_transform_method == 'sqrt':
                    results_df.loc[idx,'R2']= r2_score(np.square(y_test),np.square(y_pred))
                    results_df.loc[idx,'RMSE']= root_mean_squared_error(np.square(y_test),np.square(y_pred))
                elif self.target_transform_method == 'log':
                    results_df.loc[idx,'R2']= r2_score(np.exp(y_test),np.exp(y_pred))
                    results_df.loc[idx,'RMSE']= root_mean_squared_error(np.exp(y_test),np.exp(y_pred))
            else:
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
        test_size=0.2,
        transformed_target=False,
        transformed_target_method='sqrt' # options: 'sqrt', 'log', 'none'
        ):

        # Handle target transformation
        self.transform_target = transformed_target
        self.target_transform_method = transformed_target_method
        if self.transform_target:
            self.set_transformation_for_the_target(
                self.target_transform_method,target)

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
    

    def get_predictions(self):
        try:
            return self.y_test, self.y_pred
        except AttributeError:
            print("No predictions available. Run evaluate_regression_models first.")
            return None, None

   

