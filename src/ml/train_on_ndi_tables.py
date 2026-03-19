import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.ml.training import Training

class TrainOnNdiTables(Training):
    def __init__(
        self,
        dataset_name,
        config=None,
        fix_method='KEEP ROWS',
        task='regression',
        model=None):
        super().__init__(dataset_name, config, fix_method, task, model)

    def fix_ndi_df(self, ndi_df):
        col_idx = np.argmax(ndi_df.columns.str.contains("^Unnamed"))
        indexes = ndi_df.loc[col_idx].index
        indexes = indexes[~indexes.str.startswith("Unnamed")]
        ndi_df.index = indexes
        ndi_df = ndi_df.loc[:, ~ndi_df.columns.str.contains("^Unnamed")]
        return ndi_df

    def read_ndi_tables_from_csv_files(self):
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
    def compute_r2_for_ndi_tables(self, target_string,
                                  model = LinearRegression()):                         
        df=self.df
        # read the ndi tables
        dict_ndi_tables,cols,indexes=self.read_ndi_tables_from_csv_files()
        
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

  