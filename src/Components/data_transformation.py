import pandas as pd
import numpy as np
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class data_transformation_config:
    data_transformation_config_path=os.path.join('artifacts','datatransformer.pkl')

class data_transformation:
    def __init__(self) -> None:
        self.data_transformation_config_obj=data_transformation_config()
    
    def data_transformation_pipeline(self):
        try:
            features=[ 'AccountWeeks','ContractRenewal','DataPlan','DataUsage','CustServCalls','DayMins','DayCalls','MonthlyCharge','OverageFee','RoamMins']
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            target_feature='Churn'
            input_train_data=train_df.drop([target_feature],axis=1)
            target_train_data=train_df[target_feature]

            input_test_data=test_df.drop([target_feature],axis=1)
            target_test_data=test_df[target_feature]

            preprocessor_obj=self.data_transformation_pipeline()

            train_data_input_arr=preprocessor_obj.fit_transform(input_train_data)
            test_data_input_arr=preprocessor_obj.transform(input_test_data)

            train_arr=np.c_[train_data_input_arr,np.array(target_train_data)]
            test_arr=np.c_[test_data_input_arr,np.array(target_test_data)]

            save_object(
                self.data_transformation_config_obj.data_transformation_config_path,
                preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config_obj.data_transformation_config_path
            )
        except Exception as e:
            raise CustomException(e,sys)






