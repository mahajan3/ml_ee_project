import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_data

class predict_pipeline:
    def __init__(self,features) -> None:
        self.features=features
    
    def predict_churn(self):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/datatransformer.pkl'
            model=load_data(model_path)
            preprocessor=load_data(preprocessor_path)
            df=self.features
            scaled_data=preprocessor.transform(df)
            preds=model.predict(scaled_data)
            return preds
        except Exception as e:
            raise CustomException(e,sys)




class customData:
    def __init__(self,AccountWeeks,ContractRenewal,DataPlan,DataUsage,CustServCalls,DayMins,DayCalls,MonthlyCharge,OverageFee,RoamMins) -> None:
        self.AccountWeeks=AccountWeeks,
        self.ContractRenewal=ContractRenewal,
        self.DataPlan=DataPlan,
        self.DataUsage=DataUsage,
        self.CustServCalls=CustServCalls,
        self.DayMins=DayMins,
        self.DayCalls=DayCalls,
        self.MonthlyCharge=MonthlyCharge,
        self.OverageFee=OverageFee,
        self.RoamMins=RoamMins

    def get_dataframe(self):
        try:
            dict1=dict()

            dict1={
                'AccountWeeks':self.AccountWeeks,
                'ContractRenewal':self.ContractRenewal,
                'DataPlan':self.DataPlan,
                'DataUsage':self.DataUsage,
                'CustServCalls':self.CustServCalls,
                'DayMins':self.DayMins,
                'DayCalls':self.DayCalls,
                'MonthlyCharge':self.MonthlyCharge,
                'OverageFee':self.OverageFee,
                'RoamMins':self.RoamMins
            }

            df=pd.DataFrame(dict1)
            print(df)
            return df
        except Exception as e:
            raise CustomException(e,sys)