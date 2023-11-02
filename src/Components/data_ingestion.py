import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.Components.data_transformation import data_transformation
from src.Components.model_trainer import model_trainer

@dataclass
class dataingestion_config:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')


class dataingestion:
    def __init__(self):
        self.ingestion_config=dataingestion_config()
    
    def initiate_ingest(self):
        try:
            df=pd.read_csv('Notebook\Data\churn.csv')
            logging.info('Data has been read')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Created path')
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=1)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Train test splitting done')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=dataingestion()
    train_path,test_path=obj.initiate_ingest()
    obj1=data_transformation()
    train_arr,test_arr,_=obj1.initiate_data_transformation(train_path,test_path)
    obj2=model_trainer()
    accuracy_score=obj2.initiate_model_trainer(train_arr,test_arr)
    print(accuracy_score)