import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,model_evaluate
from sklearn.metrics import r2_score
from dataclasses import dataclass
from sklearn import metrics

@dataclass
class model_trainer_config:
    model_trainer_config_path=os.path.join('artifacts','model.pkl')

class model_trainer:
    def __init__(self) -> None:
        self.model_trainer_config_obj=model_trainer_config()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            models={
                'logisticRegression':LogisticRegression(),
                'SVM':SVC(kernel='linear'),
                'RandomForest':RandomForestClassifier(n_estimators=100, random_state=42),
                'KNN':KNeighborsClassifier(n_neighbors=3),
                'DecisionTree':DecisionTreeClassifier(criterion='gini', max_depth=None)
            }
            logging.info('Defining required models')

            x_train,y_train,x_test,y_test=[train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]]
            logging.info('splitting data')


            report=model_evaluate(x_train,y_train,x_test,y_test,models)
            logging.info('Fetching report')

            max_score=0
            for k,v in report.items():
                if(v>max_score):
                    #print(v)
                    max_score=report[k]
                    best_model_name=k

            best_model=models[best_model_name]
            logging.info(report)
            save_object(
                self.model_trainer_config_obj.model_trainer_config_path,
                best_model
            )

            y_test_pred=best_model.predict(x_test)
            accuracy_score=metrics.accuracy_score(y_test,y_test_pred)

            
            return accuracy_score

        except Exception as e:
            raise CustomException(e,sys)
        