import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import sys
import os
from src.exception import CustomException
import dill
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.logger import logging

def save_object(filepath,obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj,fileobj)
        
    except Exception as e:
        raise CustomException(e,sys)
    

def model_evaluate(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        logging.info('Creating report with scores')
        for k,v in models.items():
            m=models[k]
            m.fit(x_train,y_train)
            y_train_pred=m.predict(x_train)
            y_test_pred=m.predict(x_test)

            accuracy_score=r2_score(y_test,y_test_pred)
            report[k]=accuracy_score
        logging.info('Report created')
        return report
    except Exception as e:
        raise CustomException(e,sys)        
    
def load_data(filepath):
    try:
        with open(filepath,'rb') as obj:
            a=dill.load(obj)
            return a
        
    except Exception as e:
        raise CustomException(e,sys)