import pandas as pd
import numpy as np
import sys
import os
from src.exception import CustomException
import dill

def save_object(filepath,obj):
    try:
        dir_path=os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj,fileobj)
        
    except Exception as e:
        raise CustomException(e,sys)