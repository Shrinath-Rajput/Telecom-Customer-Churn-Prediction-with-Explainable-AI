import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_Data_Path:str=os.path.join("artifacts","train.csv")
    test_Data_Path:str=os.path.join("artifacts","test.csv")
    Row_data_path:str=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.Ingestion_data_Config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Enter the dataset and method  that points")
        try:
            df=pd.read_csv('.\\Dataset\\telo.csv')

            logging.info("Data is Read Here ")

            os.makedirs(os.path.dirname(self.Ingestion_data_Config.train_Data_Path),exist_ok=True)
            df.to_csv(self.Ingestion_data_Config.Row_data_path,index=False,header=True)
            logging.info("Data is fully read")

            #train and test data 

            train_data,test_data=train_test_split(df,random_state=42,test_size=0.2)
            train_data.to_csv(self.Ingestion_data_Config.train_Data_Path,index=False,header=True)
            test_data.to_csv(self.Ingestion_data_Config.test_Data_Path,index=False,header=True)

            return(
                self.Ingestion_data_Config.train_Data_Path,
                self.Ingestion_data_Config.test_Data_Path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__== "__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    print("Train file saved at:",train_path)
    print("Test file is saved",test_path)