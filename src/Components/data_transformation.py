import os 
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join("artifacts","preprocessor.pkl")

class DataTrasformation:
    def __init__(self):
        self.Data_Transformation_Config=DataTransformationConfig()

    def get_transformation(self):
        try:
            numerical_columns=["tenure","MonthlyCharges","TotalCharges","SeniorCitizen"]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )
          

            logging.info("Numerical Colum convert into text")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipelines",num_pipeline,numerical_columns)
                ],
                remainder="passthrough"
            )

            return preprocessor
        except Exception as es:
            raise CustomException (es,sys)
        

    def initiate_data_transformation(self,train_data,test_data):
        try:
            train_data_df=pd.read_csv(train_data)
            test_data_df=pd.read_csv(test_data)
           

            # add me

            train_data_df["TotalCharges"] = pd.to_numeric(train_data_df["TotalCharges"], errors="coerce")
            test_data_df["TotalCharges"] = pd.to_numeric(test_data_df["TotalCharges"], errors="coerce")
            
            logging.info("Read the train and test dataset")
            
            preprocessor_obj=self.get_transformation()
            target_column_name="Churn"

            #splling data

            x_train=train_data_df.drop(columns=[target_column_name])
            y_train=train_data_df[target_column_name]

            x_test=test_data_df.drop(columns=[target_column_name])
            y_test=test_data_df[target_column_name]


            logging.info("Applying the preprocessor on training and testingf dataaset")

            model_fit_train=preprocessor_obj.fit_transform(x_train)
            model_fit_test=preprocessor_obj.transform(x_test)

            save_object(
                file_path=self.Data_Transformation_Config.preprocessor_obj_file,
                obj=preprocessor_obj
            )

            return (
                model_fit_train,
                y_train.values.ravel(),
                model_fit_test,
                y_test.values.ravel()
            )
        except Exception as s:
            raise CustomException (s,sys)
        

if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion

    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTrasformation()
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    print("Data Transformation Completed")
    print("X_train shape:", X_train.shape)
