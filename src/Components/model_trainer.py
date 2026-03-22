import os
import sys
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")   # 🔥 VERY IMPORTANT
mlflow.set_experiment("TELECOM CUSTOMER CHURN PREDICTION")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn .metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object

from dataclasses import dataclass

@dataclass 
class ModelTrainerConfig:
    model_path:str=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.Model_trainer_config=ModelTrainerConfig()

    def evalute_model(self,y_true,y_pred):
        return{

            "Accuracy_Score":accuracy_score(y_true,y_pred),
            "f1":f1_score(y_true,y_pred),
            "precision":precision_score(y_true,y_pred),
            "Recall":recall_score(y_true,y_pred),
            
        }
    def initiate_model_trainer(self,x_train,y_train,x_test,y_test):
        try:
            model={
                "Logistic Regression":LogisticRegression(max_iter=1000),
                "Random Forest":RandomForestClassifier(n_estimators=100)
            }
            best_model=None
            best_recall=0
            best_model_name=""

            mlflow.set_experiment("TELECOM CUSTOMER CHURN PREDICTION") 

            for model_name,model in model.items():
                with mlflow.start_run(run_name=model_name):
                    logging.info(f"Training Model:{model_name}")
                    model.fit(x_train,y_train)

                    y_pred=model.predict(x_test)
                    # y_pred=model.predict_proba(x_test)[:1]

                    metrics=self.evalute_model(y_test,y_pred)

                    #log parameter 
                    mlflow.log_param("model_name",model_name)
                    #log metrics
                    for key,values in metrics.items():
                        mlflow.log_metric(key,float(values))

                    #log model
                    mlflow.sklearn.log_model(model,model_name)

                    print(f"\nModel:{model_name}")
                    for key,values in metrics.items():
                        print(f"{key}:{round(values,4)}")

                    if metrics["Recall"]>best_recall:
                        best_recall=metrics["Recall"]
                        best_model=model
                        best_model_name=model_name

            print(f"\nBest Model Is Selected :{best_model_name}")

            save_object(
                file_path=self.Model_trainer_config.model_path,
                obj=best_model
            )



        except Exception as es:
            raise CustomException (es,sys)
if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion
    from src.Components.data_transformation import DataTrasformation

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTrasformation()
    X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()
    best_model = trainer.initiate_model_trainer(
        X_train, y_train, X_test, y_test
    )

    print("\nTraining Completed Successfully")