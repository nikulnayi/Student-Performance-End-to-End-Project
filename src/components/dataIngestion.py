import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.dataTransformation import DataTransformation
from src.components.modelTrainer import ModelTrainer

@dataclass

class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestionConfig=DataIngestionConfig()

    def initiateDataIngestion(self):
        logging.info('Enter the data ingestion method or component')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read Dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path),exist_ok=True)

            df.to_csv(self.ingestionConfig.raw_data_path,index=False,header=True)

            logging.info('Train Test Set initiated')

            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionConfig.test_data_path,index=False,header=True)

            logging.info('Data Ingestion Complete')

            return(
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path,
            )
        except Exception as e:
            raise customException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiateDataIngestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiateDataTransformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiateModelTrainer(train_arr,test_arr))