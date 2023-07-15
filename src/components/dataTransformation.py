import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import customException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessorObjectFilePath = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    '''
    This Function is responsible for data transformation
    '''
    def __init__(self):
        self.dataTransformationConfig = DataTransformationConfig()
    
    def getDataTransformerObject(self):
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            cat_pipline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessor  = ColumnTransformer(
                [
                    ('num_pipline',num_pipline,numerical_columns),
                    ('cat_columns',cat_pipline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise customException(e,sys)
        
    def initiateDataTransformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('read train and test data')

            logging.info('Obtaining preprocessing object')

            preprocessing_object = self.getDataTransformerObject()
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.dataTransformationConfig.preprocessorObjectFilePath,
                obj=preprocessing_object

            )

            return (
                train_arr,
                test_arr,
                self.dataTransformationConfig.preprocessorObjectFilePath,
            )
        except Exception as e:
            raise customException(e,sys)
