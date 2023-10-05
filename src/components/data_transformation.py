import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Dataset',"prediction_model.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformed(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["age","bmi","children"]
            categorical_columns = [
                "gender",
                "smoker",
                "region",
                "medical_history",
                "family_medical_history",
                "exercise_frequency",
                "occupation",
                "coverage_level",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("one_hot_encoder",OneHotEncoder(drop='first'))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformed()

            target_feature_train_df = train_df["charges"]
            train_df=train_df.drop(columns=["charges"],axis=1,inplace=False)
            
            target_feature_test_df = test_df["charges"]
            test_df=test_df.drop(columns=["charges"],axis=1,inplace=False)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(train_df)
            input_feature_test_arr=preprocessing_obj.transform(test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)