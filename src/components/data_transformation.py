import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Save preprocessor.pkl file path
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function si responsible for data trnasformation        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical Pipeline - Handle missing values by SimpleImputer and scale it
            num_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                       ("scaler",StandardScaler())]
                       )

            # Categorical Pipeline - Handle missing values by SimpleImputer,do OneHotEncoder and scale it
            cat_pipeline =Pipeline(
                steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder",OneHotEncoder()),
                       ("scaler", StandardScaler(with_mean=False))]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine Numerical and Categorical Pipeline
            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            scaler = self.get_data_transformer_object()  #scaler

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            x_train = train_df.drop(columns = [target_column_name], axis=1)
            y_train = train_df[target_column_name]

            x_test = test_df.drop(columns = [target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            x_train_arr = scaler.fit_transform(x_train)
            x_test_arr = scaler.transform(x_test)

            # Concatenate arr1 and arr2 column-wise using np.c_
            train_arr = np.c_[x_train_arr, np.array(y_train)]
            test_arr = np.c_[x_test_arr, np.array(y_test)]

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                obj=scaler
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)   


