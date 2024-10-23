import sys 
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import * 

from dataclasses import dataclass

@dataclass
class DataTranformationConfig():
    preprocessor_obj_file:str = os.path.join("artifacts" , "preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.dataconfig = DataTranformationConfig()
        self.numerical_features =  ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain']
        self.categorical_features =  ['Education', 'City', 'Gender', 'EverBenched']
        self.label = "LeaveOrNot"
    def get_preprocessor(self):
        try:
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer" , SimpleImputer(strategy="mean") ) , 
                    ("scaler" , StandardScaler() ) 
                        ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                        ("imputer" , SimpleImputer(strategy = "most_frequent")),
                        ("one_hot_encoder" , OneHotEncoder()) 

                        ]
            )

            preprocessor = ColumnTransformer(
                        [
                    ("numerical_pipeline" , numerical_pipeline  , self.numerical_features) , 
                    ("categorical_pipeline" , categorical_pipeline , self.categorical_features )
                        ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e , sys)
        
    def initiate_data_transformation(self,train_df:pd.DataFrame , test_df:pd.DataFrame ):
        logging.info("enter to initate_data_transformation function")
        try:
            
            preprocessor = self.get_preprocessor()
            logging.info("separating labels from datasets")

            train_input_raw = train_df.drop(columns = [self.label])
            train_label = train_df[self.label]

            test_input_raw = test_df.drop(columns=[self.label])
            test_label = test_df[self.label]

            logging.info("applying preprocessing to the dataset")

            train_input_array = preprocessor.fit_transform(train_input_raw)
            test_input_array = preprocessor.transform(test_input_raw)

            train_array = np.hstack((train_input_array , train_label.values.reshape(-1,1)))
            test_array = np.hstack((test_input_array, test_label.values.reshape(-1,1)))
            
            logging.info("saving preprocessor")
            
            save_object(file_path = self.dataconfig.preprocessor_obj_file , obj= preprocessor)

            return (train_array , test_array )

        except Exception as e:
            raise CustomException(e , sys)