import os
import sys
from src.logger import logging 
from src.exception import CustomException

from sqlalchemy import create_engine
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig():
    #-------------------CSV Paths---------------------------
    raw_data_path:str = os.path.join("artifacts" , "dataset.csv")
    train_data_path:str = os.path.join("artifacts"  , "train.csv") 
    test_data_path:str = os.path.join("artifacts"  , "test.csv")

    ##--------------------Database credentials
    driver:str = "ODBC+Driver+17+for+SQL+Server"
    server_name:str = "localhost"
    database:str = "BDdatasets"
    UID:str = "sa"
    PWD:str = "0440"

    connection_string:str = f"mssql+pyodbc://{UID}:{PWD}@{server_name}/{database}?driver={driver}"

class DataIngestion():
    def __init__(self):
        self.dataconfig = DataIngestionConfig()
    
    def initate_data_ingestion(self):
        logging.info("enter the data ingestion method")
        try:
            engine = create_engine(self.dataconfig.connection_string)
            query = "SELECT * FROM Employees"
            
            logging.info("reading database as dataframe")
            
            df = pd.read_sql_query(query , engine)

            
            os.makedirs(os.path.dirname(self.dataconfig.raw_data_path) , exist_ok = True)

            logging.info("saving dataframe as csv")

            df.to_csv(self.dataconfig.raw_data_path , header = True )

            logging.info(" divideding  dataset as trainset and testset ")
            
            train_df ,  test_df = train_test_split(df , test_size = 0.2 , random_state= 42)
            
            return ( train_df , test_df )

        except Exception as e:
            CustomException(e, sys)
