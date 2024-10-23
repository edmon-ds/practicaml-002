from src.components.data_transformation import DataTranformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.utils import *
import pandas as pd

class CustomData:
    def __init__(self ,Education,JoiningYear,City,PaymentTier,Age,Gender,EverBenched,ExperienceInCurrentDomain):
        self.data_user = pd.DataFrame(
            {
              "Education":[Education], 
              "JoiningYear":[JoiningYear] , 
              "City":[City], 
              "PaymentTier":[PaymentTier], 
              "Age":[Age], 
              "Gender":[Gender], 
              "EverBenched":[EverBenched],
              "ExperienceInCurrentDomain":[ExperienceInCurrentDomain],
            }
            )
    def get_data_as_dataframe(self):
            return self.data_user

class PredictPipeline():
    def __init__(self):  
        self.preprocessor = load_object(DataTranformationConfig().preprocessor_obj_file)
        self.model = load_object(ModelTrainerConfig().trained_model_file_path)
    
    def predict(self , user_data):
     try:
          data_transformed = self.preprocessor.transform(user_data)
          preds = self.model.predict(data_transformed)  
          return preds    
     except Exception as e:
          raise CustomException(e , sys)