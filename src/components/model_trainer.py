import sys
from src.logger import logging 
from src.exception import CustomException 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from src.utils import *
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.dataconfig = ModelTrainerConfig()
        self.models = {
                        "LogisticRegression": LogisticRegression(), 
                        "AdaBoostClassifier": AdaBoostClassifier() , 
                        "XGBClassifier":XGBClassifier() , 
                    }
        self.params = {
                        "LogisticRegression": {
                                                'C': [ 0.1, 1,],
                                                    'solver': ['liblinear', 'saga'],  # Agrega solver para compatibilidad
                                                'penalty': ['l1', 'l2', 'none'],
                                                'max_iter': [ 200, 500],
                                                
                        },
                        "AdaBoostClassifier": {
                                                'n_estimators': [50, 100,],
                                                'learning_rate': [0.1, 0.5,],
                        },
                        "XGBClassifier": {
                                                'n_estimators': [50, 100],
                                                'learning_rate': [ 0.1, 0.2],
                                                'max_depth': [ 5, 7],
                        }
                        
                    }
    
        self.model_threshold = 0.80
        self.models_report = None
    
    def evaluate_models(self , X_train , y_train , X_test , y_test):
        try:
            report = []
            for model_name , model in self.models.items():
                params = self.params[model_name]
                gs = GridSearchCV(model , params , cv =3 )
                gs.fit(X_train , y_train )

                model.set_params(**gs.best_params_)
                model.fit(X_train , y_train)

                y_test_pred = model.predict(X_test)
                report.append(
                    {   "model_name": model_name , 
                        "precision_score": precision_score(y_test , y_test_pred ),
                        "accuracy_score": accuracy_score(y_test , y_test_pred), 
                        "recall_score":recall_score(y_test, y_test_pred) , 
                        "roc_auc_score":roc_auc_score(y_test , y_test_pred) 
                    }
                )
            self.models_report = pd.DataFrame(report)
            return self.models_report

        except Exception as e:
            raise CustomException(e , sys)
        
    def show_report(self):
        print(self.models_report)
    
    def initate_model_training(self , train_array , test_array ):
        try:
            X_train , y_train , X_test , y_test = (train_array[: , :-1] , train_array[: , -1] , test_array[: , :-1]  , test_array[:, -1])
            
            model_report_df:pd.DataFrame = self.evaluate_models(X_train , y_train , X_test , y_test)
            
            best_model_row = model_report_df.loc[model_report_df["roc_auc_score"].idxmax()]
            best_model_name = best_model_row["model_name"]
            best_model_score = best_model_row["roc_auc_score"]
            best_model = self.models[best_model_name]
            
            if best_model_score <= self.model_threshold:
                raise CustomException("the model performance was no achieve")
            
            logging.info("best model found")

            save_object(self.dataconfig.trained_model_file_path , obj = best_model)

            return (best_model_name ,best_model_score , best_model)

        except Exception as e:
            raise CustomException(e , sys)