from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import warnings
warnings.filterwarnings("ignore")


print("start data ingestion")
train_df , test_df = DataIngestion().initate_data_ingestion()
print("start the data transformation")
train_array , test_array = DataTransformation().initiate_data_transformation(train_df , test_df)

print("start the model training")
model_trainer = ModelTrainer()

best_model_name ,best_model_score , best_model = model_trainer.initate_model_training(train_array , test_array)
print("list of the model trained and their score")
model_trainer.show_report()

print()
print(f"the best model selected  is {best_model_name}  its aux score is :  {best_model_score}")