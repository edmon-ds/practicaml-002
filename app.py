from flask import Flask ,  request  , render_template
from src.pipelines.predict_pipeline import CustomData , PredictPipeline
from src.logger import logging 
from src.exception import CustomException
import sys


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict" , methods = ["GET" , "POST" ]) 
def predict_datapoint():
    if request.method =="GET":
        return render_template("predict.html")
    else:
        try:
            user_data = CustomData(
                Education = request.form.get("Education") , 
                 JoiningYear = request.form.get("JoiningYear") ,
                 City = request.form.get("City") ,
                 PaymentTier =  request.form.get("PaymentTier"),
                 Age = request.form.get("Age") ,
                 Gender = request.form.get("Gender") ,
                 EverBenched = request.form.get("EverBenched") ,
                 ExperienceInCurrentDomain = request.form.get("ExperienceInCurrentDomain") ,      
            )
            user_data_df = user_data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            preds = predict_pipeline.predict(user_data_df)
            return render_template("predict.html" , results = preds[0] )
        except Exception as e:
            raise CustomException(e , sys)

if __name__ =="__main__":
    app.run(host = "0.0.0.0" , port = 8080 , debug = True)