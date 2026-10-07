from fastapi import FastAPI
from experiments.tune_models import Tune_and_choose_right_model
from src.train import Train_model
from src.predict import predict_price
# from experiments.outlier_analysis import outlier_analysis
from experiments.error_analysis import error_analysis
# from experiments.model_comparison import model_comparison
from experiments.data_distribution_analysis import data_distribution
from experiments.price_outlier_analysis import outlier_detection
from src.data_cleaning import clean_data
from experiments.tune_log_target import tune_log_target



# Create FastAPI app instance
app = FastAPI()


# Define a simple GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/choose_model/")
def choose_model():
    try:
        output = Tune_and_choose_right_model()

        return {
            "message":"success",
            "data":output
        }
    except Exception as e:
        return {
            "message":str(e)
        }


@app.get("/train/")
def Train():
    try:
        output = Train_model()

        return {
            "message":"success",
            "data":output
        }
    except Exception as e:
            return {
                "message":str(e)
            }


@app.post("/predict/")
def Predict(data:dict):
    try:
        output = predict_price(data)

        return {
            "message":"success",
            "data":output
        }
    except Exception as e:
            return {
                "message":str(e)
            }


@app.get("/analysis/")
def Analysis(type:str):
    try:

        if type=='outlier':
            outlier_detection()
        elif type=='cleaning':
            clean_data()
        elif type=='error':
            error_analysis()
        elif type=='data_distribution':
            data_distribution()
        elif type=='tune_log_target':
            tune_log_target()
        return {
            "message":"success",
        }
    
    except Exception as e:
            return {
                "message":str(e)
            }

