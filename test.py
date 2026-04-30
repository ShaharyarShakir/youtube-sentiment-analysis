import mlflow
import os

# Must set these BEFORE calling any mlflow function
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:3900"
os.environ["AWS_ACCESS_KEY_ID"]      = "mlflow-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "mlflow-secret-key"
os.environ["AWS_DEFAULT_REGION"]     = "garage"

mlflow.set_tracking_uri("http://localhost:5000/")

model_uri = "models:/yt_chrome_plugin_model/staging"
info = mlflow.models.get_model_info(model_uri)
print(info.signature)