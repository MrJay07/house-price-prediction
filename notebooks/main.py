import logging
from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel

# Load the trained model
try:
    with open("./notebooks/best_model.pkl", "rb") as f:
        print("Model loaded")
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model file 'model.pkl' not found. Ensure the file exists in the correct path.")

# Initialize FastAPI app
app = FastAPI()

# Define request body format
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float 
    Longitude: float

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def home():
    logger.info("Accessed home endpoint.")
    return {"message": "House Price Prediction API"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        logger.info("Received prediction request with features: %s", features)

        # Prepare input data
        input_data = np.array([[features.MedInc, features.HouseAge, features.AveRooms, features.AveBedrms, features.Population, features.AveOccup, features.Latitude, features.Longitude]])
        
        # Log input data
        logger.debug("Input data prepared: %s", input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Log prediction
        logger.info("Prediction successful. Predicted price: %s", prediction)

        # Return the prediction
        return {"predicted_price": prediction}

    except Exception as e:
        # Log the error
        logger.error("Error occurred during prediction: %s", str(e))

        # Handle errors gracefully
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")