import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

# Load the pickled model
pickle_in = open('model_hc.pkl', 'rb')
pModel = pickle.load(pickle_in)

@app.get("/")
async def root():
    return {"message": "Welcome to the HAMRO COLLGE API!"}
# Define the endpoint for prediction
@app.get("/predict/")
async def predict(col1: float, col2: float, col3: float):
    try:
        # Create a numpy array from the input data
        input_data = np.array([[col1, col2, col3]])

        # Make predictions using the loaded model
        prediction = pModel.predict(input_data)

        # Return the prediction
        return {"prediction": prediction.tolist()}  # Convert prediction to list for JSON serialization
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
