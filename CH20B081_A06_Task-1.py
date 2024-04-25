import os
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn
import warnings
import argparse

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title='Digit Recognition App')

# Function to get model path from command line arguments
def get_model_path():
    parser = argparse.ArgumentParser(description="Load digit recognition model")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    args = parser.parse_args()
    return args.model_path

# Function to load model from given path
def load_model(file_path):
    digit_model = load_model(file_path)
    return digit_model

# Function to predict digit using the model
def predict_digit(digit_model, input_data):
    pred = digit_model.predict(input_data.reshape(1, -1))
    return str(np.argmax(pred))

@app.post('/predict')
async def predict_digit_model(file: UploadFile = File(...)):
    # Read the uploaded file
    file_contents = await file.read()
    # Open and resize the image
    img = Image.open(io.BytesIO(file_contents)).resize((28, 28)).convert('L')
    # Convert the image to a numpy array and normalize it
    img_array = np.array(img)/255.0
    # Flatten the array
    img_array = img_array.flatten()
    # Get the model path and load the model
    model_path = get_model_path()
    digit_model = load_model(model_path)
    # Make a prediction
    prediction = predict_digit(digit_model, img_array)
    return {"digit": prediction}

if __name__ == '__main__':
    uvicorn.run("CH20B081_A06_Task-1:app", host="0.0.0.0", port=8000, reload=True)