import os
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn
from scipy import ndimage
import warnings
import argparse

warnings.filterwarnings('ignore')

app = FastAPI(title='Digit Recognition App')

# Function to predict digit using the model
def predict_digit(digit_model, input_data):
    pred = digit_model.predict(input_data.reshape(1, -1))
    return str(np.argmax(pred))

# Function to format the image
def format_image(image):
    img_gray = image.convert('L')
    img_resized = img_gray.resize((28, 28))
    image_arr = np.array(img_resized) / 255.0

    cy, cx = ndimage.center_of_mass(image_arr)
    rows, cols = image_arr.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_centered = ndimage.shift(image_arr, (shifty, shiftx), cval=0)
    
    # Flatten the image and return it
    return img_centered.flatten()

# Parsing the command line argument
def get_model_path():
    parser = argparse.ArgumentParser(description="Load model for digit prediction")
    parser.add_argument("model_path", type=str, help="Path to the model file")
    args = parser.parse_args()
    return args.model_path

# Loading the model saved on the disk
def load_model(file_path):
    model = load_model(file_path)
    return model

@app.post('/predict')
async def predict_digit(file: UploadFile = File(...)):
    # Read the uploaded file
    file_contents = await file.read()
    # Open and resize the image
    img = Image.open(io.BytesIO(file_contents))
    # Format the image
    img_array = format_image(img)
    # Get the model path and load the model
    model_path = get_model_path()
    digit_model = load_model(model_path)
    # Make a prediction
    prediction = predict_digit(digit_model, img_array)
    return {"digit": prediction}


if __name__ == '__main__':
    uvicorn.run("CH20B081_A06_Task-2:app", host="0.0.0.0", port=8000, reload=True)