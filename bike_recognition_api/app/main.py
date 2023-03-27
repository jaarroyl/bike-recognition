import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import json
import io
import keras
import tensorflow as tf
from skimage import io, img_as_float
from skimage.transform import resize
from skimage.filters import sobel
from PIL import Image, ImageOps
from io import BytesIO
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#from werkzeug.utils import secure_filename
import sys
import os
import glob
import re
#from tempfile import TemporaryDirectory
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
#import cv2


# load trained model
#MODEL_PATH ='../model/bike_class_v5.h5'
MODEL_PATH = '/bike_app/model/bike_class_v5.h5'
model = load_model(MODEL_PATH)

# set target size
final_size = 290

app = FastAPI(
    title = "Bike Identification",
    description = "Implementación del Modelo de Identificación de Bicicletas para HDI Chile",
    version = "1.0.0")

# index
@app.get("/")
async def main():
    content = """    
    <p style="text-align:center;"><img src=https://pbs.twimg.com/profile_images/1145818978213867520/c0GKJ0gh.png alt="Logo" width="450" height="300"></p>
    <h1 style="text-align:center">Implementación Modelo de Identificación de Bicicletas.</h1>
    <p>API REST para HDI Seguros Chile</p>
    """
    return HTMLResponse(content=content)


@app.post("/predict_image/")
def predict_image(image_file: UploadFile = File(...)):

    extension = image_file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg, jpeg or png format!"

    dir_path = os.path.dirname(__file__)
    upload_path = os.path.join(dir_path, 'uploads')

    if not os.path.exists(upload_path):
        os.mkdir(upload_path)
    else:
        pass

    path_save_img = os.path.join(upload_path, image_file.filename)

    img = io.imread(image_file.file, as_gray=True).astype(np.float32)
    img_resize = resize(img, (final_size, final_size), anti_aliasing=True)
    img_sobel = sobel(img_resize)

    #io.imsave(path_save_img, img_sobel)
    plt.imsave(path_save_img, img_sobel)

    final_img = load_img(path_save_img, color_mode='grayscale', target_size=(final_size, final_size))
    final_img = img_to_array(final_img)
    final_img = final_img / 255.
    x = np.expand_dims(final_img, axis=0)

    pred = model.predict(x)
    classes = model.predict(x).argmax(axis=1)
    prob = model.predict(x)[0][0]

    if prob >= 0.95:
        clasif = "Se ha identificado una bicicleta en la fotografia."
    elif prob >=0.8:
        clasif = "Se recomienda revisión humana."
    else:
        clasif = "No se ha identificado una bicicleta en la fotografia."

    shutil.rmtree(upload_path)

    return {#'image':image_file.filename,
            #'pixel mean io_load':float(img.mean()),
            #'pixel mean resize':float(img_resize.mean()),
            #'pixel mean sobel':float(img_sobel.mean()),
            #'pixel mean tf_load':float(x.mean()),
            'Class predict':float(classes[0]),
            'Probability':float(prob),
            'Classifier': clasif} 


#Run the API with uvicorn api_bike:app --reload
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)