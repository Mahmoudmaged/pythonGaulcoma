# 1. Library imports

# from numba import jit, cuda
import math   
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plta
from numpy import asarray
from PIL import Image
import os 
import cv2
from PIL import Image
from skimage.transform import resize 
import skimage
from skimage import io 
# 2. Create the app object
app = FastAPI()
# pickle_in = open("galucoma58.pkl","rb")
# model=pickle.load(pickle_in)
# from joblib import dump, load
# model =  load('./g.sav')



import urllib.request
from PIL import Image
  

 
@app.get('/{name}')
def get_name(name: str):
    name = name.replace('@','/')
    name  = 'http://localhost:3000/'+name
    urllib.request.urlretrieve(
    name,
   "./predImg/pp.png")
  
    img = Image.open("./predImg/pp.png")
    # img.show()

    optionsss=tf.saved_model.LoadOptions(
        allow_partial_checkpoint=False,
        experimental_io_device=None,
        experimental_skip_checkpoint=True
    )
    reconstructed_model =tf.keras.models.load_model("my_model", options=optionsss)
    test_image = tf.keras.preprocessing.image.load_img('./predImg/pp.png',
    target_size = (150, 150))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = reconstructed_model.predict(test_image)
    # training_datagen.class_indices
    print(result[0][0])
    print(result[0][0] <= 1)
    if result[0][0] <= 0.90:
        prediction = 'Normal'
    else:
        prediction = 'Glaucoma'
    print(prediction)
   
    return {f'{prediction}'}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)