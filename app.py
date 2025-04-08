import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template,session,flash,redirect, url_for, session,flash
from werkzeug.utils import secure_filename
# Define a flask app
app = Flask(__name__)
app.secret_key = '1a2b3c4d5e'
# Model saved with Keras model.save()

MODEL_PATH ='skitype.h5'

# Load your trained model
model = load_model(MODEL_PATH)
class_names = ['Dry','Normal','Oil']


def arecadis(fpath):
    image=cv2.imread(fpath)
    example = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #plt.imshow(example)
    image_resized= cv2.resize(image, (256,256))
    image=np.expand_dims(image_resized,axis=0)
    pred=model.predict(image)
    output=class_names[np.argmax(pred)]
    pred = np.argmax(pred)
    print(pred)
    if pred==0:
        return "Dry", '0.html'
    elif pred==1:
        return "Normal", '1.html'
        
    elif pred==2:
        return "Oil", '2.html'
    return output
    
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('./static/upload/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred,output_page = arecadis(fpath=file_path)
        #img_preproc(file_path)
              
        return render_template(output_page, pred_output = pred, user_image = './static/upload/'+filename)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5050) 
