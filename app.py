import tensorflow as tf
from tensorflow import keras
import streamlit as st
from PIL import Image,ImageOps
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_option('deprecation.showfileUploaderEncoding',False)
#@st.cache(allow_output_mutation = True)

model = keras.models.load_model("model3.h5")

labels = {0:'Apple: Black Rot',
 1:'Apple: Cedar apple rust',
 2:'Apple: Healthy',
 3:'Apple: Scab',
 4:'Bell Pepper: Bacterial_Spot',
 5:'Bell Pepper: Healthy',
 6:'Cherry: Healthy',
 7:'Cherry: Powdery Mildew',
 8:'Corn: Cercospora Leaf Spot',
 9:'Corn: Common Rust',
 10:'Corn: Healthy',
 11:'Corn: Northern Leaf Blight',
 12:'Grape: Black Measles',
 13:'Grape: Black Rot',
 14:'Grape: Healthy',
 15:'Grape: Leaf Blight',
 16:'Peach: Bacterial Spot',
 17:'Peach: Healthy',
 18:'Potato: Early Blight',
 19:'Potato: Healthy',
 20:'Potato: Late Blight',
 21:'Strawberry: Healthy',
 22:'Strawberry: Leaf Scorch',
 23:'Tomato: Bacterial Spot',
 24:'Tomato: Healthy',
 25:'Tomato: Leaf Mold',
 26:'Tomato: Mosaic Virus',
 27:'Tomato: Spider Mites',
 28:'Tomato: Target Spot',
 29:'Tomato: Yellow Leaf Curl Virus'}

def preprocess(img):
  img = ImageOps.fit(img,(224, 224))
  img = np.asarray(img)
  img = img/255.0
  img = np.expand_dims(img,axis = 0)
  return img


title = st.title('Plant Disease Detection')

file1 = st.file_uploader("Please uploade plant leaf image",type = ['jpg','png'])
if file1 is not None:
  image = Image.open(file1)
  st.image(image,use_column_width= True)
  img = preprocess(image)
  pred = np.argmax(model.predict(img))
  string = "Plant and disease mostly is:    " + labels[pred]
  st.success(string)