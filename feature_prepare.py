import cv2
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

model = tf.keras.models.load_model('cnn_model.h5')
feature_extractor = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[-7].output)

def read_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)
    return img

def extract_name(file_name):
    return file_name.split(".")[0]

def process_folder(folder_path):
    file_data = []
    for file in os.listdir(folder_path):
        name = extract_name(file)
        if file.endswith(".jpg") or file.endswith(".png"):
            img = read_image(os.path.join(folder_path, file))
            feature = feature_extractor.predict(img)
            feature = feature.reshape(1024)
            # print(feature)
            file_data.append(np.insert(feature, 0, name))

    if file_data:
        df = pd.DataFrame(file_data)
        df.to_csv("features.csv", index=False)


process_folder("Traffic_sign_old")
