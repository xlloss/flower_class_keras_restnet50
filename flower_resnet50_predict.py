#
# https://medium.com/@bravinwasike18/building-a-deep-learning-model-with-keras-and-resnet-50-9dd6f4eb3351
#

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
import cv2
import pathlib
import numpy as np
import os
import sys

class_flower_name = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')
height, width = 180, 180
training_batch_size = 32
model_name = 'flower_model.036.h5'
pred_img = '/media/Dandelion_yellow_flower.jpg'

image = cv2.imread(pred_img)
image_resized = cv2.resize(image, (height, width))

image = np.expand_dims(image_resized, axis = 0)
print(image.shape)

#Building the deep learning model
dnn_model = tf.keras.Sequential()

imported_model = tf.keras.applications.ResNet50(include_top = False,
    input_shape=(180, 180, 3),
    pooling='avg',classes = 5)

for layer in imported_model.layers:
    layer.trainable = False

#Fine-tuning the imported pre-trained ResNet-50 network
dnn_model.add(imported_model)
dnn_model.add(Flatten())
dnn_model.add(Dense(512, activation = 'relu'))
dnn_model.add(Dense(5, activation = 'softmax'))

save_dir = os.path.join(os.getcwd(), 'saved_models')
filepath = os.path.join(save_dir, model_name)

# Loads the weights
dnn_model.load_weights(filepath)

model_pred = dnn_model.predict(image)
model_pred_class = "model_pred %d" %np.argmax(model_pred)
print(model_pred_class)

predicted_class = class_flower_name[np.argmax(model_pred)]
print("The predicted category is", predicted_class)
