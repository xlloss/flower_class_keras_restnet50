import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten


def model_get():
    #Building the deep learning model
    dnn_model = tf.keras.Sequential()
    
    imported_model = tf.keras.applications.ResNet50(include_top = False,
                                                    input_shape = (180, 180, 3),
                                                    pooling='avg',classes = 5)
    
    for layer in imported_model.layers:
        layer.trainable = False
    
    #Fine-tuning the imported pre-trained ResNet-50 network
    dnn_model.add(imported_model)
    dnn_model.add(Flatten())
    dnn_model.add(Dense(512, activation = 'relu'))
    dnn_model.add(Dense(5, activation = 'softmax'))
    
    return dnn_model
