###############################################################################################################
# ref : https://medium.com/@bravinwasike18/building-a-deep-learning-model-with-keras-and-resnet-50-9dd6f4eb3351
###############################################################################################################

from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import tensorflow as tf
import os
import numpy as np
import pathlib

# Learning Rate Schedule
def lr_schedule(epoch):
    learning_rate = 0.5e-3
    if epoch > 40:
        learning_rate *= 0.5e-7
    elif epoch > 30:
        learning_rate *= 0.5e-6
    elif epoch > 20:
        learning_rate *= 0.5e-5
    elif epoch > 10:
        learning_rate *= 0.5e-4
    print('Learning rate: ', learning_rate)
    return learning_rate

epoch_cnt = 50

#Downloading the dataset
flowers_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

flowers_data = tf.keras.utils.get_file('flower_photos', origin = flowers_url, untar = True)
flowers_data = pathlib.Path(flowers_data)
print(flowers_data)

#Displaying the images
all_sunflowers = list(flowers_data.glob('sunflowers/*'))
print(all_sunflowers[1])

#Pre-processing the image dataset
height,width = 180, 180
training_batch_size = 32

train_set = tf.keras.preprocessing.image_dataset_from_directory(
	flowers_data,
	validation_split = 0.2,
	subset = "training",
	seed = 123,
	image_size = (height, width),
	batch_size = training_batch_size)

validation_set = tf.keras.preprocessing.image_dataset_from_directory(
	flowers_data,
	validation_split = 0.2,
	subset = "validation",
	seed = 123,
	image_size = (height, width),
	batch_size = training_batch_size)

#Building the deep learning model
dnn_model = tf.keras.Sequential()

imported_model= tf.keras.applications.ResNet50(include_top = False,
    input_shape = (180, 180, 3),
    pooling = 'avg',classes = 5,
    weights = 'imagenet')

for layer in imported_model.layers:
    layer.trainable = False

#Fine-tuning the imported pre-trained ResNet-50 network
dnn_model.add(imported_model)
dnn_model.add(Flatten())
dnn_model.add(Dense(512, activation = 'relu'))
dnn_model.add(Dense(5, activation = 'softmax'))

#Getting the model summary
dnn_model.summary()
plot_model(dnn_model, to_file = 'model.png', show_shapes = True, show_dtype = True)

#Compiling the deep learning model
dnn_model.compile(loss = 'sparse_categorical_crossentropy',
    #optimizer=Adam(lr=lr_schedule(0)),
    optimizer = 'adam',
    metrics = ['accuracy'])

#Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'flower_model.{epoch:03d}.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)

#Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath = filepath,
                             monitor = 'val_accuracy',
                             verbose = 1,
                             save_weights_only = True,
                             save_best_only = True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1),
                               cooldown = 0,
                               patience = 5,
                               min_lr = 0.5e-7)

#callbacks = [checkpoint, lr_reducer, lr_scheduler]
history = dnn_model.fit(train_set,
    validation_data=validation_set,
    epochs = epoch_cnt, verbose = 1, callbacks = [checkpoint])
