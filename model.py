#%%
# Importing necessary module
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import pandas as pd

#%%
# Define constant
BATCH_SIZE = 32
IMAGE_SIZE = (608, 736)
VALIDATION_RATE = 0.2
DATA_DIR = 'datasets/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')


#%%
# Building model
model = VGG19(
    include_top=False,
    weights="imagenet",
    classes=1000,
)
output = model.layers[-1].output
output = tf.keras.layers.Flatten()(output)
vgg_model = Model(model.input, output)
vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
layer_df = pd.DataFrame(layers, columns=["Layer type", "layer name", "layer trainable"])
layer_df.head(10)

