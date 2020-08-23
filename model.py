#%%
# Importing necessary module
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19

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
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)


print(model)







