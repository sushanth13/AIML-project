import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import re
#import matplotlib.pyplot as plt

def ImageEncoder(embed_dim):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(embed_dim, activation='relu')(x)

    model = tf.keras.Model(inputs, outputs, name="ImageEncoder")
    return model