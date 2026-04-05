import tensorflow as tf
from tensorflow.keras import layers, models
from src.tokenizer import tokenizer
from src.data_cleaning_processing import load_image
from src.img_cap_pair import image_paths, caption_list
import numpy as np
import pandas as pd
import os
import re
#import matplotlib.pyplot as plt

def train_test_split(image_paths, captions, test_size=0.2):
    split = int(len(image_paths) * (1 - test_size))
    return (image_paths[:split], captions[:split]), (image_paths[split:], captions[split:])

(train_imgs,train_caps),(test_imgs,test_caps) = train_test_split(image_paths, caption_list)

def build_tf_dataset(image_paths, captions, batch_size=32, buffer_size=1000):
    seqs = tokenizer.texts_to_sequences(captions)
    seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding='post')

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, seqs))

    def map_func(img_path, cap):
        img = load_image(img_path)
        return (img, cap[:-1]), cap[1:]
    
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = build_tf_dataset(train_imgs, train_caps, tokenizer)
test_dataset = build_tf_dataset(test_imgs, test_caps, tokenizer)
    