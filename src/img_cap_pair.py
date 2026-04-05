import tensorflow as tf
from tensorflow.keras import layers, models
from src.data_cleaning_processing import captions_dict, image_dir
import numpy as np
import pandas as pd
import os
import re
#import matplotlib.pyplot as plt

def create_pairs(captions_dict, image_dir):
    image_paths, captions = [], []
    for img, caps in captions_dict.items():
        img_path = os.path.join(image_dir, img)
        for cap in caps:
            image_paths.append(img_path)
            captions.append(cap)
    return image_paths, captions

image_paths, caption_list = create_pairs(captions_dict, image_dir)