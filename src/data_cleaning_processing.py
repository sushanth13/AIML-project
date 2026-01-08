import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import re
#import matplotlib.pyplot as plt

image_dir = "dataset/raw_data/Images"
caption_file = "dataset/raw_data/captions.txt"

def validate_paths(image_dir,caption_file):
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")
    if not os.path.exists(caption_file):
        raise FileNotFoundError(f"Caption file {caption_file} does not exist")

    print("Dataset paths are validated.")
    print("Images directory:", len(os.listdir(image_dir)), "files found.")
    print("Caption file:", caption_file, "found.")

validate_paths(image_dir, caption_file)

def load_and_clean_captions(caption_file):
    captions = {}

    with open(caption_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # ✅ Skip empty or invalid lines
            if not line or '\t' not in line:
                continue

            img_id, caption = line.split('\t', 1)
            img_id = img_id.split("#")[0]

            # Text cleaning
            caption = caption.lower()
            caption = re.sub(r"[^a-z ]", "", caption)
            caption = "<start> " + caption.strip() + " <end>"

            captions.setdefault(img_id, []).append(caption)

    return captions


captions_dict = load_and_clean_captions(caption_file)


def image_preprocessing(image_path, target_size=(224, 224)):
    image = tf.io.read_file(image_path) 
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0
    return image