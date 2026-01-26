import tensorflow as tf
from tensorflow.keras import layers, models
from src.data_cleaning_processing import captions_dict, image_paths
from src.tokenizer import tokenizer
import numpy as np
import pandas as pd
import os
import re
#import matplotlib.pyplot as plt

def dataset_stats(captions_dict, img_paths, tokenizer):
    caption_lenghts = []
    for caps in captions_dict.values():
        for cap in caps:
            caption_lenghts.append(len(cap.split()))
    #return caption_lenghts
    print("\nDataset Statistics:")
    print("-" * 30)
    print("Total Images:", len(img_paths))
    print("Total captions:", len(caption_lenghts))
    print("Vocabulary Size:", len(tokenizer.word_index)+1)
    print("Average Caption Length:", np.mean(caption_lenghts))
    print("Max Caption Length:", np.max(caption_lenghts))

dataset_stats(captions_dict, image_paths, tokenizer)