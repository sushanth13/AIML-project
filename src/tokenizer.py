import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import re
#import matplotlib.pyplot as plt

def build_tokenizer(captions_dict, vocab_size=5000):
    all_captions = []
    for cap_list in captions_dict.values():
        all_captions.extend(cap_list)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<unk>", filters='')
    tokenizer.fit_on_texts(all_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer

#tokenizer = build_tokenizer(captions_dict)
#vocab_size = len(tokenizer.word_index)