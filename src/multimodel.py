import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import re
#import matplotlib.pyplot as plt
from src.text_encoding import TextEncoder
from src.img_encoding import ImageEncoder


class MultimodalFusion(layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim)
        
    def call(self, image_features, text_features):
        # Expand dimensions of text_features to match image_features for attention
        image_features = tf.expand_dims(image_features, axis=1)
        text_features = tf.expand_dims(text_features, axis=1)
        fused_features = self.att(text_features, image_features, image_features)
        return tf.squeeze(fused_features, axis=1)
    
def MultimodalTransformer(vocab_size, embed_dim):
    # Image feature input
    image_inputs = layers.Input(shape=(224,224,3))

    # Text feature input
    text_inputs = layers.Input(shape=(None,))

    # Image feature extraction
    image_features = ImageEncoder(embed_dim)(image_inputs)
    
    
    # Text feature extraction
    text_features = TextEncoder(vocab_size, embed_dim)(text_inputs)

    
    # Fusion
    fusion_layer = MultimodalFusion(embed_dim)
    fused_features = fusion_layer(image_features, text_features)
    
    # Output layer
    outputs = layers.Dense(1, activation="softmax")(fused_features)
    
    return models.Model(inputs=[image_inputs, text_inputs], outputs=outputs, name="MultimodalTransformer")