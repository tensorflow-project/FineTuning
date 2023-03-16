
#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install -q git+https://github.com/keras-team/keras-cv.git')
get_ipython().system('pip install -q tensorflow==2.11.0')
get_ipython().system('pip install pyyaml h5py')

### clone our Github Repository
#get_ipython().system('git clone https://github.com/tensorflow-project/FineTuning')
#get_ipython().run_line_magic('cd', 'FineTuning/models')

import math
import random

import keras_cv
import numpy as np
import tensorflow as tf
from keras_cv import layers as cv_layers
from keras_cv.models.stable_diffusion import NoiseScheduler
from tensorflow import keras
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import os
from google.colab import drive
from PIL import Image
import shutil

### import the different models from our Github repository
#from text_encoder import TextEncoder
#from decoder import Decoder
#from diffusion_model import DiffusionModel
#from stable_diffusion import StableDiffusion
from models.text_encoder import TextEncoder
from models.decoder import Decoder
from models.diffusion_model import DiffusionModel
from models.stable_diffusion import StableDiffusion

### create an instance of the StableDiffusion() class
stable_diffusion = StableDiffusion()

def plot_images(images):
    """function to plot images in subplots
     Args: 
      - images: numpy arrays we want to visualize
    """
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        

def assemble_image_dataset(urls):
    """Downloads a list of image URLs, resizes and normalizes the images, shuffles them, and adds random noise to create a 
    TensorFlow dataset object for them. 
    Args:
    - urls: A list of image URLs to download and use for the dataset.
    Returns:
    - image_dataset: A TensorFlow dataset object containing the preprocessed images.
    Notes:
    - This function assumes that all images have the same dimensions and color channels. 
    """
  
    # Fetch all remote files
    files = [tf.keras.utils.get_file(origin=url) for url in urls]

    # Resize images
    resize = keras.layers.Resizing(height=512, width=512, crop_to_aspect_ratio=True)
    images = [keras.utils.load_img(img) for img in files]
    images = [keras.utils.img_to_array(img) for img in images]
    images = np.array([resize(img) for img in images])

    # The StableDiffusion image encoder requires images to be normalized to the
    # [-1, 1] pixel value range
    images = images / 127.5 - 1

    # Create the tf.data.Dataset
    image_dataset = tf.data.Dataset.from_tensor_slices(images)

    # Shuffle and introduce random noise
    image_dataset = image_dataset.shuffle(50, reshuffle_each_iteration=True)
    image_dataset = image_dataset.map(
        cv_layers.RandomCropAndResize(
            target_size=(512, 512),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(1.0, 1.0),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    image_dataset = image_dataset.map(
        cv_layers.RandomFlip(mode="horizontal"),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return image_dataset

MAX_PROMPT_LENGTH = 77

### our new concept which is later inserted in the different prompts (for training and image generation)
placeholder_token_broccoli = ""
placeholder_token_emoji = ""
placeholder_token_combined = ""


def pad_embedding(embedding):
    ### pad the text embedding to have equal prompt length
    return embedding + (
        [stable_diffusion.tokenizer.end_of_text] * (MAX_PROMPT_LENGTH - len(embedding))
    )


stable_diffusion.tokenizer.add_tokens(placeholder_token_broccoli)
stable_diffusion.tokenizer.add_tokens(placeholder_token_emoji)
stable_diffusion.tokenizer.add_tokens(placeholder_token_combined)


def assemble_text_dataset(prompts, placeholder_token):
    """ creates text dataset consisting of prompt embeddings"""
    ### inserts our placeholder_token into the different prompts
    prompts = [prompt.format(placeholder_token) for prompt in prompts]
    ### prompts are tokenized and encoded and then added to the embedding
    embeddings = [stable_diffusion.tokenizer.encode(prompt) for prompt in prompts]
    embeddings = [np.array(pad_embedding(embedding)) for embedding in embeddings]
    ### creates a dataset consisting of the different prompt embeddings and shuffles it
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
    return text_dataset
