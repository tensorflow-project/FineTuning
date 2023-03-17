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
import sys
from google.colab import drive
from PIL import Image
import shutil

###select path to find the models used here
py_file_location = "/content/FineTuning"
sys.path.append(os.path.abspath(py_file_location))
py_file_location = "/content/FineTuning/models"
sys.path.append(os.path.abspath(py_file_location))


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
      - images (array): numpy arrays we want to visualize
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
    - urls (list): A list of image URLs to download and use for the dataset

    Returns:
    - image_dataset (ds): A TensorFlow dataset object containing the preprocessed images

    Notes:
    - This function assumes that all images have the same dimensions and color channels
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

### our new concept which is later inserted in the different prompts (for training and image generation).
###The goal is to create an embedding for our placeholder_token
placeholder_token = "<my-broccoli-token>"


def pad_embedding(embedding):
    """Pads the input embedding with the end-of-text token to ensure that it has the same length as the maximum prompt length.

    Args:
    - embedding (list): A list of tokens representing the input embedding

    Returns:
    - padded_embedding (list): A list of tokens representing the padded input embedding
    """
    return embedding + (
        [stable_diffusion.tokenizer.end_of_text] * (MAX_PROMPT_LENGTH - len(embedding))
    )

### Add our placeholder_token to our stable_diffusion Model
stable_diffusion.tokenizer.add_tokens(placeholder_token)


def assemble_text_dataset(prompts, placeholder_token):
    """Creates a text dataset consisting of prompt embeddings. 
    
    Args:
    - prompts (str): A list of string prompts to be encoded and turned into embeddings
    - placeholder_token (str): our placeholder token
  
    Returns:
    - text_dataset: A text dataset containing the prompt embeddings
    """
    ### inserts our placeholder_token into the different prompts
    prompts = [prompt.format(placeholder_token) for prompt in prompts]

    ### prompts are tokenized and encoded and then embeddings are padded
    embeddings = [stable_diffusion.tokenizer.encode(prompt) for prompt in prompts]
    embeddings = [np.array(pad_embedding(embedding)) for embedding in embeddings]

    ### creates a dataset consisting of the different prompt embeddings and shuffles it
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
    return text_dataset


def assemble_dataset(urls, prompts, placeholder_token):
    """ Assembles a TensorFlow Dataset containing pairs of images and text prompts.

    Args:
    - urls: A list of URLs representing the image dataset
    - prompts: A list of text prompts corresponding to the images
    - placeholder_token: A string token representing the location where the prompt text will be inserted in the final text

    Returns:
    - A TensorFlow Dataset object containing pairs of images and their corresponding text prompts
    """
    ### creating the image and test dataset
    image_dataset = assemble_image_dataset(urls)
    text_dataset = assemble_text_dataset(prompts, placeholder_token)
    
    ### repeat both datasets to get several different combinations of images and text prompts
    # the image dataset is quite short, so we repeat it to match the length of the text prompt dataset
    image_dataset = image_dataset.repeat()

    # we use the text prompt dataset to determine the length of the dataset.  Due to
    # the fact that there are relatively few prompts we repeat the dataset 5 times.
    # we have found that this anecdotally improves results.
    text_dataset = text_dataset.repeat(5)
    return tf.data.Dataset.zip((image_dataset, text_dataset))


### create a dataset consisting of happy broccoli stickers and happy prompts
happy_ds = assemble_dataset(
    urls = [
        "https://i.imgur.com/9zAwPyt.jpg",
        "https://i.imgur.com/qCNFRl4.jpg",
        "https://i.imgur.com/YLnylVr.png",
    ],
    prompts = [
        "a photo of a happy {}",
        "a photo of {}",
        "a photo of one {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of a cool {}",
        "a rendition of the {}",
        "a nice sticker of a {}",
        "a sticker of a {}",
        "a sticker of a happy {}",
        "a sticker of a lucky {}",
        "a sticker of a lovely {}",
        "a sticker of a {} in a positive mood",
        "a pixar chracter of a satisfied {}",
        "a disney character of a positive {}",
        "a sticker of a delighted {}",
        "a sticker of a joyful {}",
        "a sticker of a cheerful {}",
        "a drawing of a glad {}",
        "a sticker of a merry {}",
        "a sticker of a pleased {}",
    ],
    placeholder_token = placeholder_token
)

### create a dataset consisting of broccoli in love stickers and matching prompts
love_ds = assemble_dataset(
    urls = [
        "https://i.imgur.com/hFqqp3p.jpg",
        "https://i.imgur.com/uGkSrzg.jpg",
        "https://i.imgur.com/zTVXw0D.png",
        "https://i.imgur.com/XJxG3f0.png", 
    ],
    prompts = [
        "a photo of a {} in love",
        "a photo of {}",
        "a photo of one {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of an amorous {}",
        "a rendition of the {}",
        "a nice sticker of a {}",
        "a sticker of a {}",
        "a sticker of a {} in love",
        "a sticker of an amorous {}",
        "a sticker of a lovely {}",
        "a sticker of a {} in a positive mood",
        "a pixar chracter of a {} in love",
        "a disney character of a positive {}",
        "a sticker of a delighted {}",
        "a sticker of a joyful {}",
        "a drawing of a {} in love",
        "a drawing of a glad {}",
        "a sticker of a loving {}",
        "a sticker of a pleased {}",
    ],
    placeholder_token = placeholder_token
)

### create a dataset consisting of sad broccoli stickers and matching prompts
sad_ds = assemble_dataset(
    urls = [
        "https://i.imgur.com/hlkuxBX.jpg",
        "https://i.imgur.com/kPH9XIh.jpg",
        "https://i.imgur.com/OR2oxyK.jpg",
    ],
    prompts = [
        "a photo of a sad {}",
        "a photo of {}",
        "a photo of one {}",
        "a photo of an unhappy {}",
        "a good photo of a {}",
        "a photo of the unhappy {}",
        "a photo of a depressed {}",
        "a rendition of the sad {}",
        "a nice sticker of a miserable {}",
        "a sticker of a {}",
        "a sticker of a downhearted {}",
        "a sticker of a sorrowful {}",
        "a sticker of an unhappy {}",
        "a sticker of a {} in a negative mood",
        "a pixar chracter of a depressed {}",
        "a disney character of a negative {}",
        "a sticker of a mourning {}",
        "a sticker of a grieving {}",
        "a drawing of a sad {}",
        "a drawing of a miserable {}",
        "a sticker of a sorrowful {}",
        "a sticker of a sobbing {}",
    ],
    placeholder_token=placeholder_token
)

### create a dataset consisting of sad broccoli stickers and matching prompts
angry_ds = assemble_dataset(
    urls = [
        "https://i.imgur.com/mZswnIx.jpg",
        "https://i.imgur.com/TmlHZRY.png",
        "https://i.imgur.com/BmVIZlO.png",
    ],
    prompts = [
        "A photo of an angry {}",        
        "A photo of a {} with a furious expression",        
        "A photo of a {} in a fit of rage",        
        "A photo of a {} yelling in anger",        
        "A photo of a {} with a scowling face",        
        "A photo of a {} clenching their fists in anger",        
        "A photo of a {} looking aggressive",        
        "A photo of a {} with an angry glare",        
        "A photo of a {} shouting in fury",        
        "A photo of a {} in a heated argument",        
        "A photo of a {} with a fiery temper",        
        "A photo of a {} with a hostile expression",        
        "A photo of a {} with a look of outrage",        
        "A photo of a {} with a seething rage",        
        "A photo of a {} with a temper tantrum",        
        "A photo of a {} with a wrathful expression",        
        "A photo of a {} fuming with anger",        
        "A photo of a {} with a burning rage",        
        "A photo of a {} with a stormy expression",        
        "A photo of a {} with a resentful look",        
        "A photo of a {} with an irate expression", 
        "a sticker of a {}",
        "a sticker of an angry {}",
        "a sticker of a mad {}",
        "a sticker of an annoyed {}",
        "a sticker of a {} in a negative mood",
        "a pixar chracter of an enraged {}",
        "a disney character of a negative {}",
        "a sticker of a furious {}",
        "a sticker of an upset {}",
        "a drawing of an angry {}",
        "a drawing of a miserable {}",
        "a sticker of a miserable {}",
        "a sticker of an enraged {}",
    ],
    placeholder_token = placeholder_token,
)

### concatenate the different datasets with the different emotions
positive_ds = happy_ds.concatenate(love_ds)
negative_ds = sad_ds.concatenate(angry_ds)
train_ds = positive_ds.concatenate(negative_ds)
train_ds = train_ds.batch(1).shuffle(
    train_ds.cardinality(), reshuffle_each_iteration=True
)

### defining concept we want to build our new concept on
tokenized_initializer = stable_diffusion.tokenizer.encode("broccoli")[1]

### get the embedding of our basis concept to clone it to our new placeholder's embedding
new_weights = stable_diffusion.text_encoder.layers[2].token_embedding(tf.constant(tokenized_initializer))

# Get len of .vocab instead of tokenizer
new_vocab_size = len(stable_diffusion.tokenizer.vocab)

# The embedding layer is the 2nd layer in the text encoder
### get the weights of the embedding layer
old_token_weights = stable_diffusion.text_encoder.layers[2].token_embedding.get_weights()
old_position_weights = stable_diffusion.text_encoder.layers[2].position_embedding.get_weights()

### unpack the old weights
old_token_weights = old_token_weights[0]

### old_token_weights has now the shape (vocab_size, embedding_dim)
### expand the dimension to be able to concatenate it with old_token_weights
new_weights = np.expand_dims(new_weights, axis=0)
new_weights = np.concatenate([old_token_weights, new_weights], axis=0)


# Have to set download_weights False so we can initialize the weights ourselves
### create a new text encoder 
new_encoder = TextEncoder(
    MAX_PROMPT_LENGTH,
    vocab_size = new_vocab_size,
    download_weights = False,
)
### we set the weights of the new_encoder to the same as in the old text_encoder except from the embedding layer
for index, layer in enumerate(stable_diffusion.text_encoder.layers):
    # Layer 2 is the embedding layer, so we omit it from our weight-copying
    if index == 2:
        continue
    new_encoder.layers[index].set_weights(layer.get_weights())

### set the weights of the embedding layer according to our new_weights
new_encoder.layers[2].token_embedding.set_weights([new_weights])

### set all weights of the other embeddings to the same values as in the initial text encoder
new_encoder.layers[2].position_embedding.set_weights(old_position_weights)

### set the stable_diffusion text encoder to our new_encoder and compile it
### thus the stable_diffusion.text_encoder has the adjusted weights
stable_diffusion._text_encoder = new_encoder
stable_diffusion._text_encoder.compile(jit_compile=True)

### we only train the encoder as we want to fine-tune the embeddings
stable_diffusion.diffusion_model.trainable = False
stable_diffusion.decoder.trainable = False
stable_diffusion.text_encoder.trainable = True

stable_diffusion.text_encoder.layers[2].trainable = True

def traverse_layers(layer):
    """ Traverses the layers and embedding attributes of a layer

    Args:
    - layer: A text encoder layer

    Yields:
    -  layers and their corresponding embedding attributes
    """
    if hasattr(layer, "layers"):
        for layer in layer.layers:
            yield layer
    if hasattr(layer, "token_embedding"):
        yield layer.token_embedding
    if hasattr(layer, "position_embedding"):
        yield layer.position_embedding

### iterates through the generator and adjusts the trainable attribute of the layers to trainable = True if it is part of the embedding
for layer in traverse_layers(stable_diffusion.text_encoder):
    if isinstance(layer, keras.layers.Embedding) or "clip_embedding" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

### set the layer that only encodes the position of tokens in the prompts to trainable = False
new_encoder.layers[2].position_embedding.trainable = False


### put all the different components of stable diffusion model into a list
all_models = [
    stable_diffusion.text_encoder,
    stable_diffusion.diffusion_model,
    stable_diffusion.decoder,
]

# Remove the top layer from the encoder, which cuts off the variance and only returns the mean
### we make the encoder more efficient while still preserving the most important features
training_image_encoder = keras.Model(
    stable_diffusion.image_encoder.input,
    stable_diffusion.image_encoder.layers[-2].output,
)


def sample_from_encoder_outputs(outputs):
    """Returns a random sample from the embedding distribution given the mean and log variance tensors

    Args:
    - outputs (tensor): A tensor of shape (batch_size, embedding_dim*2), where the first embedding_dim values correspond to the mean of the distribution, 
               and the second embedding_dim values correspond to the log variance of the distribution

    Returns:
    - a tensor of shape (batch_size, embedding_dim), representing a random sample from the embedding distribution
    """
    mean, logvar = tf.split(outputs, 2, axis=-1)
    logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    std = tf.exp(0.5 * logvar)
    sample = tf.random.normal(tf.shape(mean))
    return mean + std * sample


def get_timestep_embedding(timestep, dim=320, max_period=10000):
    """Returns the embedding of a specific timestep in the denoising process.

    Args:
    - timestep (int): The timestep for which the embedding is requested
    - dim (int, optional): The dimensionality of the embedding, default is 320
    - max_period (int, optional): The maximum period, default is 10000

    Returns:
    - embedding (tf.Tensor): A tensor of shape (dim,) containing the embedding of the specified timestep
    """
    ### calculate half the dimensionality of the embedding
    half = dim // 2
    
    ### calculate frequencies using logarithmically decreasing values
    freqs = tf.math.exp(
        -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
    )
    
    ### compute arguments for cosine and sine functions
    args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    
    ### concatenate cosine and sine values to create embedding
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    
    ### return the embedding tensor
    return embedding

#### used for hidden state (output of text encoder)
def get_position_ids():
    """returns position IDs for the transformer model,
        the IDs range from 0 to MAX_PROMPT_LENGTH-1

    Returns:
    - position_ids (tf.Tensor): A tensor of shape (1, MAX_PROMPT_LENGTH) containing the position IDs
    """
    
    ### create a list of integers from 0 to MAX_PROMPT_LENGTH-1
    positions = list(range(MAX_PROMPT_LENGTH))
    
    ### convert the list to a tensor with dtype int32
    position_ids = tf.convert_to_tensor([positions], dtype=tf.int32)
    
    return position_ids

@tf.function
def textual_inversion(model, noise_scheduler, data):
    """Performs textual inversion using a given model and noise scheduler. Uses a gradient tape to calculate the mean squared error between predicted noise and actual noise,
     uses this loss to update the weights of the text encoder with the goal of only training the embedding of the placeholder token

    Arguments:
    - model: A model that takes in noisy latents, timestep embeddings, and the output of the text encoder, and predicts noise
    - noise_scheduler: A noise scheduler that adds noise to latents based on a given timestep
    - data: A tuple containing images and prompt embeddings

    Returns:
    - a dictionary containing the loss value of the model

    """


    images, prompt_embeddings = data

    with tf.GradientTape() as tape:
        
        ### creating embeddings out of the images 
        image_embeddings = training_image_encoder(images)
        ### pass the embeddings to the sampler and save some sammples in latents
        latents = sample_from_encoder_outputs(image_embeddings)
        ### match the latents with those used in the training of Stable Diffusion (just a random number they used in the training)
        latents = latents * 0.18215

        ### random noise in the same shape as latents
        noise = tf.random.normal(tf.shape(latents))
        
        ### get the batch dimension of our input data
        batch_dim = tf.shape(latents)[0]

        ### for each sample in the batch we choose a different random timestep to later determine the specific timestep embedding
        timesteps = tf.random.uniform((batch_dim,), minval=0, maxval=noise_scheduler.train_timesteps, dtype=tf.int64,)


        ### add the noise corresponding to the timestep to the latents by use of the scheduler
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        ### tensor containing all possible indices
        indices = get_position_ids()
        
        ### calculate the output of the encoder
        output_encoder = model.text_encoder([prompt_embeddings, indices])
        
        ### getting the timestep embeddings for each timestep
        timestep_embeddings = tf.map_fn(fn=get_timestep_embedding, elems=timesteps, fn_output_signature=tf.float32,)

        ### calculate the noise predictions with help of the latents, the time step embeddings and the output of the encoder
        noise_pred = model.diffusion_model([noisy_latents, timestep_embeddings, output_encoder])

        ### compute the mean squared error between the noise and the predicted noise and reduce it by taking the mean
        loss = tf.keras.losses.mean_squared_error(noise_pred, noise)
        loss = tf.reduce_mean(loss, axis=2)
        loss = tf.reduce_mean(loss, axis=1)
        loss = tf.reduce_mean(loss)

        ### load the the weights we want to train from the text encoder and calculate the gradients for them
        trainable_weights = model.text_encoder.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)

        ### we only want to update the gradient of the placeholder token, therefore we create the tensor condition which has the value true for the index of the placeholder token (49408) and otherwise false
        condition = gradients[0].indices == 49408

        ### add an extra dimension to later zero out the gradients for other tokens
        condition = tf.expand_dims(condition, axis=-1)

        ### freeze the weights for all tokens by setting the gradients to 0 except for the placeholder token
        gradients[0] = tf.IndexedSlices(values=tf.where(condition, gradients[0].values,0), indices=gradients[0].indices, dense_shape=gradients[0].dense_shape)

        ### apply the gradients to the trainable weights of the encoder and thus only training the placeholder token's embedding
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return {"loss": loss}


### beta is the diffusion rate 
noise_scheduler = NoiseScheduler(
    ### beta_start determines the amount of noise added at the start of the denoising process
    beta_start=0.00085,
    ### beta_end at the end of the denoising process
    beta_end=0.012,
    ### the beta_schedule determines that the diffusion rate increases linearly
    beta_schedule="scaled_linear",
    train_timesteps=1000,
)


#EPOCHS = 50
### learning rate decays depending on the number of epochs to avoid convergence issues in few epochs 
### in the originial tutorial a scheduler is used but we experienced to have better results without a scheduler
"""learning_rate = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4, decay_steps=train_ds.cardinality() * EPOCHS
)"""
### inizialize the optimizer
optimizer = keras.optimizers.Adam(
    weight_decay=0.004, learning_rate=1e-4, epsilon=1e-8, global_clipnorm=10
)

def cosine_sim(e1, e2):
    """Calculate the cosine similarity between two vectors.

    Args:
    - e1 (array): First vector
    - e2 (array): Second vector

    Returns:
    - float: The cosine similarity between the two vectors
    """
  sim = dot(e1, e2)/(norm(e1)*norm(e2))
  return sim

def get_embedding(token):
    """Encodes a given token into a vector embedding using a pre-trained text encoder model.

    Args:
    - token (str): A single word or token to encode into a vector embedding

    Returns:
    - A tensor vector representing the embedding for the given token

    Raises:
    - ValueError: If the input token is empty or None
    """
  tokenized = stable_diffusion.tokenizer.encode(token)[1]
  embedding = stable_diffusion.text_encoder.layers[2].token_embedding(tf.constant(tokenized))

  return embedding


sticker_embedding = []
cosine_similarity = []
broccoli = get_embedding("broccoli")
cosine_similarity.append(cosine_sim(broccoli, get_embedding(placeholder_token)))


def training(epoch=5, model=stable_diffusion, data = train_ds):
    """Trains the Stable Diffusion model for a specified number of epochs by iterating over a given dataset, and computing
    textual inversions for each batch of data. After each epoch, the embedding of the placeholder token is retrieved
    and its cosine similarity with the broccoli emoji embedding is computed and stored in a list.

    Args:
    - epoch (int): The number of epochs to train the model for. Default is 5
    - model (keras.Model): The Stable Diffusion model to train. Default is `stable_diffusion`
    - data (tf.data.Dataset): The dataset to train the model on. Default is `train_ds`

    Returns:
    - None
    """
    for i in range(epoch):
        for batch in data:
            textual_inversion(model=stable_diffusion, noise_scheduler=noise_scheduler, data=batch)
            
        emb = get_embedding(placeholder_token)
        sticker_embedding.append(emb)
        cosine_similarity.append(cosine_sim(broccoli, emb))

def cosine_plot(epoch_num, cosine_similarity):
    """Plot the cosine similarity between the basis and the new concept across epochs.

    Args:
    - epoch_num (list): A list of epoch numbers
    - cosine_similarity (list): A list of cosine similarity scores between the basis and the new concept

    Returns:
    - None. Shows a plot of the cosine similarity scores across epochs.
    """
    plt.plot(epoch_num, cosine_similarity)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity between the basis and the new concept")
    plt.show()

def image_generation(prompt, drive_folder, number):
    """Generates an image using stable diffusion model by passing a string with a placeholder token. 
    The generated image is saved as a JPG file and then copied to a Google Drive folder. A counter is used to ensure unique file names. 

    Args:
    - prompt (str): The prompt used for generating the image
    - drive_folder (str): The path to the Google Drive folder where the image will be saved
    - number (int): How many images are to be generated

    Returns:
    - None
    """
    ### get the number of the last image generated, to ensure each picture gets a different name
    i_file = os.path.join(drive_folder, 'i.txt')
    if os.path.isfile(i_file):
        with open(i_file, 'r') as f:
            i = int(f.read())
    else:
        i = 0
        
    for j in range(number):

        generated = stable_diffusion.text_to_image(
        prompt, batch_size=1,  num_steps=25 )
        broc = generated[0]

        ### convert the array generated from our stable diffusion model into a picture
        broc = Image.fromarray(broc, mode='RGB')

        broc.save(f'image_{i}.jpg')

        ### save the picture to Google Drive
        local_path = f'image_{i}.jpg'
        drive_path = os.path.join(drive_folder, f'image_{i}.jpg')  # Use f-string to include variable in file name
        shutil.copy(local_path, drive_path)

        ### store the value of i in the file, to ensure no picture will have the same name
        i += 1
        with open(i_file, 'w') as f:
            f.write(str(i))










