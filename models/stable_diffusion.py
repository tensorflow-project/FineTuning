# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keras implementation of StableDiffusion.

Credits:

- Original implementation: https://github.com/CompVis/stable-diffusion

The current implementation is a rewrite of the initial TF/Keras port by Divam Gupta.
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.constants import _ALPHAS_CUMPROD
from keras_cv.models.stable_diffusion.constants import _UNCONDITIONAL_TOKENS

from decoder import Decoder
from text_encoder import TextEncoder
from text_encoder import TextEncoderV2
from diffusion_model import DiffusionModel
from diffusion_model import DiffusionModelV2


MAX_PROMPT_LENGTH = 77

from keras_cv.models.stable_diffusion.__internal__.layers.attention_block import (
    AttentionBlock,
)
from keras_cv.models.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)
from keras_cv.models.stable_diffusion.__internal__.layers.resnet_block import (
    ResnetBlock,
)


class ImageEncoder(keras.Sequential):
    """ImageEncoder is the VAE Encoder for StableDiffusion."""

    def __init__(self, img_height=512, img_width=512, download_weights=True):
        super().__init__(
            [
                keras.layers.Input((img_height, img_width, 3)),
                PaddedConv2D(128, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                PaddedConv2D(128, 3, padding=1, strides=2),
                ResnetBlock(256),
                ResnetBlock(256),
                PaddedConv2D(256, 3, padding=1, strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                PaddedConv2D(512, 3, padding=1, strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(8, 3, padding=1),
                PaddedConv2D(8, 1),
                # TODO(lukewood): can this be refactored to be a Rescaling layer?
                # Perhaps some sort of rescale and gather?
                # Either way, we may need a lambda to gather the first 4 dimensions.
                keras.layers.Lambda(lambda x: x[..., :4] * 0.18215),
            ]
        )

        if download_weights:
            image_encoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/vae_encoder.h5",
                file_hash="c60fb220a40d090e0f86a6ab4c312d113e115c87c40ff75d11ffcf380aab7ebb",
            )
            self.load_weights(image_encoder_weights_fpath)

class StableDiffusionBase:
    """Base class for stable diffusion and stable diffusion v2 model
    Args:
    - img_height (int): height of input image, default is 512
    - img_width (int): width of input image, default is 512
    - jit_compile (bool): whether to use just-in-time compilation, default is False  
    """
    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=False,
    ):
        # UNet requires multiples of 2**7 = 128
        img_height = round(img_height / 128) * 128
        img_width = round(img_width / 128) * 128
        self.img_height = img_height
        self.img_width = img_width

        # lazy initialize the component models and the tokenizer
        self._image_encoder = None
        self._text_encoder = None
        self._diffusion_model = None
        self._decoder = None
        self._tokenizer = None
        ### Store just-in-time compilation flag
        self.jit_compile = jit_compile

    def text_to_image(
        self,
        prompt,
        negative_prompt=None,
        batch_size=1,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        seed=None,
    ):
        """Generate an image from a given text prompt using the Stable Diffusion model.

        Args:
        - prompt (str): the text prompt to generate an image from
        - negative_prompt (str): an optional negative text prompt for contrastive learning, default is None
        - batch_size (int): the number of images to generate in each batch, default is 1
        - num_steps (int): the number of diffusion steps to use during generation, default is 50
        - unconditional_guidance_scale (float): a scale factor for unconditional guidance, default is 7.5
        - seed (int): an optional seed for the random number generator, default is None

        Returns:
        - An image generated from the given text prompt
        """
        encoded_text = self.encode_text(prompt)

        return self.generate_image(
            encoded_text,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            num_steps=num_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            seed=seed,
        )

    def encode_text(self, prompt):
        """Encodes a prompt into a latent text encoding.

        The encoding produced by this method should be used as the
        `encoded_text` parameter of `StableDiffusion.generate_image`. Encoding
        text separately from generating an image can be used to arbitrarily
        modify the text encoding priot to image generation, e.g. for walking
        between two prompts.

        Args:
        - prompt (string): a string to encode, must be 77 tokens or shorter
        """
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        ### check prompt length
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        ### pad prompts and convert them to a tensor    
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)
        
        ### calculate latent text encoding of the prompt
        context = self.text_encoder.predict_on_batch(
            [phrase, self._get_pos_ids()]
        )

        return context

    def generate_image(
        self,
        encoded_text,
        negative_prompt=None,
        batch_size=1,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        diffusion_noise=None,
        seed=None,
    ):
        """Generates an image based on encoded text.
        The encoding passed to this method should be derived from
        `StableDiffusion.encode_text`.

        Args:
        - encoded_text (tensor): When the batch axis is omitted, the same encoded
            text will be used to produce every generated image
        - batch_size (int): number of images to generate. Default: 1
        - negative_prompt (string): A string containing information to negatively guide the image generation (e.g. by removing or altering certain aspects
            of the generated image). Default: None
        - num_steps (int): number of diffusion steps (controls image quality). Default: 50
        - unconditional_guidance_scale (float): float controling how closely the image should adhere to the prompt. 
            Larger values result in more closely adhering to the prompt, but will make the image noisier. Default: 7.5
        - diffusion_noise (tensor, optional): Tensor. Optional custom noise to seed the diffusion
                process. When the batch axis is omitted, the same noise will be
                used to seed diffusion for every generated image
        - seed (int): integer which is used to seed the random generation of diffusion noise, only to be specified if `diffusion_noise` is None
        """
        if diffusion_noise is not None and seed is not None:
            raise ValueError(
                "`diffusion_noise` and `seed` should not both be passed to "
                "`generate_image`. `seed` is only used to generate diffusion "
                "noise when it's not already user-specified."
            )

        context = self._expand_tensor(encoded_text, batch_size)
        ### if there is no negative prompt get another unconditional context
        if negative_prompt is None:
            unconditional_context = tf.repeat(
                self._get_unconditional_context(), batch_size, axis=0
            )
        ### else, set the unconditional context to the encoded negative prompt
        else:
            unconditional_context = self.encode_text(negative_prompt)
            unconditional_context = self._expand_tensor(
                unconditional_context, batch_size
            )
        ### If diffusion noise is provided, use it as the initial noise
        if diffusion_noise is not None:
            diffusion_noise = tf.squeeze(diffusion_noise)
            if diffusion_noise.shape.rank == 3:
                diffusion_noise = tf.repeat(
                    tf.expand_dims(diffusion_noise, axis=0), batch_size, axis=0
                )
            latent = diffusion_noise
        else:
            ### Otherwise, generate initial diffusion noise with a seed
            latent = self._get_initial_diffusion_noise(batch_size, seed)
        
        ### Iteratively perform reverse diffusion to generate image
        
        ### setting up the range of timesteps for the reverse diffusion process
        timesteps = tf.range(1, 1000, 1000 // num_steps)
        
        ### get the alpha values, they repesent the amount of noise in a specific timestep
        alphas, alphas_prev = self._get_initial_alphas(timesteps)        
        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0  
        
        ### for each timestep, but beginning at the end of the list, as the process is performed in a reverse manner
        for index, timestep in list(enumerate(timesteps))[::-1]:
            ### store the latent for later noise prediction
            latent_prev = latent  # Set aside the previous latent vector
            
            ### get the embedding for the current timestep
            t_emb = self._get_timestep_embedding(timestep, batch_size)
            
            ### calculating the unconditional latent, is used such that the image generation does not overly rely on the context
            unconditional_latent = self.diffusion_model.predict_on_batch(
                [latent, t_emb, unconditional_context]
            )
            
            ### calculating the latent conditioned on the context for the current timestep
            latent = self.diffusion_model.predict_on_batch(
                [latent, t_emb, context]
            )
            
            ### combine the conditional and the unconditional latent by means of a scale
            latent = unconditional_latent + unconditional_guidance_scale * (
                latent - unconditional_latent
            )
            ### getting the alpha values for the current timestep
            a_t, a_prev = alphas[index], alphas_prev[index]
            
            ### predicting the latent for the initial image 
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(
                a_t
            )
            ### update the latent by help of the initial image and the alpha values
            latent = (
                latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
            )
            iteration += 1
            progbar.update(iteration)

        # Decoding stage
        ### pass the initial image latent to the decoder and decode it
        decoded = self.decoder.predict_on_batch(latent)
        ### scaling the values up to between 0 and 255
        decoded = ((decoded + 1) / 2) * 255
        ### clipping the values below 0 to 0 and those higher than 255 to 255
        ### convert the type to unsigned 8-bit integers (common type for images)
        return np.clip(decoded, 0, 255).astype("uint8")

    def inpaint(
        self,
        prompt,
        image,
        mask,
        negative_prompt=None,
        num_resamples=1,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7.5,
        diffusion_noise=None,
        seed=None,
        verbose=True,
    ):
        """Inpaints a masked section of the provided image based on the provided prompt.
        Is mostly used for training to provide more stability to the model, is needed to fill in missing or corrupted parts of an image
        helps to improve visual quality

        Args:
        - prompt (string): A string representing the prompt for generation
        - image (tensor): Tensor of shape with RGB values in [0, 255]. When the batch is omitted, the same image will be used as the starting image
        - mask (tensor): Tensor with binary values 0 or 1. When the batch is omitted, the same mask will be used on all images
        - negative_prompt (string): A string containing information to negatively guide the image generation (e.g. by removing or altering certain aspects
            of the generated image). Default: None
        - num_resamples (int): number of times to resample the generated mask region. Increasing the number of resamples improves the semantic fit of the
            generated mask region w.r.t the rest of the image. Default: 1
        - batch_size (int): number of images to generate. Default: 1
        - num_steps (int): number of diffusion steps (controls image quality). Default: 25
        - unconditional_guidance_scale (float): controlling how closely the image should adhere to the prompt. Larger values result in more
             closely adhering to the prompt, but will make the image noisier. Default: 7.5
        - diffusion_noise (tensor, optional): Tensor. Optional custom noise to
              seed the diffusion process. When the batch axis is omitted, the same noise will be used to seed diffusion for every generated image
        - seed (int, optional): is used to seed the random generation of diffusion noise, only to be specified if `diffusion_noise` is None
        - verbose (bool): whether to print progress bar. Default: True
        """
        if diffusion_noise is not None and seed is not None:
            raise ValueError(
                "Please pass either diffusion_noise or seed to inpaint(), seed "
                "is only used to generate diffusion noise when it is not provided. "
                "Received both diffusion_noise and seed."
            )

        encoded_text = self.encode_text(prompt)
        encoded_text = tf.squeeze(encoded_text)
        if encoded_text.shape.rank == 2:
            ### the encoded_text needs to have the same shape as the image
            encoded_text = tf.repeat(
                tf.expand_dims(encoded_text, axis=0), batch_size, axis=0
            )
        ### making the image compatible with the network
        image = tf.squeeze(image)
        
        ### normalize the image
        image = tf.cast(image, dtype=tf.float32) / 255.0 * 2.0 - 1.0
        image = tf.expand_dims(image, axis=0)
        
        ### get a latent representation of the image
        known_x0 = self.image_encoder(image)
        
        if image.shape.rank == 3:
            ### match the shape of the input tensor
            known_x0 = tf.repeat(known_x0, batch_size, axis=0)

        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.cast(
            tf.nn.max_pool2d(mask, ksize=8, strides=8, padding="SAME"),
            dtype=tf.float32,
        )
        
        ### preparing the mask tensor that indicates which pixels of the input image should be inpainted
        mask = tf.squeeze(mask)
        if mask.shape.rank == 2:
            mask = tf.repeat(tf.expand_dims(mask, axis=0), batch_size, axis=0)
        mask = tf.expand_dims(mask, axis=-1)

        context = encoded_text
        ### if there is no negative prompt, get a random unconditional context
        if negative_prompt is None:
            unconditional_context = tf.repeat(
                self._get_unconditional_context(), batch_size, axis=0
            )
        ### else encode the negative prompt to get an unconditional context
        else:
            unconditional_context = self.encode_text(negative_prompt)
            unconditional_context = self._expand_tensor(
                unconditional_context, batch_size
            )
            
        ### if diffusion noise is provided preprocess it to match the shape of a latent
        if diffusion_noise is not None:
            diffusion_noise = tf.squeeze(diffusion_noise)
            if diffusion_noise.shape.rank == 3:
                diffusion_noise = tf.repeat(
                    tf.expand_dims(diffusion_noise, axis=0), batch_size, axis=0
                )
            latent = diffusion_noise
            
        ### else, generate random noise
        else:
            latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Iterative reverse diffusion stage
        
        ### setting up a range of timesteps
        timesteps = tf.range(1, 1000, 1000 // num_steps)
        
        ### get the initial alpha values
        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        if verbose:
            progbar = keras.utils.Progbar(len(timesteps))
            iteration = 0
        ### iterate through the timesteps in a reverse manner
        for index, timestep in list(enumerate(timesteps))[::-1]:
            a_t, a_prev = alphas[index], alphas_prev[index]
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._get_timestep_embedding(timestep, batch_size)
            
            ### for the desired number of resamples
            for resample_index in range(num_resamples):
                
                ### calculate the unconditional latent
                unconditional_latent = self.diffusion_model.predict_on_batch(
                    [latent, t_emb, unconditional_context]
                )
                ### calculate the conditional latent
                latent = self.diffusion_model.predict_on_batch(
                    [latent, t_emb, context]
                )
                ### combine the unconditional and the conditional latent by means of a guidance scale
                latent = unconditional_latent + unconditional_guidance_scale * (
                    latent - unconditional_latent
                )
                ### calculate the predicted initial image
                pred_x0 = (
                    latent_prev - math.sqrt(1 - a_t) * latent
                ) / math.sqrt(a_t)
                
                ### update the current latent by help of the predicted image
                latent = (
                    latent * math.sqrt(1.0 - a_prev)
                    + math.sqrt(a_prev) * pred_x0
                )

                # Use known image (x0) to compute latent
                if timestep > 1:
                    noise = tf.random.normal(tf.shape(known_x0), seed=seed)
                else:
                    noise = 0.0
                known_latent = (
                    math.sqrt(a_prev) * known_x0 + math.sqrt(1 - a_prev) * noise
                )
                # Use known latent in unmasked regions
                latent = mask * known_latent + (1 - mask) * latent
                # Resample latent
                if resample_index < num_resamples - 1 and timestep > 1:
                    beta_prev = 1 - (a_t / a_prev)
                    latent_prev = tf.random.normal(
                        tf.shape(latent),
                        mean=latent * math.sqrt(1 - beta_prev),
                        stddev=math.sqrt(beta_prev),
                        seed=seed,
                    )

            if verbose:
                iteration += 1
                progbar.update(iteration)

        # Decoding stage
        ### pass the latent to the decoder
        decoded = self.decoder.predict_on_batch(latent)
        ### upscale the values of the image
        decoded = ((decoded + 1) / 2) * 255
        ### clip values below 0 and above 255 and convert the type into unsigned 8-bit integers
        return np.clip(decoded, 0, 255).astype("uint8")

    def _get_unconditional_context(self):
        """Returns the unconditional context for the text encoder
        The unconditional context is a tensor representing the encoding of a fixed set of unconditional tokens

        Returns:
        - unconditional_context (tensor): Tensor representing the unconditional context
        """
        unconditional_tokens = tf.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS], dtype=tf.int32
        )
        unconditional_context = self.text_encoder.predict_on_batch(
            [unconditional_tokens, self._get_pos_ids()]
        )

        return unconditional_context

    def _expand_tensor(self, text_embedding, batch_size):
        """Expands a tensor by repeating it to fit the shape of the given batch size. 
        The function first removes any dimensions of size 1 from the tensor using `tf.squeeze()`.
        Then, if the tensor shape rank is 2, it repeats the tensor along the first dimension
        to match the desired batch size using `tf.repeat()`.
        Then, the expanded tensor is returned.

        Args:
        - text_embedding (tf.Tensor): The tensor to be expanded
        - batch_size (int): The desired batch size

        Returns:
        - tf.Tensor: The expanded tensor

        Raises:
        - ValueError: If the tensor rank is not 2.
        """
        text_embedding = tf.squeeze(text_embedding)
        if text_embedding.shape.rank == 2:
            text_embedding = tf.repeat(
                tf.expand_dims(text_embedding, axis=0), batch_size, axis=0
            )
        return text_embedding

    @property
    def image_encoder(self):
        """Returns the VAE Encoder with pretrained weights.
        The method first checks if the `_image_encoder` attribute is None. If it is, it creates a new
        instance of the `ImageEncoder` class with the image height and width specified by the
        `img_height` and `img_width` attributes. If the `jit_compile` attribute is True, the encoder
        is compiled with JIT compilation enabled. The `_image_encoder` attribute is then set to the
        newly created encoder instance. Finally, the `_image_encoder` attribute is returned.
        If the `_image_encoder` attribute is not None, the existing encoder instance is returned.

        Returns:
        - ImageEncoder: The VAE Encoder with pretrained weights
        """
        if self._image_encoder is None:
            self._image_encoder = ImageEncoder(self.img_height, self.img_width)
            if self.jit_compile:
                self._image_encoder.compile(jit_compile=True)
        return self._image_encoder

    @property
    def text_encoder(self):
        """Property that returns the text encoder used by the object.    
        """
        pass

    @property
    def diffusion_model(self):
        """Property that returns the fiffusion model used by the object.    
        """
        pass

    @property
    def decoder(self):
        """Returns the diffusion image decoder model with pretrained weights.
        The decoder model is used to reconstruct images from the latent space generated
        by the diffusion image model. By default, the method returns a pre-trained decoder
        model that is suitable for a wide range of tasks. Can be overridden
        to modify the decoder for the Stable Diffusion or to use a different implementation.

        Returns:
        - An instance of the diffusion image decoder model with pretrained weights
        """
        if self._decoder is None:
            self._decoder = Decoder(self.img_height, self.img_width)
            if self.jit_compile:
                self._decoder.compile(jit_compile=True)
        return self._decoder

    @property
    def tokenizer(self):
        """Returns the tokenizer used for text inputs.

        The tokenizer is responsible for breaking down text inputs into smaller units,
        such as words or subwords, that can be processed by machine learning models.
        By default, the method returns a simple tokenizer that is suitable for a wide range
        of tasks. However, the method can be overridden to modify the tokenizer for specific
        tasks such as the Stable Diffusion or to use a different implementation.

        Returns:
        - An instance of the tokenizer used for text inputs
        """
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
        return self._tokenizer

    def _get_timestep_embedding(
        self, timestep, batch_size, dim=320, max_period=10000
    ):
        """Get the timestep embedding for the current time step.

        Args:
        - timestep (int): the current time step
        - batch_size (int): the batch size
        - dim (int): the dimension of the embedding, default: 320
        - max_period (int): the maximum period, default: 10000

        Returns:
        - tensor: the timestep embedding for the current time step
        """
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    def _get_initial_alphas(self, timesteps):
        """Gets the alphas (cumulative product of the square root of the inverse
        timescales) for the given timesteps, as well as the previous alphas

        Args:
        - timesteps (tensor): tensor containing the timesteps for which to get
            the alphas

        Returns:
        - alphas (list): a list of the alphas for each timestep in `timesteps`
        - alphas_prev (list): a list of the previous alphas, with the first
            element set to 1.0
        """
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_initial_diffusion_noise(self, batch_size, seed):
        """Returns an initial noise tensor for the diffusion process, with a specific shape and optional random seed

        Args:
        - batch_size (int): The batch size of the noise tensor
        - seed (int, optional): The random seed to use for generating the noise tensor

        Returns:
        - A tensor of shape (batch_size, img_height//8, img_width//8, 4) representing the initial noise tensor for the diffusion process
        """
        if seed is not None:
            return tf.random.stateless_normal(
                (batch_size, self.img_height // 8, self.img_width // 8, 4),
                seed=[seed, seed],
            )
        else:
            return tf.random.normal(
                (batch_size, self.img_height // 8, self.img_width // 8, 4)
            )

    @staticmethod
    def _get_pos_ids():
        """Returns a tensor representing the position IDs of tokens in a prompt sequence.
        The position IDs are used to indicate the position of each token in a sequence.
        By default, the method returns a tensor representing the position IDs for a
        prompt sequence of maximum length.

        Returns:
        - A tensor representing the position IDs of tokens in a prompt sequence
        """
        return tf.convert_to_tensor(
            [list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32
        )


class StableDiffusion(StableDiffusionBase):
    """Keras implementation of Stable Diffusion.

    Stable Diffusion is an image generation model that can be used,
    among other things, to generate pictures according to a short text description
    (called a "prompt").

    Arguments:
    - img_height (int): Height of the images to generate, in pixel. Default: 512
    - img_width (int): Width of the images to generate, in pixel. Note that only
            multiples of 128 are supported. Default: 512
    - jit_compile (bool): Whether to compile the underlying models to XLA. This can lead to a speedup on some systems. Default: False
    """
    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=False,
    ):
        super().__init__(img_height, img_width, jit_compile)
        print(
            "By using this model checkpoint, you acknowledge that its usage is "
            "subject to the terms of the CreativeML Open RAIL-M license at "
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE"
        )

    @property
    def text_encoder(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.

        Returns:
        - An instance of the text encoder with pretrained weights
        """
        if self._text_encoder is None:
            self._text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def diffusion_model(self):
        """diffusion_model returns the diffusion model with pretrained weights.
        Can be overriden for tasks where the diffusion model needs to be modified.

        Returns:
        - An instance of the diffusion model with pretrained weights
        """
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModel(
                self.img_height, self.img_width, MAX_PROMPT_LENGTH
            )
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model


class StableDiffusionV2(StableDiffusionBase):
    """Keras implementation of Stable Diffusion.

    Stable Diffusion is an image generation model that can be used,
    among other things, to generate pictures according to a short text description
    (called a "prompt").

    Arguments:
    - img_height (int): Height of the images to generate, in pixel. Default: 512
    - img_width (int): Width of the images to generate, in pixel. Note that only
            multiples of 128 are supported. Default: 512
    - jit_compile (bool): Whether to compile the underlying models to XLA. This can lead to a speedup on some systems. Default: False
    """
    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=False,
    ):
        super().__init__(img_height, img_width, jit_compile)
        print(
            "By using this model checkpoint, you acknowledge that its usage is "
            "subject to the terms of the CreativeML Open RAIL++-M license at "
            "https://github.com/Stability-AI/stablediffusion/main/LICENSE-MODEL"
        )

    @property
    def text_encoder(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        
        Returns:
        - An instance of the text encoder with pretrained weights
        """
        if self._text_encoder is None:
            self._text_encoder = TextEncoderV2(MAX_PROMPT_LENGTH)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def diffusion_model(self):
        """diffusion_model returns the diffusion model with pretrained weights.
        Can be overriden for tasks where the diffusion model needs to be modified.
        
        Returns:
        - An instance of the diffusion model with pretrained weights
        """
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModelV2(
                self.img_height, self.img_width, MAX_PROMPT_LENGTH
            )
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model
