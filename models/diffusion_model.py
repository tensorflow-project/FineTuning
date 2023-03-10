# %% [markdown]
# <a href="https://colab.research.google.com/github/tensorflow-project/stable-diffusion/blob/main/keras_cv_diffusion_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
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

import tensorflow as tf
from tensorflow import keras

from keras_cv.models.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)


class DiffusionModel(keras.Model):
    """A U-Net model used for stable diffusion, which generates images by downsampling and then upsampling random noise. The model takes in three inputs: 
    1. `context`, a tensor of shape `(max_text_length, 768)`, which represents contextual information.
    2. `t_embed_input`, a tensor of shape `(320,)`, which contains time embedding information.
    3. `latent`, a tensor of shape `(img_height // 8, img_width // 8, 4)`, which represents random noise.
    
    The model consists of a downsampling flow, a middle flow, an upsampling flow, and an exit flow. 
    The downsampling flow contains several ResBlocks, each of which uses a SpatialTransformer to transform the feature maps with respect to the contextual information. 
    The outputs of each ResBlock are saved in a list to improve the upsampling process later on. 
    After three downsampling steps, the middle flow contains a single ResBlock, which is also transformed using a SpatialTransformer. 
    The upsampling flow consists of three steps that each use a Concatenate layer to concatenate the output of the previous ResBlock 
    with the saved output from the corresponding downsampling step. 
    The concatenated tensor is then processed using another ResBlock and a SpatialTransformer, before being upsampled. 
    Finally, the exit flow applies GroupNormalization, Swish activation, and a PaddedConv2D layer to produce the final output.
    
    Args:
    - img_height (int): height of the input image.
    - img_width (int): width of the input image.
    - max_text_length (int): maximum length of the contextual text information.
    - name (string): name of the model.
    - download_weights (bool): whether to download the pre-trained weights.
    """
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name=None,
        download_weights=True,
    ):
        context = keras.layers.Input((max_text_length, 768))
        t_embed_input = keras.layers.Input((320,))
        latent = keras.layers.Input((img_height // 8, img_width // 8, 4))

        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Downsampling flow
        ### save the different downsampling steps to improve later upsampling
        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            ### using the outputs of the downsampling steps to improve the upsampling
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])

        # Exit flow

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__([latent, t_embed_input, context], output, name=name)

        if download_weights:
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",
                file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",
            )
            self.load_weights(diffusion_model_weights_fpath)


class DiffusionModelV2(keras.Model):
    """ALternative to DiffusionModel"""
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name=None,
        download_weights=True,
    ):
        context = keras.layers.Input((max_text_length, 1024))
        t_embed_input = keras.layers.Input((320,))
        latent = keras.layers.Input((img_height // 8, img_width // 8, 4))

        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Downsampling flow

        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(5, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(10, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(10, 64, fully_connected=True)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(5, 64, fully_connected=True)([x, context])

        # Exit flow

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__([latent, t_embed_input, context], output, name=name)

        if download_weights:
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/diffusion_model_v2_1.h5",
                file_hash="c31730e91111f98fe0e2dbde4475d381b5287ebb9672b1821796146a25c5132d",
            )
            self.load_weights(diffusion_model_weights_fpath)


class ResBlock(keras.layers.Layer):
    """A residual block layer that consists of an entry flow, an embedding flow,
    and an exit flow. The entry flow and exit flow are convolutional blocks,
    while the embedding flow is a dense block that takes in an additional
    input tensor. The output of the entry flow is added to the output of the
    embedding flow, and the result is passed through the exit flow. Finally,
    the output of the residual projection (if necessary) is added to the result.

    Args:
    - output_dim (int): The number of filters for the convolutional layers determining the output dimension
    """
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.entry_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]
        self.embedding_flow = [
            keras.layers.Activation("swish"),
            keras.layers.Dense(output_dim),
        ]
        self.exit_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]

    def build(self, input_shape):
        """Builds the layer by setting up the residual projection layer if needed.

        Args:
        - input_shape (tuple): A tuple of two shapes, the input tensor shape and the embedding tensor shape
        """
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        """Performs a forward pass on the layer.

        Args:
        - inputs (tuple): A tuple of two tensors, the input tensor and the embedding tensor

        Returns:
        - tensor representing the output of the residual block layer
        """
        inputs, embeddings = inputs
        x = inputs
        for layer in self.entry_flow:
            x = layer(x)
        for layer in self.embedding_flow:
            embeddings = layer(embeddings)
        x = x + embeddings[:, None, None]
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)


class SpatialTransformer(keras.layers.Layer):
    """
    Spatial transformer layer that transforms the input image with respect to the context embedding.
    This layer applies a transformer block to the input image, conditioned on a context embedding,
    to produce a transformed image that incorporates the contextual information.
    
    Args:
        num_heads (int): Number of attention heads.
        head_size (int): Size of each attention head.
        fully_connected (bool): Whether to use fully connected layers for projection, 
                                defaults to False.
    
    Returns:
        The transformed image tensor.
    """
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        channels = num_heads * head_size
        if fully_connected:
            self.proj1 = keras.layers.Dense(num_heads * head_size)
        else:
            self.proj1 = PaddedConv2D(num_heads * head_size, 1)
        self.transformer_block = BasicTransformerBlock(
            channels, num_heads, head_size
        )
        if fully_connected:
            self.proj2 = keras.layers.Dense(channels)
        else:
            self.proj2 = PaddedConv2D(channels, 1)

    def call(self, inputs):
        """
        Applies the spatial transformer to the input image tensor.
        
        Args:
            inputs (tuple): A tuple of two input tensors: the image tensor and the context embedding tensor.
            
        Returns:
            The transformed image tensor.
        """
        inputs, context = inputs
        _, h, w, c = inputs.shape
        x = self.norm(inputs)
        x = self.proj1(x)
        x = tf.reshape(x, (-1, h * w, c))
        x = self.transformer_block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj2(x) + inputs


class BasicTransformerBlock(keras.layers.Layer):
    """
    A basic Transformer block consisting of two layers of multi-head self-attention and one feedforward layer.

    Args:
        dim (int): Dimension of the input and output tensor.
        num_heads (int): Number of attention heads.
        head_size (int): Size of each attention head.
    """
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(num_heads, head_size)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(num_heads, head_size)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        """Apply a basic transformer block to the inputs.

        Args:
            inputs: A tuple of two tensors, (query, context).
                `query` has shape (batch_size, seq_len_query, dim),
                `context` has shape (batch_size, seq_len_context, dim).

        Returns:
            A tensor with the same shape as `inputs`, representing the
            result of applying the basic transformer block to the inputs.
        """
        inputs, context = inputs
        x = self.attn1([self.norm1(inputs), None]) + inputs
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class CrossAttention(keras.layers.Layer):
    """
    Computes cross-attention between two sequences of input vectors.

    Args:
        num_heads (int): The number of attention heads to use.
        head_size (int): The size of each attention head.
    
    Returns:
        output (tf.Tensor): The output of the cross-attention operation, with shape
                            `(batch_size, sequence_length, num_heads * head_size)`.
    """
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.to_q = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_k = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_v = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = keras.layers.Dense(num_heads * head_size)

    def call(self, inputs):
        """
        Computes the cross-attention between two sequences of input vectors.

        Args:
            inputs (tuple): A tuple containing two input tensors:
                            `inputs` - The input tensor to the query layer with shape `(batch_size, sequence_length, input_size)`.
                            `context` - The input tensor to the key and value layers with shape `(batch_size, sequence_length, input_size)`.

        Returns:
            output (tf.Tensor): The output of the cross-attention operation, with shape
                                `(batch_size, sequence_length, num_heads * head_size)`.
        """
        inputs, context = inputs
        context = inputs if context is None else context
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        q = tf.reshape(q, (-1, inputs.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(
            k, (-1, context.shape[1], self.num_heads, self.head_size)
        )
        v = tf.reshape(
            v, (-1, context.shape[1], self.num_heads, self.head_size)
        )

        q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(
            score
        )  # (bs, num_heads, time, time)
        attn = td_dot(weights, v)
        attn = tf.transpose(
            attn, (0, 2, 1, 3)
        )  # (bs, time, num_heads, head_size)
        out = tf.reshape(
            attn, (-1, inputs.shape[1], self.num_heads * self.head_size)
        )
        return self.out_proj(out)


class Upsample(keras.layers.Layer):
    """
    Upsamples the spatial dimensions of a tensor by a factor of 2 and applies a 3x3 convolution.

    Args:
        channels (int): The number of output channels in the convolutional layer.

    Input shape:
        A 4D tensor with shape `(batch_size, height, width, channels)`.

    Output shape:
        A 4D tensor with shape `(batch_size, 2*height, 2*width, channels)`.
    """
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.ups = keras.layers.UpSampling2D(2)
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, inputs):
        """
        Applies upsampling and convolution to the input tensor.

        Args:
            inputs: A 4D tensor with shape `(batch_size, height, width, channels)`.

        Returns:
            A 4D tensor with shape `(batch_size, 2*height, 2*width, channels)`.
        """
        return self.conv(self.ups(inputs))


class GEGLU(keras.layers.Layer):
    """
    Gated Linear Unit with Gaussian Error Linear Units (GEGLU) activation function.
    This layer applies a dense layer followed by the GEGLU activation function to its inputs.

    Args:
        output_dim (int): dimensionality of the output space.

    Input shape:
        2D tensor of shape `(batch_size, input_dim)`.

    Output shape:
        2D tensor of shape `(batch_size, output_dim)`.
    """
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = keras.layers.Dense(output_dim * 2)

    def call(self, inputs):
        """
        Apply GEGLU activation function to inputs.

        Args:
            inputs (tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            tensor: A tensor of shape `(batch_size, output_dim)` after applying the GEGLU activation function.
        """
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        return x * 0.5 * gate * (1 + tanh_res)


def td_dot(a, b):
    """Computes the dot product between two tensors, where the last two dimensions of both tensors are contracted.

    Args:
        a: A tensor with shape (batch_size, num_elements_a, dim_a1, dim_a2).
        b: A tensor with shape (batch_size, num_elements_b, dim_b1, dim_b2).

    Returns:
        A tensor with shape (batch_size, num_elements_a, dim_a1, dim_b2).
    """
    aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.backend.batch_dot(aa, bb)
    return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))


