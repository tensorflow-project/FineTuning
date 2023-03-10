# %% [markdown]
# <a href="https://colab.research.google.com/github/tensorflow-project/stable-diffusion/blob/main/keras_cv_text_encoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
from tensorflow.experimental import numpy as tfnp


class TextEncoder(keras.Model):
    """Transformer-based text encoder for the OpenAI CLIP model.

    This class constructs a text encoder with 12 Transformer layers with the 
    embedding dimension 768 and 12 attention heads with quick_gelu activation.

    Args:
        max_length (int): Maximum length of the input text sequence.
        vocab_size (int, optional): Size of the vocabulary. Defaults to 49408.
        name (str, optional): Name of the model. Defaults to None
        download_weights (bool, optional): Whether to download pre-trained weights 
            Defaults to True

    Attributes:
        tokens (keras.layers.Input): Input layer for the text tokens
        positions (keras.layers.Input): Input layer for the token positions

    Raises:
        ValueError: If max_length is less than or equal to 0

    """
    def __init__(self, max_length, vocab_size=49408, name=None, download_weights=True):  
     
        tokens = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="tokens"
        )
        positions = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="positions"
        )
        ### construct an embedding
        x = CLIPEmbedding(vocab_size, 768, max_length)([tokens, positions])

        ### build 12 layers with the embedding dimension 768 and 12 attention heads with quick_gelu activation
        for _ in range(12):
            x = CLIPEncoderLayer(768, 12, activation=quick_gelu)(x)

        ### normalize the embedding
        embedded = keras.layers.LayerNormalization(epsilon=1e-5)(x)
        super().__init__([tokens, positions], embedded, name=name)
        
        ### get the weights from hugging face
        if download_weights:
            text_encoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",
                file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",
            )
            self.load_weights(text_encoder_weights_fpath)

### same as above but different layers
class TextEncoderV2(keras.Model):
    def __init__(
        self, max_length, vocab_size=49408, name=None, download_weights=True
    ):
        tokens = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="tokens"
        )
        positions = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="positions"
        )
        x = CLIPEmbedding(vocab_size, 1024, max_length)([tokens, positions])
        for _ in range(23):
            x = CLIPEncoderLayer(1024, 16, activation=tf.nn.gelu)(x)
        ### normalize the embedding
        embedded = keras.layers.LayerNormalization(epsilon=1e-5)(x)
        super().__init__([tokens, positions], embedded, name=name)

        if download_weights:
            text_encoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/text_encoder_v2_1.h5",
                file_hash="985002e68704e1c5c3549de332218e99c5b9b745db7171d5f31fcd9a6089f25b",
            )
            self.load_weights(text_encoder_weights_fpath)

### defines the activation function
def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)


class CLIPEmbedding(keras.layers.Layer):
     """Layer that creates embeddings for tokens and positions and combines them.

    Args:
        input_dim (int, optional): Size of the vocabulary. Defaults to 49408.
        output_dim (int, optional): Dimension of the output embeddings. Defaults to 768.
        max_length (int, optional): Maximum length of input sequences. Defaults to 77.

    Returns:
        tf.Tensor: Combined embeddings of tokens and positions.
    """
    def __init__(
        self, input_dim=49408, output_dim=768, max_length=77, **kwargs
    ):
        super().__init__(**kwargs)
        self.token_embedding = keras.layers.Embedding(input_dim, output_dim)
        self.position_embedding = keras.layers.Embedding(max_length, output_dim)

    def call(self, inputs):
        tokens, positions = inputs
        ### create the embeddings
        tokens = self.token_embedding(tokens)
        positions = self.position_embedding(positions)
        ### combine the two embeddings
        return tokens + positions


class CLIPEncoderLayer(keras.layers.Layer):
     """
    A single encoder layer in the CLIP model architecture that processes the input data with a self-attention
    mechanism followed by a feedforward neural network, while preserving the input with residual connections.

    Args:
        embed_dim (int): The dimension of the embedding space.
        num_heads (int): The number of heads in the self-attention mechanism.
        activation (function, optional): Activation function to use. If not specified, no activation is applied.

    Returns:
        Tensor: The output tensor of the encoder layer.
    """
    def __init__(self, embed_dim, num_heads, activation=None, **kwargs):
        super().__init__(**kwargs)
        ### layer norm to avoid overfitting
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.clip_attn = CLIPAttention(embed_dim, num_heads, causal=True)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.fc1 = keras.layers.Dense(embed_dim * 4)
        self.fc2 = keras.layers.Dense(embed_dim)
        self.activation = activation

    def call(self, inputs):
        ### input travels through the layers with residual connections to keep the input alive
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.clip_attn(x)
        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x + residual


class CLIPAttention(keras.layers.Layer):
    """
    Layer that performs attention calculation to get the attention weights and embeddings.

    Args:
        embed_dim (int): The size of the embedding dimension. Default is 768.
        num_heads (int): The number of attention heads. Default is 12.
        causal (bool): Whether to attend only to tokens before the current one or to all tokens. Default is True.

    Attributes:
        q_proj (Dense): Layer that projects the input to the query states.
        k_proj (Dense): Layer that projects the input to the key states.
        v_proj (Dense): Layer that projects the input to the value states.
        out_proj (Dense): Layer that projects the attention output to the final output.
        head_dim (int): The size of the embedding dimension divided by the number of attention heads.
        scale (float): Scaling factor for the attention weights.

    Methods:
        reshape_states(x, sequence_length, batch_size):
            Reshapes the different states for later matrix multiplication.
        call(inputs, attention_mask=None):
            Performs the attention calculation to get the attention weights and embeddings.

    """
    def __init__(self, embed_dim=768, num_heads=12, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        ### only attending to the tokens before and not after, why?
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        ### initialize embedding layers (projection) for the query, key, values and the output 
        self.q_proj = keras.layers.Dense(self.embed_dim)
        self.k_proj = keras.layers.Dense(self.embed_dim)
        self.v_proj = keras.layers.Dense(self.embed_dim)
        self.out_proj = keras.layers.Dense(self.embed_dim)

    def reshape_states(self, x, sequence_length, batch_size):
        """
        Reshapes the different states for later matrix multiplication.

        Args:
            x (Tensor): Input tensor to be reshaped.
            sequence_length (int): Length of the input sequence.
            batch_size (int): Batch size of the input sequence.

        Returns:
            Tensor: Reshaped tensor.

        """
        x = tf.reshape(
            x, (batch_size, sequence_length, self.num_heads, self.head_dim)
        )
        return tf.transpose(
            x, (0, 2, 1, 3)
        )  # bs, heads, sequence_length, head_dim

    def call(self, inputs, attention_mask=None):
         """
        Performs the attention calculation to get the attention weights and embeddings.

        Args:
            inputs (Tensor): Input tensor to be processed.
            attention_mask (Tensor): Attention mask tensor for masking the tokens. Default is None.

        Returns:
            Tensor: The embeddings of the output.

        """
        if attention_mask is None and self.causal:
            length = tf.shape(inputs)[1]
            attention_mask = tfnp.triu(
                tf.ones((1, 1, length, length), dtype=self.compute_dtype)
                * -tfnp.inf,
                k=1,
            )

        _, tgt_len, embed_dim = inputs.shape
        ### define the query, key and value states and reshape them
        query_states = self.q_proj(inputs) * self.scale
        key_states = self.reshape_states(self.k_proj(inputs), tgt_len, -1)
        value_states = self.reshape_states(self.v_proj(inputs), tgt_len, -1)

        ### projection shape depends on the target length, is used for the later reshaping
        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self.reshape_states(query_states, tgt_len, -1)
        query_states = tf.reshape(query_states, proj_shape)
        key_states = tf.reshape(key_states, proj_shape)

        ### in this case the source length is equal to the target length
        src_len = tgt_len
        value_states = tf.reshape(value_states, proj_shape)

        ###  calculate the attention weights using the query states
        attn_weights = query_states @ tf.transpose(key_states, (0, 2, 1))

        attn_weights = tf.reshape(
            attn_weights, (-1, self.num_heads, tgt_len, src_len)
        )
        ### include whether the token is masked
        attn_weights = attn_weights + attention_mask
        attn_weights = tf.reshape(attn_weights, (-1, tgt_len, src_len))

        ### apply softmax on the attention weights in order to get a probability distribution
        attn_weights = tf.nn.softmax(attn_weights)
        ### multiply the attention weights with the value states to determine how much attention has to be paid on which token
        attn_output = attn_weights @ value_states

        attn_output = tf.reshape(
            attn_output, (-1, self.num_heads, tgt_len, self.head_dim)
        )
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (-1, tgt_len, embed_dim))

        ### return the embedding of the output
        return self.out_proj(attn_output)


