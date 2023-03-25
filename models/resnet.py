# -*- coding: utf-8 -*-
"""ResNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VOu-YtbSDcuF7HGvHtqqjgF-_lTUfdoO
"""

import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam

class ResNet(tf.keras.Model):
    def __init__(self, dropout_rate=0.2):
        super().__init__()

        self.res = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in self.res.layers[:-10]:
            layer.trainable = False

        self.optimizer = Adam(learning_rate=0.0001)
        
        self.dropout_rate = dropout_rate

        """self.training_custom_layers = [
                         Flatten(),
                         Dense(512, activation='relu'),
                         BatchNormalization(),
                         Dropout(self.dropout_rate),
                         Dense(256, activation='relu'),
                         BatchNormalization(),
                         Dropout(self.dropout_rate),
                         Dense(4, activation='softmax')
                         ]"""
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.batch1 = BatchNormalization()
        self.drop1 = Dropout(self.dropout_rate)
        self.dense2 = Dense(256, activation='relu')
        self.batch2 = BatchNormalization()
        self.drop2 = Dropout(self.dropout_rate)
        self.out = Dense(4, activation='softmax')
        
    def call(self, x, trainable=False):
        x = self.res(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch1(x, trainable=trainable)
        x = self.drop1(x, trainable=trainable)
        x = self.dense2(x)
        x = self.batch2(x, trainable=trainable)
        x = self.drop2(x, trainable=trainable)
        x = self.out(x)
        """if trainable:
            for layer in self.training_custom_layers:
                x = layer(x)
        else:
            for layer in self.training_custom_layers:"""
        return x
