import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import tqdm

def image_preprocessing(path, images):
  file_list = os.listdir(path)
  for file in file_list:
    img = Image.open(path + file)
    img = img.resize((224, 224))
    images.append(np.array(img))

def dataset(path, label):
  images = []
  image_preprocessing(path, images)
  labels = np.full(len(images), label)
  images = tf.convert_to_tensor(images)
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  dataset = dataset.map(lambda img, label: (tf.cast(img, tf.float32), label))
  dataset = dataset.map(lambda img, label: ((img/128.)-1., label))
  dataset = dataset.map(lambda img, label: (img, tf.cast(label, tf.int32)))
  dataset = dataset.map(lambda img, label: (img, tf.one_hot(label, depth=4)))
  dataset = dataset.cache()

  dataset = dataset.shuffle(100)
  dataset = dataset.batch(2)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)

  return dataset



class ResLayer(tf.keras.layers.Layer):
  def __init__(self, num_filters):
    super(ResLayer, self).__init__()
    self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')

  def call(self, x):
    c = self.conv(x)
    x = c+x
    return x

class ResBlock(tf.keras.layers.Layer):
  def __init__(self, depth, layers):
    super(ResBlock, self).__init__()
    self.deeper_layer = tf.keras.layers.Conv2D(filters=depth, kernel_size=3, padding='same', activation='relu')
    self.layers = [ResLayer(depth) for _ in range(layers)]

  def call(self, x):
    x = self.deeper_layer(x)
    for layer in self.layers:
      x = layer(x)
    return x


class ResNet(tf.keras.Model):
  def __init__(self):
    super(ResNet, self).__init__()

    self.loss_function = tf.keras.losses.CategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam()

    self.metrics_list = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Mean(name = "loss")]
    self.loss_function = tf.keras.losses.CategoricalCrossentropy()

    self.layers_list = [
        ResBlock(24, 4),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        ResBlock(48, 4),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        ResBlock(96, 4),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ]

  def call(self,x):
    for layer in self.layers_list:
        x = layer(x)
    return x


  @property
  def metrics(self):
    return self.metrics_list

  def reset_metrics(self):
    for metric in self.metrics:
      metric.reset_states()

  @tf.function
  def train_step(self, data):
    img, label = data

    with tf.GradientTape() as tape:
      output = self(img, training = True)
      loss = self.loss_function(label, output)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    self.metrics[0].update_state(label, output)
    self.metrics[1].update_state(loss)

    return {m.name : m.result() for m in self.metrics}

  @tf.function
  def test_step(self, data):
    img, label = data

    output = self(img, training = True)
    loss = self.loss_function(label, output)


    self.metrics[0].update_state(label, output)
    self.metrics[1].update_state(loss)
    return {m.name : m.result() for m in self.metrics}  
    

def create_summary_writers(config_name):
    
    #define where to save logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_path = f"logs/{config_name}/{current_time}/train"
    val_log_path = f"logs/{config_name}/{current_time}/val"

    # log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)

    # log writer for validation metrics
    val_summary_writer = tf.summary.create_file_writer(val_log_path)
    
    return train_summary_writer, val_summary_writer
   
#training loop
def training_loop(model, epochs, train_ds, test_ds):
  for e in range(epochs):
    #training
    for data in tqdm.tqdm(train_ds, position = 0, leave = True):
      metrics = model.train_step(data)
    #for scalar metrics: save logs
    with train_summary_writer.as_default(): 
      for metric in model.metrics:
        tf.summary.scalar(f"{metric.name}", metric.result(), step=e)

    print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

    model.reset_metrics()

    #testing
    for data in test_ds:
      metrics = model.test_step(data)

    with val_summary_writer.as_default():
        # for scalar metrics:
        for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=e)

    print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])

    # reset metric objects
    model.reset_metrics()
