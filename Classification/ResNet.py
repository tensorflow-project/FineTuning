import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import tqdm

def image_preprocessing(path, images):
  """Preprocesses images by resizing them to (224, 224) and converting them to numpy arrays
    
    Args:
    - path (str): The directory path containing the images
    - images (List[np.ndarray]): a list to which the preprocessed images will be appended
    
  """

  file_list = os.listdir(path)
  for file in file_list:
    img = Image.open(path + file)
    img = img.resize((224, 224))
    images.append(np.array(img))

def dataset(path, label):
  """Creates a TensorFlow dataset from image files in the specified directory path and assigns the given label to all images

    Args:
    - path (str): the path to the directory containing image files
    - label (int): the label to be assigned to all images in the dataset

    Returns:
    - dataset (tf.data.Dataset): a TensorFlow dataset object containing the preprocessed images and their corresponding labels

  """

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
  """A residual layer implementation for use in a neural network.

    Args:
    - num_filters (int): the number of filters for the convolutional layer

    Attributes:
    - conv (tf.keras.layers.Conv2D): the convolutional layer with relu activation
  """
  def __init__(self, num_filters):
    super(ResLayer, self).__init__()
    self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation='relu')

  def call(self, x):
    """Performs the forward pass with a residual connection to keep the input alive
    
    """
    c = self.conv(x)
    x = c+x
    return x

class ResBlock(tf.keras.layers.Layer):
  """A Residual Block layer for a ResNet model, it consists of a convolutional layer followed by a series of Residual Layers

    Args:
    - depth (int): number of filters in the convolutional layer
    - layers (int): number of residual layers in the block
    
  """
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
  """A Residual Network for image classification

    Attributes:
    - loss_function: The loss function used for training the model.
    - optimizer: The optimizer used for training the model.
    - metrics_list: A list of metrics used to evaluate the model's performance during training and testing.
    - layers_list: A list of layers that define the ResNet architecture.
  """
  def __init__(self):
    super(ResNet, self).__init__()
    ### we use Categorical Crossentropy and Categorical Accuracy as we have more than two classes
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
        ### we need 4 units in the output layer as we have four different classes
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
    """Train one batch of data using the model

    Args:
    - data: a tuple of (images, labels) representing a batch of training data

    Returns:
    - a dictionary of the model's metrics and their results for the batch
    
    """
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
    """Runs a single evaluation step on a batch of data. Computes the forward pass and calculates the loss and evaluation metrics

    Args:
    - data: A batch of data consisting of images and labels

    Returns:
    - a dictionary of the evaluation metrics' names and their corresponding results
    """
    img, label = data

    output = self(img, training = True)
    loss = self.loss_function(label, output)


    self.metrics[0].update_state(label, output)
    self.metrics[1].update_state(loss)
    return {m.name : m.result() for m in self.metrics}  
    

def create_summary_writers(config_name):
  """Create TensorFlow summary writers for training and validation metrics

  Args:
  - config_name (str): name of the configuration being used. Used to organize log files

  Returns:
  - Tuple[tf.summary.SummaryWriter, tf.summary.SummaryWriter]: a tuple of two summary writers, the first is for writing
    training metrics, and the second is for writing validation metrics
  """
    
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
def training_loop(model, epochs, train_ds, test_ds, train_summary_writer, val_summary_writer):
  """training loop for a given model, iterates through the batches of data and calls the train and test step on them
  Args:
  - model: instance of a tf.keras.Model subclass
  - epochs: number of epochs to train the model for
  - train_ds: training dataset
  - test_ds: testing dataset
  - train_summary_writer: summary writer for training metrics
  - val_summary_writer: summary writer for validation metrics
  """
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
