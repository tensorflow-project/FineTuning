from numpy import dot
import tensorflow as tf
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('ps')


def cosine_sim(e1,e2):
    """Calculate the cosine similarity between two vectors.
    Args:
    - e1 (array): First vector
    - e2 (array): Second vector
    Returns:
    - float: The cosine similarity between the two vectors
    """
    sim = dot(e1, e2)/(norm(e1)*norm(e2))
    return sim
  
  
def plot(x_values, y_values, color, xlabel, ylabel, title):
    """Plot a line graph using Matplotlib.

    Args:
    - x_values (array-like): x-coordinates of data points to be plotted
    - y_values (array-like): y-coordinates of data points to be plotted
    - color (str): color of the line
    - xlabel (str): label of the x-axis
    - ylabel (str): label of the y-axis
    - title (str): title of the plot

    Returns:
    - None

    Raises:
    - None
    """
    plt.figure()

    plt.plot(x_values, y_values, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
