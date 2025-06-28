import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
print("TensorFlow Version:", tf.__version__)
print("Using CPU:", not tf.config.list_physical_devices('GPU'))

from tensorflow.keras.applications import MobileNetV2  
print("MobileNetV2 Loaded Successfully")

