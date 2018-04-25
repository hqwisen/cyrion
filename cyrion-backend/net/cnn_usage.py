# imports for array-handling and plotting
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# imports for operators images loading
import glob
from scipy import misc

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

matplotlib.get_backend()


if __name__ == "__main__":
    print("Running script")
