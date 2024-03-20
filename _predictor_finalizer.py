# -*- coding: utf-8 -*-

# input libraries
import pickle
import numpy as np


tf.__version__

# google drive mount
from google.colab import drive
drive.mount('/path/to/your/drive')

# Replace '/path/to/your/predictions1' with the path to your pickle file
file_path1 = '/path/to/your/predictions1'

# Load the array from the pickle file
with open(file_path1, 'rb') as file:
    predictions1 = pickle.load(file)

# Replace '/path/to/your/predictions2' with the path to your pickle file
file_path2 = '/path/to/your/predictions2'

# Load the array from the pickle file
with open(file_path2, 'rb') as file:
    predictions2 = pickle.load(file)

# Replace '/path/to/your/predictions3' with the path to your pickle file
file_path3 = '/path/to/your/predictions3'

# Load the array from the pickle file
with open(file_path3, 'rb') as file:
    predictions3 = pickle.load(file)

weights=[0.3, 0.2, 0.5]

combined_predictions = (weights[0] * predictions1 +
                        weights[1] * predictions2 +
                        weights[2] * predictions3)

predicted_classes = np.argmax(combined_predictions[0], axis=1)
