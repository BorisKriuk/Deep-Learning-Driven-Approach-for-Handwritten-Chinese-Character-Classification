# -*- coding: utf-8 -*-

# input libraries
import numpy as np
import tensorflow as tf
import csv
import os
import shutil
import pickle


batch_size=128
crop_size = 256

# google drive mount
from google.colab import drive
drive.mount('/path/to/your/drive')

!unzip '/path/to/your/drive/folder/CASIA-HWDB.zip'

source_dir = '/content/CASIA-HWDB_Test/Test'
destination_dir = '/content/safe_images_test'
specific_file_to_delete = os.path.join(destination_dir, 'X/49.png')  # Delete corrupted image if needed

# Copy the entire directory tree from source_dir to destination_dir
shutil.copytree(source_dir, destination_dir)

# After the copy is finished, delete the specific file
if os.path.exists(specific_file_to_delete):
    os.remove(specific_file_to_delete)
    print(f"Deleted: {specific_file_to_delete}")
else:
    print(f"File not found, could not delete: {specific_file_to_delete}")

print("Finished copying the directory tree and deleting the specific file.")

# load model
model2 = tf.keras.models.load_model('/path/to/your/model2')

# Define the dataset loading function
def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, 280, 280)
    return image

# Create a dataset from the image directory
list_ds = tf.data.Dataset.list_files(os.path.join('/content/safe_images_test', '*.png'), shuffle=False)
dataset = list_ds.map(load_image).batch(1)  # Keep batch size 1 for individual processing

for batch in dataset.take(1):
    images = batch  # Adjust this line if load_image returns something different

    # Inspect the shapes
    print("Image batch shape:", images.shape)

def multicrop(image, size):
    image = tf.image.resize(image, [size, size])
    crops = []
    # Center crop
    crops.append(tf.image.resize_with_crop_or_pad(image, crop_size, crop_size))
    # Corner crops
    crops.append(tf.image.crop_to_bounding_box(image, 0, 0, crop_size, crop_size))
    crops.append(tf.image.crop_to_bounding_box(image, 0, size - crop_size, crop_size, crop_size))
    crops.append(tf.image.crop_to_bounding_box(image, size - crop_size, 0, crop_size, crop_size))
    crops.append(tf.image.crop_to_bounding_box(image, size - crop_size, size - crop_size, crop_size, crop_size))
    # Stack crops to create a new batch dimension
    return tf.stack(crops, axis=0)

# Preprocess and predict, 5 crops
all_predictions = []
for batch in dataset:
    # Apply multicrop to each image in the batch
    # Make sure the multicrop function returns the correct shape
    crops_batch = tf.map_fn(
    lambda x: multicrop(x, size=max(crop_size + 1, x.shape[1], x.shape[2])),
    batch,
    fn_output_signature=tf.TensorSpec(shape=(5, crop_size, crop_size, 3), dtype=tf.float32))
    # Predict on the crops
    # Reshape to a large batch of images rather than a batch of batches
    crops_batch = tf.reshape(crops_batch, [-1, crop_size, crop_size, 1])

    # Make predictions on the crops
    predictions = model2.predict(crops_batch, batch_size=batch_size)

    # Aggregate predictions for each image (e.g., mean)
    num_crops = crops_batch.shape[0]
    predictions = np.reshape(predictions, (batch.shape[0], num_crops, -1))
    aggregated_predictions = np.mean(predictions, axis=1)

    # Store the aggregated predictions
    all_predictions.extend(aggregated_predictions)

predictions2= tf.nn.softmax(all_predictions).numpy()

# Specify the filename
filename = '/path/to/your/predictions2'

# Save the array to a file using pickle
with open(filename, 'wb') as file:
    pickle.dump(predictions2, file)

print(f"predictions2 saved as {filename}")
