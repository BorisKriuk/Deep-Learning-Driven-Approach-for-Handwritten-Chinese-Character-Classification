# -*- coding: utf-8 -*-

# input libraries
!pip install tensorflow-addons
import keras.utils as image
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, Concatenate


image_size=256
batch_size=128

tf.__version__

# google drive mount
from google.colab import drive
drive.mount('/path/to/your/drive')

!unzip '/path/to/your/drive/folder/CASIA-HWDB.zip'

source_dir = '/content/CASIA-HWDB_Train/Train'
destination_dir = '/content/safe_images_train'
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

# Replace '/content/safe_images_train/я┐е/98.png' with your actual image file path
path_to_image = '/content/safe_images_train/я┐е/98.png'

# Open the image file
with Image.open(path_to_image) as img:
    # Get image size
    width, height = img.size

# Print out image size
print(f"The image size is {width}x{height} pixels.")

idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=False,
    vertical_flip=False,
    validation_split=0.05
)

original_train_gen = idg.flow_from_directory('/content/safe_images_train',
                                                    target_size=(image_size, image_size),
                                                    subset='training',
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',
                                                    shuffle=True,
                                                    seed=1
                                                )

original_val_gen = idg.flow_from_directory('/content/safe_images_train',
                                                    target_size=(image_size, image_size),
                                                    subset='validation',
                                                    class_mode='categorical',
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',
                                                    shuffle=True,
                                                    seed=1
                                                )

x,y = next(original_train_gen)

def show_grid(image_list, nrows, ncols, label_list=None, show_labels=False, figsize=(10,10)):

    fig = plt.figure(None, figsize,frameon=False)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.2,
                     share_all=True,
                     )
    for i in range(nrows*ncols):
        ax = grid[i]
        ax.imshow(image_list[i],cmap='Greys_r')
        ax.axis('off')

show_grid(x,2,4,show_labels=True,figsize=(10,10))

# Define the corrected residual block
def residual_block(x, filters, kernel_size=(3, 3), activation='relu'):
    # Shortcut path
    shortcut = Conv2D(filters, (1, 1), strides=(2, 2), padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    # Main path
    y = Conv2D(filters, kernel_size, strides=(2, 2), padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)

    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)

    # Add the shortcut to the main path
    y = Add()([y, shortcut])
    y = Activation(activation)(y)
    return y

def inception_block(x, filters):
    # Each branch is a stack of layers with a common input 'x' and different filter operations
    branch1x1 = Conv2D(filters=filters[0], kernel_size=(1, 1), padding='same', activation='relu')(x)

    branch3x3 = Conv2D(filters=filters[1], kernel_size=(1, 1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(filters=filters[2], kernel_size=(3, 3), padding='same', activation='relu')(branch3x3)

    branch5x5 = Conv2D(filters=filters[3], kernel_size=(1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters=filters[4], kernel_size=(5, 5), padding='same', activation='relu')(branch5x5)

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters=filters[5], kernel_size=(1, 1), padding='same', activation='relu')(branch_pool)

    # Concatenate the outputs (on the channel dimension) of each branch
    output = Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool])
    return output

# Access the image_shape attribute from the original DirectoryIterator
image_shape = original_train_gen.image_shape
print(f"The shape of the images is: {image_shape}")

input_layer = Input(shape=(image_size, image_size, 1))

# First Conv Block
x = Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

# First residual block
x = residual_block(x, filters=32)

# First Inception module
x = inception_block(x, filters=[32, 48, 64, 16, 32, 32])

# Auxiliary output 1
aux_output1 = Flatten()(x)
aux_output1 = Dense(512, activation='relu')(aux_output1)
aux_output1 = Dropout(0.5)(aux_output1)
aux_output1 = Dense(256, activation='relu')(aux_output1)
aux_output1 = Dense(7330, activation='linear', name='aux_output1')(aux_output1)

# Second Conv Block
x = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

# Second residual block
x = residual_block(x, filters=64)

# Second Inception module
x = inception_block(x, filters=[64, 96, 128, 32, 64, 64])

# Auxiliary output 2
aux_output2 = Flatten()(x)
aux_output2 = Dense(512, activation='relu')(aux_output2)
aux_output2 = Dropout(0.5)(aux_output2)
aux_output2 = Dense(256, activation='relu')(aux_output2)
aux_output2 = Dense(7330, activation='linear', name='aux_output2')(aux_output2)

# Third Conv Block
x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

# Third residual block
x = residual_block(x, filters=128)

# Third Inception module
x = inception_block(x, filters=[128, 182, 256, 64, 128, 128])

# Auxiliary output 3
aux_output3 = Flatten()(x)
aux_output3 = Dense(512, activation='relu')(aux_output3)
aux_output3 = Dropout(0.5)(aux_output3)
aux_output3 = Dense(256, activation='relu')(aux_output3)
aux_output3 = Dense(7330, activation='linear', name='aux_output3')(aux_output3)

# Fourth Conv Block
x = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)

# Fourth residual block
x = residual_block(x, filters=256)

# Fourth Inception module
x = inception_block(x, filters=[256, 364, 512, 128, 256, 256])

# Auxiliary output 4
aux_output4 = Flatten()(x)
aux_output4 = Dense(512, activation='relu')(aux_output4)
aux_output4 = Dropout(0.5)(aux_output4)
aux_output4 = Dense(256, activation='relu')(aux_output4)
aux_output4 = Dense(7330, activation='linear', name='aux_output4')(aux_output4)

# Fifth Conv Block
x = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Main output 4
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.6)(x)
main_output = Dense(7330, activation='linear', name='main_output')(x)

# Define the model with input layer and three output layers
model = Model(inputs=input_layer, outputs=[main_output, aux_output1, aux_output2, aux_output3, aux_output4])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, gamma=4, from_logits=True), metrics=['accuracy'], loss_weights={'main_output': 1.0, 'aux_output1': 0.025, 'aux_output2': 0.05, 'aux_output3': 0.5, 'aux_output4': 0.2})

model.summary()

train_steps=int(len(original_train_gen.filenames))
val_steps=int(len(original_val_gen.filenames))

# Define the checkpoint path to save the best weights
checkpoint_path = 'path/to/your/checkpoint'

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_main_output_accuracy',  # Monitor validation accuracy
    save_best_only=True,  # Save only the best model
    save_weights_only=True,  # Save only the model weights
    mode='max',  # Mode for monitoring (maximize validation accuracy)
    verbose=1  # Print information about saving the weights
)

# Train your model
history=model.fit(
    original_train_gen,  # Training data
    validation_data=original_val_gen,  # Validation data
    steps_per_epoch= train_steps//batch_size,
    epochs=20,  # Number of epochs
    verbose = 1,
    validation_steps = val_steps//batch_size,
    callbacks=[checkpoint_callback]  # Add the checkpoint callback
)
model.save(checkpoint_path)
