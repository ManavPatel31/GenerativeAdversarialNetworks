# -*- coding: utf-8 -*-
"""Assignment4_301147682.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iAOJv2LVQwH2HtX0dnUfbOsLBTXsPG48
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""#Load the fashion_mnist dataset"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

"""# Create dictionaries for the two datasets

"""

ds1_manav = {'images': x_train[:60000], 'labels': y_train[:60000]}
ds2_manav = {'images': x_test[:10000], 'labels': y_test[:10000]}

"""# Load the datasets into variables ds1_manav and ds2_manav

## Step 1: Normalize pixel values to range between -1 and 1
"""

ds1_manav['images'] = ds1_manav['images'] / 255.0
ds2_manav['images'] = ds2_manav['images'] / 255.0

"""## Step 2: Print shape of images in both datasets"""

print("Shape of ds1", ds1_manav['images'].shape)
print("Shape of ds2", ds2_manav['images'].shape)

"""## Select only the pants images (class label 1) from the datasets

"""

ds1_manav = ds1_manav['images'][ds1_manav['labels'] == 1]
ds2_manav = ds2_manav['images'][ds2_manav['labels'] == 1]

"""## Concatenate the two datasets into one

"""

dataset_manav = np.concatenate((ds1_manav, ds2_manav))

"""## Print the shape of the concatenated dataset

"""

print("Shape of df", dataset_manav.shape)

"""## Display first 12 images from the dataset

"""

plt.figure(figsize=(8,8))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.imshow((dataset_manav[i] + 1) / 2) 
    plt.xticks([])
    plt.yticks([])
plt.show()

"""## Step 6: Create a training dataset with shuffled and batched images

"""

train_dataset_manav = tf.data.Dataset.from_tensor_slices(dataset_manav)
train_dataset_manav = train_dataset_manav.shuffle(7000).batch(256)

ds = tf.data.Dataset.from_tensor_slices(dataset_manav)

"""# Shuffle the dataset with a buffer size of 7000 (to shuffle all the images)"""

ds = ds.shuffle(7000)

"""# Batch the dataset with a batch size of 256

"""

ds = ds.batch(256)

"""# Set the prefetch option to improve performance

"""

ds = ds.prefetch(tf.data.AUTOTUNE)

"""# Assign the shuffled and batched dataset to train_dataset_manav

"""

train_dataset_manav = ds

"""# Define the generator model

"""

generator_model_manav = tf.keras.Sequential([
    tf.keras.layers.Dense(7*7*256, input_dim=100, use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Reshape((7, 7, 256)),
    tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
])

tf.keras.utils.plot_model(generator_model_manav, show_shapes=True, show_layer_names=True)

"""# Print the generator model summary

"""

generator_model_manav.summary()

import matplotlib.pyplot as plt

"""# Create a sample vector with dimension size 100

"""

sample_vector = tf.random.normal([1, 100])

"""# Disable training

"""

tf.keras.backend.set_learning_phase(0)

"""# Generate an image from generator_model_manav using the sample vector

"""

generated_image = generator_model_manav(sample_vector, training=False)

"""# Plot the generated image

"""

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

"""# Define the discriminator model

"""

discriminator_model_manav = tf.keras.Sequential([
    # 1st Layer: Convolutional Layer with 64 filters
    tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', input_shape=(28,28,1)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(rate=0.3),
    
    # 2nd Layer: Convolutional Layer with 128 filters
    tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(rate=0.3),
    
    # 3rd Layer: Transposed Convolutional Layer with 64 filters
    tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    
    # Output Layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

tf.keras.utils.plot_model(discriminator_model_manav, show_shapes=True, show_layer_names=True)

"""# Display a summary of the model

"""

discriminator_model_manav.summary()

cross_entropy_manav = tf.keras.losses.BinaryCrossentropy(from_logits=False)

generator_optimizer_manav = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer_manav = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

@tf.function
def train_step(images):
    noise = tf.random.normal([256, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model_manav(noise, training=True)

        real_output = discriminator_model_manav(images, training=True)
        fake_output = discriminator_model_manav(generated_images, training=True)

        gen_loss = cross_entropy_manav(tf.ones_like(fake_output), fake_output)
        real_loss = cross_entropy_manav(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy_manav(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model_manav.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model_manav.trainable_variables)

    generator_optimizer_manav.apply_gradients(zip(gradients_of_generator, generator_model_manav.trainable_variables))
    discriminator_optimizer_manav.apply_gradients(zip(gradients_of_discriminator, discriminator_model_manav.trainable_variables))

import time
epochs = 10

"""# Training loop

"""

for epoch in range(epochs):
    start_time = time.time()
    print("Epoch {}/{}".format(epoch+1, epochs))
    for image_batch in train_dataset_manav:
        train_step(image_batch)
    end_time = time.time()
    print("Time taken for epoch {} is {:.2f} seconds\n".format(epoch+1, end_time - start_time))

"""# 1. Generate 16 sample vectors

"""

noise = tf.random.normal([16, 100])

"""# 2. Generate image from trained generator model

"""

generated_images = generator_model_manav(noise, training=False)

"""# 3. Normalize pixels in generated images

"""

generated_images = generated_images * 127.5 + 127.5

# Plot generated images
plt.figure(figsize=(8, 8))
for i in range(generated_images.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()

