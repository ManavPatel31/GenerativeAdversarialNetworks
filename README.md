# Generative Adversarial Network
Creating a Generative Adversarial Network Model

In this exercise, we are required to build a Generative Adversarial Network (GAN) using the fashion_mnist dataset from TensorFlow. The steps to be followed are outlined below.

# Step 1: Get the data
The first step is to import and load the fashion_mnist dataset from TensorFlow. We are required to store the first 60,000 data samples in ds1_firstname directory with keys 'images' and 'labels', while the next 10,000 data samples will be stored in ds2_firstname directory with keys 'images' and 'labels'. We are also required to normalize the pixel values in the dataset to a range between -1 to 1, and then create a new dataset named dataset_firstname containing pants images (class label 1) from ds1_firstname and ds2_firstname. Finally, we are to display the shape of ds1_firstname['images'], ds2_firstname['images'], and dataset_firstname, respectively.

# Step 2: Display the first 12 images from the dataset
We are required to display the first 12 images from the dataset using matplotlip. We need to remove xticks and yticks when plotting the image, and plot the images using a figure size of 8x8 and a subplot dimension of 4x3.

# Step 3: Build the Generator Model of the GAN
We are required to use TensorFlow's Sequential() to build a CNN model (generator_model_firstname) with the following architecture:

Input = Vector with dimension size 100
1st Layer = Fully connected Layer with 77256 neurons and no bias term
2nd Layer = Batch Normalization
3rd Layer = Leaky ReLU activation
4th Layer = Transposed Convolution Layer with 128 kernels with window size 5x5, no bias, 'same' padding, stride of 1x1. Note: Input to the Transposed Layer should first be reshaped to (7,7,256).
5th Layer = Batch Normalization
6th Layer = Leaky ReLU
7th Layer = Transposed Convolution Layer with 64 kernels with window size 5x5, no bias, 'same' padding, stride of 2x2.
8th Layer = Batch Normalization
9th Layer = Leaky ReLU
7th Layer = Transposed Convolution Layer with 1 kernels with window size 5x5, no bias, 'same' padding, stride of 2x2, and tanh activation
We are also required to display a summary of the model using summary(). Finally, we need to draw a diagram illustrating the structure of the neural network model, making note of the size of each layer (# of neurons) and number of weights in each layer.

# Step 4: Sample untrained generator
We are required to sample the untrained generator_model_firstname by generating a batch of 12 images using random noise as input. We need to plot the generated images in a 4x3 subplot using matplotlib, with no xticks and yticks, and a figure size of 8x8.

# Step 5: Build the Discriminator Model of the GAN
We are required to use TensorFlow's Sequential() to build a CNN model (discriminator_model_firstname) with the following architecture:

1st Layer = Convolution Layer with 64 kernels with window size 5x5, no bias, 'same' padding, stride of 2x2
2nd Layer = Leaky ReLU
3rd Layer = Convolution Layer with 128 kernels with window size 5x5, no bias, 'same' padding, stride of 2x2
4th Layer = Leaky ReLU
5th Layer = Flatten
6th Layer = Fully Connected Layer with 1024 neurons and no bias term
7th Layer = Leaky ReLU
8th Layer = Fully Connected Layer with 1 neuron and sigmoid activation
We are also required to display a summary of the model using summary(). Finally, we need to draw a diagram illustrating the structure of the neural network model, making note of the size of each layer (# of neurons) and number of weights in each layer.

# Step 6: Compile the GAN model
We are required to compile the GAN model by setting the loss function to binary_crossentropy, optimizer to Adam with learning rate 0.0002, beta1=0.5, and beta2=0.999. We also need to compile the discriminator_model_firstname separately with the same optimizer and loss function.

# Step 7: Train the GAN model
We are required to train the GAN model for 300 epochs, with a batch size of 128. We need to display the progress of the model training by plotting the losses of both generator and discriminator during training using matplotlib. We also need to display a grid of 12 generated images after every 50 epochs of training using the same method as in step 4.

# Step 8: Save the GAN model
We are required to save the generator_model_firstname and discriminator_model_firstname as h5 files in the current directory.


# Conclusion: 
Overall, the exercise requires the creation of a Generative Adversarial Network (GAN) using the Fashion MNIST dataset. The dataset is first preprocessed by normalizing the pixel values and creating a new dataset by concatenating pants images from the two subsets of the dataset. The GAN consists of two models, a generator model and a discriminator model. The generator model is designed to generate images from random input vectors, while the discriminator model is trained to distinguish between real images from the dataset and generated images from the generator model. The two models are trained together in an iterative process, where the generator model aims to generate realistic images that fool the discriminator model, and the discriminator model aims to correctly distinguish between real and fake images. The training process involves optimizing the loss function using the Adam optimizer and minimizing the binary cross-entropy loss. Finally, the trained generator model is used to generate new images from random input vectors.
