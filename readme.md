VAE-based Image Reconstruction for MNIST Dataset

This repository implements a Variational Autoencoder (VAE) for image reconstruction using the MNIST dataset. The VAE model learns to encode and decode images, handling the latent space in a probabilistic manner.
Table of Contents

    Introduction
    Project Structure
    Installation
    Usage
    Training
    Results
    Licenses
    Contributions

Introduction

This project demonstrates the application of a Variational Autoencoder (VAE) to reconstruct handwritten digits from the MNIST dataset. The VAE is a generative model that learns a probabilistic mapping from data to a latent space, allowing for more robust image reconstruction and handling of anomalies compared to regular autoencoders. The model's architecture is built with PyTorch and utilizes convolutional layers in both the encoder and decoder parts.
Project Structure

The repository has the following structure:

VAE-MNIST/
│
├── data/                     # Directory to store MNIST data
├── images/                   # Folder for images (if any visualizations or examples are included)
├── models_mnist/             # Folder where trained models are saved
│
├── main.py                   # Main script to train the VAE model
├── utils.py                  # Utility functions
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies

Installation

To set up the environment and install dependencies, follow these steps:

    Clone the repository:

git clone https://github.com/yourusername/VAE-MNIST.git
cd VAE-MNIST

Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # For Windows, use `venv\Scripts\activate`

Install the required dependencies:

    pip install -r requirements.txt

Usage
Running the Model

To train the VAE model on the MNIST dataset, run:

python main.py

This will start the training process. The best model will be saved in the models_mnist/ directory.
Visualizing Results

After training, the best model is saved, and you can use it to visualize the reconstruction results. The reconstructed images will be displayed for the sample input.
Training
Loss Function

The VAE uses a combination of Binary Cross-Entropy (BCE) for the reconstruction loss and KL Divergence to regularize the latent space. The training is done for a number of epochs, and the model saves the best performing one based on validation loss.
Optimizer

The Adam optimizer is used to minimize the loss function, with an initial learning rate of 1e-3.
Hyperparameters

    Batch size: 64
    Learning rate: 1e-3
    Epochs: 20

Results

After training, you will be able to visualize the reconstructed images from the VAE model. The model's performance is evaluated based on the reconstruction loss and KL divergence.
Sample Results:

Licenses

This project is licensed under the MIT License - see the LICENSE file for details.
Contributions

Feel free to fork the project and create pull requests for any improvements or bug fixes. Contributions are always welcome!