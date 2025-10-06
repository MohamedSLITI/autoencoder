import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
from matplotlib.animation import FuncAnimation

# Load MNIST dataset
(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# Define Autoencoder architecture
input_dim = x_train.shape[1]
encoding_dim = 2  # for visualization

input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(encoding_dim, activation='linear', name='latent')(encoded)

decoded = Dense(64, activation='relu')(latent)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
encoder = Model(input_layer, latent)

# Compile & train
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(x_train, x_train,
                          epochs=10,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test),
                          verbose=1)

# Encode test data into 2D latent space
encoded_data = encoder.predict(x_test)

# Create animated scatter of latent space
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(encoded_data[:, 0], encoded_data[:, 1], c=y_test, cmap='tab10', s=8)
ax.set_title("2D Latent Space Representation (Autoencoder)")
ax.set_xlabel("Latent Dimension 1")
ax.set_ylabel("Latent Dimension 2")

# Create a simple zoom animation
def update(frame):
    zoom = 1 + 0.02 * frame
    ax.set_xlim(-10/zoom, 10/zoom)
    ax.set_ylim(-10/zoom, 10/zoom)
    return sc,

ani = FuncAnimation(fig, update, frames=60, interval=100)
ani.save("autoencoder_latent.gif", writer="pillow", fps=10)

plt.show()
