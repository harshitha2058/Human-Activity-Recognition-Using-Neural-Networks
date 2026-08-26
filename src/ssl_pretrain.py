import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_ssl_encoder(input_dim: int):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    bottleneck = layers.Dense(64, activation='relu', name="encoder_bottleneck")(x)
    reconstruction = layers.Dense(input_dim, activation='linear', name="decoder_output")(bottleneck)
    
    autoencoder = models.Model(inputs=inputs, outputs=reconstruction, name="SSL_Autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def run_ssl_training(X_train: np.ndarray, mask_ratio: float = 0.2):
    # Mask random features to force the model to learn signal dynamics
    X_masked = X_train.copy()
    mask = np.random.rand(*X_train.shape) < mask_ratio
    X_masked[mask] = 0.0
    
    ssl_model = build_ssl_encoder(input_dim=X_train.shape[1])
    ssl_model.fit(X_masked, X_train, epochs=20, batch_size=32, validation_split=0.1)
    return ssl_model