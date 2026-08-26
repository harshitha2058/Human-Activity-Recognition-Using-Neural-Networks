import tensorflow as tf
from tensorflow.keras import layers, models

def build_transformer_har_model(input_dim: int, num_classes: int = 6, num_heads: int = 4, key_dim: int = 32) -> models.Model:
    inputs = layers.Input(shape=(input_dim,))
    
    # Reshape input into sequence form for attention (e.g., sequence length x feature dimension)
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # Multi-Head Attention Mechanism
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-Forward Network
    ffn = layers.Conv1D(filters=64, kernel_size=1, activation='relu')(x)
    ffn = layers.Dropout(0.2)(ffn)
    ffn = layers.Conv1D(filters=1, kernel_size=1)(ffn)
    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="HAR_Transformer")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model