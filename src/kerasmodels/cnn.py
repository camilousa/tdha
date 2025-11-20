import keras
from keras.layers import *
import tensorflow as tf

def build_model():
     model = keras.Sequential([
            Input(shape=(18, 18, 1), name="input_layer"),
            Conv2D(128, (3,3), activation='relu', name="conv2d_layer1"),
            Conv2D(256, (3,3), activation='relu', name="conv2d_layer2"),
            Conv2D(384, (3,3), activation='relu', name="conv2d_layer3"),
            Flatten(name="flatten"),
            Dense(192, activation='relu', name="dense96"),
            Dense(1, activation='sigmoid', name="dense_output"),
     ])
     model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall'])

     return model
