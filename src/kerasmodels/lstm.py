import keras
from keras.layers import *
import tensorflow as tf


METRICS = [
                      'accuracy',
                     'precision',
                     'recall',
                     'auc',
                     'true_positives',
                     'true_negatives',
                     'false_positives',
                     'false_negatives'
                ]


def build_model(n_features):
     model = keras.Sequential([
    Conv1D(16, 10, activation='relu', input_shape=(52, n_features)),
    Conv1D(32, 3, activation='relu'),
    Conv1D(64, 3, activation='relu'),
 
    Flatten(),
    Dense(1, activation='sigmoid')
])
     
     model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='binary_crossentropy',
                  metrics=[
                      'accuracy',
                     'precision',
                     'recall',
                     'auc',
                     'true_positives',
                     'true_negatives',
                     'false_positives',
                     'false_negatives'
                ])


     return model
