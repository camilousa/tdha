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
     Input(shape=(52, n_features)),
     LSTM(128, return_sequences=False),

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
