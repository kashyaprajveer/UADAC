import tensorflow as tf
import numpy as np

class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x): # x is a list
        
        # print("This is x:", len(x[0][0]), len(x[1][0]))  # -> This is x: 100 10
        
        items_eb = tf.transpose(x[1], perm=(0,2,1))/self.embedding_dim        # print(items_eb.shape) -> (1, 100, 10)
        wav = self.wav(items_eb)                                              # print(wav.shape) -> (1, 100, 1)
        wav = tf.transpose(wav, perm=(0,2,1))
        wav = tf.squeeze(wav, axis=1)
        user_wav = tf.keras.layers.multiply([x[0], wav])
        concat = self.concat([x[0], user_wav, wav])
        return self.flatten(concat)