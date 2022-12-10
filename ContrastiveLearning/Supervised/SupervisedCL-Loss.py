import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


# The Supervised Contrastive Loss Function, used when pretraining the encoder
class SupervisedContrastiveLoss(keras.losses.Loss): 
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        logits = (feature_vectors_normalized@feature_vectors_normalized.T)/self.temperature
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def add_projection_head(encoder):
    features = encoder(keras.Input(shape=input_shape))
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

encoder = create_encoder()
