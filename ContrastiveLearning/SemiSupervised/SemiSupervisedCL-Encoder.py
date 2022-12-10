# adapted and modified from: https://github.com/keras-team/keras-io/blob/master/examples/vision/semisupervised_simclr.py
import tensorflow as tf
from tensorflow import keras

def get_encoder():
    return keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(256,256,1), pooling="avg")

class ContrastiveModel(keras.Model):
    def __init__(self):
        self.temperature = temperature
        self.contrastive_augmenter = get_augmenter(**contrastive_augmentation)
        self.encoder = get_encoder()
        self.projection_head = keras.Sequential([keras.Input(shape=(width,)),
                layers.Dense(width, activation="relu"),layers.Dense(width)])

    def compile(self, contrastive_optimizer):
        self.contrastive_optimizer = contrastive_optimizer

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name="c_acc")


    def metrics(self):
        return [self.contrastive_loss_tracker, self.contrastive_accuracy]

    def contrastive_loss(self, projections_1, projections_2):
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (projections_1@projections_2.T)/self.temperature)

        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(contrastive_labels, tf.transpose(similarities))

        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True)
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True)
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        # Both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(contrastive_loss,self.encoder.trainable_weights + self.projection_head.trainable_weights)
        self.contrastive_optimizer.apply_gradients(
            zip(gradients, self.encoder.trainable_weights + self.projection_head.trainable_weights))
        self.contrastive_loss_tracker.update_state(contrastive_loss)


        with tf.GradientTape() as tape:
            features = self.encoder(labeled_images, training=False)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data
        features = self.encoder(labeled_images, training=False)

        return {m.name: m.result() for m in self.metrics[2:]}