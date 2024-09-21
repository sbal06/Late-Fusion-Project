# ResNet50 for contrastive learning
# Split into different functions for training


import tensorflow as tf
import tensorflow_addons as tfa
import os
import sys

# keras version has to be under <3.0.0 or else tensorflow_addons throws an error


# add the absolute path before import package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


import models.ResNet50 as rn


data_augmentation = tf.keras.Sequential(
    [
     tf.keras.layers.Rescaling(1.0/255),
     tf.keras.layers.RandomFlip('horizontal'),
     #tf.keras.layers.RandomShear(x_factor = 0.3, y_factor = 0.3),
     tf.keras.layers.RandomRotation(0.02),
     tf.keras.layers.RandomTranslation(height_factor = 0.1, width_factor = 0.1),
     tf.keras.layers.RandomZoom(height_factor = 0.1, width_factor = 0.1)
    ]
)


# create encoder
def encoder():
    resNet50_sequence = tf.keras.applications.ResNet50(input_shape = rn.INPUT_SHAPE, include_top = False, weights = 'imagenet')
    inputs = tf.keras.Input(shape = rn.INPUT_SHAPE)
    augmented_data = data_augmentation(inputs)
    outputs = resNet50_sequence(augmented_data)
    encoder_model = tf.keras.Model(inputs = inputs, outputs = outputs)


    return encoder_model


def create_classifier(encoder_model):
    for layer in encoder_model.layers:
        layer.trainable = False
   
    inputs = tf.keras.Input(shape = rn.INPUT_SHAPE)
    inputs = tf.keras.layers.Flatten()(inputs)
    x = encoder_model(inputs)
    x = tf.keras.layers.Dense(256, activation = tf.keras.activations.relu, kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    outputs = tf.keras.layers.Dense(rn.NUM_CLASSES, actiation = tf.keras.activations.softmax)(x)
   
    full_model = tf.keras.Model(inputs = inputs, outputs = outputs)


    return full_model


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__() # or super(SupervisedContrastiveLoss, self).__init__()
    def __call__(self, labels, feature_vectors, sample_weight = None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(tf.matmul(feature_vectors_normalized, tf.transpose(feature_vectors_normalized)), self.temperature)
        logits = tf.convert_to_tensor(logits)


        # N Pair loss computed for batch
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
   
# project high level features into lower feature space
def add_projection_head(encoder):
    inputs = tf.keras.Inputs(shape = rn.INPUT_SHAPE)
    inputs = tf.keras.layers.Flatten()(inputs)
    outputs = encoder(inputs)
    outputs = tf.keras.layers.Dense(128, activation = tf.keras.activation.relu)(inputs)
    cl_model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'contrastive_loss_projection_head')
   
    return cl_model