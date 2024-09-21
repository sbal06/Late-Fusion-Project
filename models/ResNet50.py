# This file contains the architecture of the ResNet50 and the contrastive learning network
import numpy as np
import tensorflow as tf
import pickle

# define variables
NUM_CLASSES = 3
BATCH_SIZE = 16
INPUT_SHAPE = (224, 224, 3)

class ResNet50(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.augment1 = tf.keras.layers.Rescaling(1.0/255)
            self.augment2 = tf.keras.layers.RandomFlip('horizontal')
            self.augment3 = tf.keras.layers.RandomRotation(0.02)
            self.augment4 = tf.keras.layers.RandomTranslation(height_factor = 0.1, width_factor = 0.1)
            self.augment5 =  tf.keras.layers.RandomZoom(height_factor = 0.1, width_factor = 0.1)
            self.ResNet50 = tf.keras.applications.ResNet50(input_shape = INPUT_SHAPE, include_top = False, weights = 'imagenet')
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(activation = tf.keras.activations.relu, kernel_regularizer = tf.keras.regularizers.l2(0.01))
            self.dense2 = tf.keras.layers.Dense(activation = tf.keras.activation.softmax)
            
            
        def call(self, inputs, trainable):
             x = self.augment1(x)
             x = self.augment2(x)
             x = self.augment3(x)
             x = self.augment4(x)
             x = self.augment5(x)
             x = self.ResNet50(x)
             x = self.flatten(x)
             x = self.dense1(x)
             outputs = self.dense2(x)
             return outputs
           
           
            
       
      
       
                 
       
    
        
        
       
    

