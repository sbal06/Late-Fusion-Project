from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

NUM_CLASSES = 3

MODEL = 'cardiffnlp/twitter-roberta-base-sentiment' # model we are using
tokenizer_used = AutoTokenizer.from_pretrained(MODEL)

Twitter_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL, num_labels = 3)

def tokenization(data):
    encodings = tokenizer_used(data.tolist(), truncation = True, padding = True, return_tensors = 'np')
    encodings = dict(encodings)
    return encodings

# add optional layer

class twitterModel(tf.keras.Model):
    def __init__(self, base_model):
        self.base_model = base_model
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)
        
    def call(self, inputs):
        outputs = self.base_model(inputs)
        outputs = self.dropout(outputs)
        outputs = self.dense(outputs)
        return outputs

    
        