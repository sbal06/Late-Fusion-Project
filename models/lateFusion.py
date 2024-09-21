# architecture of Late Fusion Models
# You need to run the individual sub-models, and access the predicted outputs and prediction probabilities for this late fusion model.
import numpy as np


total_predictions = 451 # size of test set
BALANCE_POSITIVE = 2683/4511
BALANCE_NEUTRAL = 470/4511
BALANCE_NEGATIVE = 1358/4511
# Detection Rate Approach
class detectionRateApproach():
    def set_weights_text(self, trupos_pos_text, truneu_neu_text, truneg_neg_text):
        self.weights = []
        DR_positive_text = trupos_pos_text / total_predictions
        DR_neutral_text = truneu_neu_text / total_predictions
        DR_negative_text = truneg_neg_text / total_predictions
       
        # depends on the dataset
        W_pos_text = (1 - DR_positive_text) + (BALANCE_POSITIVE)
        W_neu_text = (1 - DR_neutral_text) + (BALANCE_NEUTRAL)
        W_neg_text = (1 - DR_negative_text) + (BALANCE_NEGATIVE)
        sum_weights = W_pos_text + W_neu_text + W_neg_text
       
        W_pos_normalized_text = W_pos_text / sum_weights
        W_neu_normalized_text = W_neu_text / sum_weights
        W_neg_normalized_text = W_neg_text / sum_weights
       
        self.weights.append(W_neg_normalized_text)    
        self.weights.append(W_neu_normalized_text)
        self.weights.append(W_pos_normalized_text)
       
        return self.weights
   
   
    def set_weights_image(self, trupos_pos_image, trupos_neu_image, trupos_neg_image):
        DR_positive_image = trupos_pos_image / total_predictions
        DR_neutral_image = trupos_neu_image / total_predictions
        DR_negative_image = trupos_neg_image / total_predictions
       
        # depends on the dataset
        W_pos_image = (1 - DR_positive_image) + (2683/4511)
        W_neu_image = (1 - DR_neutral_image) + (470/4511)
        W_neg_image = (1 - DR_negative_image) + (1358/4511)
        sum_weights = W_pos_image + W_neu_image + W_neg_image
       
        W_pos_normalized_image = W_pos_image / sum_weights
        W_neu_normalized_image = W_neu_image / sum_weights
        W_neg_normalized_image  = W_neg_image / sum_weights
       
        self.weights.append(W_neg_normalized_image)    
        self.weights.append(W_neu_normalized_image)
        self.weights.append(W_pos_normalized_image)
       
        return self.weights  
   
    def weighted_average(self, probabilities):
        text_output_probabilities = []
        image_output_probabilities = []
       
        for i in range(6):
            if (i < 3):
                text_output_probabilities.append(self.weights[i] * probabilities[0][:, i])
            else:
                image_output_probabilities.append(self.weights[i] * probabilities[1][:, i-3])
           
        return text_output_probabilities, image_output_probabilities
   
    def combine(self, text_output_probabilities, image_output_probabilities):
        text_image_probabilities = []
        for text, image in zip(text_output_probabilities, image_output_probabilities):
             text_image_probabilities.append((text + image) / 2)
             
        return text_image_probabilities
   
    def predictions(self, text_image_probabilities):
        predictions = []
        for element in range(text_image_probabilities[0].shape[0]):
            probs = []
            for index in range(text_image_probabilities.shape[0]):
                probs.append(text_image_probabilities[index][element])
               
            predictions.append(np.argmax(np.asarray(probs)))
           
        return predictions
           
       
       
# Minimizing Loss Approach
class minimizingLossApproach():
    def retrieve(self, image_history, text_history):
        best_epoch_image = np.argmin(image_history['val_loss'])
        best_epoch_text = np.argmin(text_history['val_loss'])
       
        # weights based on the validation accuracies
        validation_accuracy_image = image_history['val_acc'][best_epoch_image]
        validation_accuracy_text = text_history['val_acc'][best_epoch_text]
       
        return validation_accuracy_image, validation_accuracy_text
   
    def setWeights(self, val_accuracy_image, val_accuracy_text, inverse):
        self.weights = []
        total_accuracy = val_accuracy_image + val_accuracy_text
       
        self.weights.append(val_accuracy_image / total_accuracy)
        self.weights.append(val_accuracy_text / total_accuracy)
       
        if inverse == False:
            inverse_weights_array = []
            inverse_weights = [1/weight for weight in self.weights]
            for element in inverse_weights:
                inverse_weights_array.append(element / np.sum(inverse_weights))
           
            self.weights = inverse_weights_array
           
        return self.weights
   
   
    def returnSentiment(self, text_probability, image_probability):
        predictions = np.asarray([text_probability, image_probability])
        weighted_probability = [self.weights[i] * predictions[i] for i in range(len(predictions))]
        weighted_probability = np.asarray(weighted_probability)


        prediction_probabilities = np.add(weighted_probability[0], weighted_probability[1])
        final_sentiments = [np.argmax(row) for row in prediction_probabilities]
       
        return final_sentiments
