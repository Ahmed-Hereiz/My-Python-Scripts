import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

        

class TrainingHistoryPlotter:
    """
    A class to plot the accuracy and loss of a Keras model training history.
    """

    def __init__(self, history):
        """
        Parameters:
        history (History): The history object returned by the `fit` method of a Keras model.
        """
        self.history = history

    def plot(self):
        """
        Plots the accuracy and loss of the model training history.
        """
        # Plot accuracy
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

        ax1.plot(self.history.history['binary_accuracy'],color='#e32b2b')
        ax1.plot(self.history.history['val_binary_accuracy'],color='#ab1313')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Val'], loc='upper left')

        # Plot loss
        ax2.plot(self.history.history['loss'],color='#e32b2b')
        ax2.plot(self.history.history['val_loss'],color='#ab1313')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Val'], loc='upper left')

        # Plot f1_score
        ax3.plot(self.history.history['f1_score'],color='#e32b2b')
        ax3.plot(self.history.history['val_f1_score'],color='#ab1313')
        ax3.set_title('Model f1_score')
        ax3.set_ylabel('f1_score')
        ax3.set_xlabel('Epoch')
        ax3.legend(['Train', 'Val'], loc='upper left')

        # Show the plot
        plt.show()
        