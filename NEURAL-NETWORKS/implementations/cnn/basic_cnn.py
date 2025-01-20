import os 
import keras.activations
import keras.losses
import keras.optimizers
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import mnist
from keras import Sequential
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
physical_devices=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
        

#initialization of data 
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0
# Define a custom callback class for live training plot
class LiveTrainingPlot(tf.keras.callbacks.Callback):
    # Initialize lists to store metrics at the beginning of training
    def on_train_begin(self, logs={}):
        self.epochs = []  # List to store epoch numbers
        self.accuracy = []  # List to store training accuracy
        self.val_accuracy = []  # List to store validation accuracy
        self.loss = []  # List to store training loss
        self.val_loss = []  # List to store validation loss

    # Update metrics and plot them at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch)  # Append current epoch number
        self.accuracy.append(logs.get('accuracy'))  # Append current training accuracy
        self.val_accuracy.append(logs.get('val_accuracy'))  # Append current validation accuracy
        self.loss.append(logs.get('loss'))  # Append current training loss
        self.val_loss.append(logs.get('val_loss'))  # Append current validation loss

        clear_output(wait=True)  # Clear previous output
        plt.figure(figsize=(12, 8))  # Create a new figure with specified size
        
        # Plot accuracy metrics
        plt.subplot(1, 2, 1)  # Create subplot for accuracy
        plt.plot(self.epochs, self.accuracy, label='Training Accuracy')  # Plot training accuracy
        plt.plot(self.epochs, self.val_accuracy, label='Validation Accuracy')  # Plot validation accuracy
        plt.title('Accuracy')  # Set title for accuracy plot
        plt.xlabel('Epoch')  # Set x-axis label
        plt.ylabel('Accuracy')  # Set y-axis label
        plt.legend()  # Show legend

        # Plot loss metrics
        plt.subplot(1, 2, 2)  # Create subplot for loss
        plt.plot(self.epochs, self.loss, label='Training Loss')  # Plot training loss
        plt.plot(self.epochs, self.val_loss, label='Validation Loss')  # Plot validation loss
        plt.title('Loss')  # Set title for loss plot
        plt.xlabel('Epoch')  # Set x-axis label
        plt.ylabel('Loss')  # Set y-axis label
        plt.legend()  # Show legend

        plt.show()  # Display the plots

# Define another custom callback class for live training plot with real-time updates
class LiveTrainingPlotlive(tf.keras.callbacks.Callback):
    # Initialize lists and figure for plotting
    def __init__(self):
        self.epochs = []  # List to store epoch numbers
        self.accuracy = []  # List to store training accuracy
        self.val_accuracy = []  # List to store validation accuracy
        self.loss = []  # List to store training loss
        self.val_loss = []  # List to store validation loss
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))  # Create subplots for accuracy and loss
        self.fig.show()  # Show the figure
        self.fig.canvas.draw()  # Draw the canvas

    # Update metrics and plot them at the end of each epoch
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)  # Append current epoch number
        self.accuracy.append(logs.get('accuracy'))  # Append current training accuracy
        self.val_accuracy.append(logs.get('val_accuracy'))  # Append current validation accuracy
        self.loss.append(logs.get('loss'))  # Append current training loss
        self.val_loss.append(logs.get('val_loss'))  # Append current validation loss

        self.ax1.clear()  # Clear previous accuracy plot
        self.ax2.clear()  # Clear previous loss plot

        # Plot accuracy metrics
        self.ax1.plot(self.epochs, self.accuracy, label='Training Accuracy')  # Plot training accuracy
        self.ax1.plot(self.epochs, self.val_accuracy, label='Validation Accuracy')  # Plot validation accuracy
        self.ax1.set_xlabel('Epochs')  # Set x-axis label for accuracy plot
        self.ax1.set_ylabel('Accuracy')  # Set y-axis label for accuracy plot
        self.ax1.legend()  # Show legend for accuracy plot
        self.ax1.grid(True)  # Enable grid for accuracy plot

        # Plot loss metrics
        self.ax2.plot(self.epochs, self.loss, label='Training Loss')  # Plot training loss
        self.ax2.plot(self.epochs, self.val_loss, label='Validation Loss')  # Plot validation loss
        self.ax2.set_xlabel('Epochs')  # Set x-axis label for loss plot
        self.ax2.set_ylabel('Loss')  # Set y-axis label for loss plot
        self.ax2.legend()  # Show legend for loss plot
        self.ax2.grid(True)  # Enable grid for loss plot

        self.fig.canvas.draw()  # Redraw the canvas with updated plots
        plt.pause(0.001)  # Pause to update the plot in real-time

def my_model():
    # Input layer: Expects 28x28 grayscale images (MNIST format)
    # Shape (28,28,1) where 1 represents single channel for grayscale
    INPUTs = keras.Input(shape=(28,28,1))

    # First Convolutional Block
    # Conv2D: 32 filters, 3x3 kernel, no padding (valid)
    # Followed by BatchNorm for stability and ReLU activation
    # MaxPooling reduces spatial dimensions by half
    x = layers.Conv2D(32,3, padding="valid")(INPUTs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)

    # Second Convolutional Block
    # Increased filters to 64, same kernel size
    # Added Dropout (14%) to prevent overfitting
    x = layers.Conv2D(64,3,activation='relu', use_bias=True)(x)
    x = layers.Dropout(0.14)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Third Convolutional Block
    # Further increased filters to 128
    # Uses softplus activation instead of ReLU
    # 'same' padding maintains spatial dimensions
    
    x = layers.Conv2D(128, (3, 3), activation=keras.activations.softplus, padding='same')(x)

    x = layers.BatchNormalization()(x)
    
    # Flatten the 3D feature maps to 1D feature vector
    x = layers.Flatten()(x)
    
    # Dense layers for classification
    # First dense layer: 64 neurons with ReLU
    # Output layer: 10 neurons (one per digit for MNIST)
    x = layers.Dense(64,activation='relu')(x)
    outputs = layers.Dense(10)(x)

    # Create and return the model
    model = keras.Model(INPUTs,outputs)
    return model

"""
Architecture Summary:
1. This is a  basic CNN architecture for image classification
2. Uses progressively increasing filter sizes (32 -> 128 -> 256)
3. Implements modern best practices:
   - Batch Normalization for training stability
   - Dropout for regularization ("Regularization adds a penalty to the loss function,
                               discouraging large weights that can lead to overfitting".)
   - Multiple conv layers for hierarchical feature learning
4. Suitable for MNIST digit classification (28x28 input, 10 classes output
5. Uses the Functional API style of Keras (more flexible than Sequential)
"""

#lets build sequential model for convolutional neural network
model= my_model()
    
        # Adjusting the model to prevent dimension issues
# model=Sequential(
#     [
#         keras.Input(shape=(32,32,3)),
#         layers.Conv2D(32, (3, 3), padding='valid', activation='relu'), 
#         layers.MaxPooling2D(pool_size=(2, 2)),  
#         layers.Conv2D(64, (3, 3), activation='relu',use_bias=True),  
#         layers.MaxPooling2D(pool_size=(2, 2)),  
#         layers.Conv2D(128, (3, 3), activation='relu'),
#         layers.Conv2D(256, (3, 3), activation=keras.activations.tanh),  
#         layers.MaxPooling2D(pool_size=(2, 2)),   
         
#         layers.Flatten(),
#         layers.Dense(128, activation=keras.activations.relu),
#         layers.Dense(10)
#     ])
        
#compiling the model
model.compile(
loss=keras.losses.sparse_categorical_crossentropy,
optimizer= keras.optimizers.Adadelta(learning_rate=0.01),
metrics=['accuracy']
)  

model.fit(x_train,y_train, validation_data=(x_test,y_test),validation_batch_size=32,batch_size=64,
          epochs=29,verbose=2, callbacks=[LiveTrainingPlotlive()])
model.evaluate(x_test,y_test,batch_size=64,verbose=2)
print(model.summary)
