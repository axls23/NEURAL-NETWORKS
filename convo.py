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
class LiveTrainingPlot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epochs = []
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch)
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        clear_output(wait=True)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.accuracy, label='Training Accuracy')
        plt.plot(self.epochs, self.val_accuracy, label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.loss, label='Training Loss')
        plt.plot(self.epochs, self.val_loss, label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
class LiveTrainingPlotlive(tf.keras.callbacks.Callback):
    def __init__(self):
        self.epochs = []
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.show()
        self.fig.canvas.draw()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(self.epochs, self.accuracy, label='Training Accuracy')
        self.ax1.plot(self.epochs, self.val_accuracy, label='Validation Accuracy')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.legend()
        self.ax1.grid(True)

        self.ax2.plot(self.epochs, self.loss, label='Training Loss')
        self.ax2.plot(self.epochs, self.val_loss, label='Validation Loss')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Loss')
        self.ax2.legend()
        self.ax2.grid(True)

        self.fig.canvas.draw()
        plt.pause(0.001)

def my_model():
    INPUTs=keras.Input(shape=(28,28,1))
    x=layers.Conv2D(32,3, padding="valid")(INPUTs)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(128,3,activation='relu', use_bias=True)(x)
    x=layers.Dropout(0.14)(x)
    x= layers.BatchNormalization()(x)
    x=layers.MaxPooling2D()(x)
    x= layers.Conv2D(256, (3, 3), activation= keras .activations.softplus,padding='same')(x)
    x=layers.BatchNormalization()(x)
    
    
    x=layers.Flatten()(x)
    x=layers.Dense(64,activation='relu')(x)
    outputs=layers.Dense(10)(x)
    model=keras.Model(INPUTs,outputs)
    return model



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
optimizer= keras.optimizers.Adadelta(learning_rate=0.0025),
metrics=['accuracy']
)  

model.fit(x_train,y_train, validation_data=(x_test,y_test),validation_batch_size=32,batch_size=64,
          epochs=10,verbose=2, callbacks=[LiveTrainingPlotlive()])
model.evaluate(x_test,y_test,batch_size=64,verbose=2)
'''metrics to  analyze
Commonly Used Metrics:
1. Accuracy:

command:keras.metrics.Accuracy()

usecase:Measures the proportion of correctly predicted instances out of the total instances.

2. Binary Accuracy:

command:keras.metrics.BinaryAccuracy()

usecase:Used for binary classification problems, measuring the accuracy of predictions.

3. Categorical Accuracy:

command:keras.metrics.CategoricalAccuracy()

usecase:Used for multi-class classification problems where labels are one-hot encoded.

4. Sparse Categorical Accuracy:

command:keras.metrics.SparseCategoricalAccuracy()

usecase:Similar to categorical accuracy but used when labels are integers instead of one-hot encoded.

5. Precision:

command:keras.metrics.Precision()

usecase:Measures the ratio of true positive predictions to the total predicted positives.

6. Recall:

command:keras.metrics.Recall()

usecase:Measures the ratio of true positive predictions to the total actual positives.

7. F1 Score:

command:keras.metrics.F1Score()

usecase:The harmonic mean of precision and recall, providing a balance between the two.

8. AUC (Area Under the Curve):

command:keras.metrics.AUC()

usecase:Measures the area under the ROC curve, useful for binary classification problems.

9. Mean Squared Error (MSE):

command:keras.metrics.MeanSquaredError()

usecase:Measures the average of the squares of the errors, commonly used for regression tasks.

10. Mean Absolute Error (MAE):

command:keras.metrics.MeanAbsoluteError()

usecase:Measures the average of the absolute errors, also used for regression tasks.

11. Mean Absolute Percentage Error (MAPE):

command:keras.metrics.MeanAbsolutePercentageError()

usecase:Measures the average absolute percentage error between predicted and actual values.



'''

print(model.summary)


