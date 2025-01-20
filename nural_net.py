import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
from convo import LiveTrainingPlotlive
''' what is keras?
Keras: High-level API simplifying neural network creation, training, and evaluation.  
 keras include Various Model Types:
 Keras supports a wide range of neural network architectures, including:

Sequential models: Simple, linear stacks of layers.

Functional models: More complex models with multiple inputs and outputs, allowing for intricate connections between layers.

Model subclassing: Provides the most flexibility for creating custom models with complex control flow.

Key uses include rapid prototyping, ease of use, flexibility, and backend integration (TensorFlow, etc.).'''
from keras import layers
from keras.datasets import mnist
(x_train,ytrain),(x_test,y_test)= mnist.load_data()
#print('this is the x train shape',x_train.shape)
x_train= x_train.reshape(-1,784).astype('float32')/255.0# float32 is to reduce computation  while  we divide by 255 to  normalize  values from  range (0-255 to 0-1)
x_test= x_test.reshape(-1,784).astype('float32')/255.0
#

# lets create a basic model  of NURAL NETWORK
  # we will use sequential api  from keras ( its convineant but not flexible)
model = keras.Sequential([
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(128, activation= keras.activations.tanh,use_bias= True),
    layers.BatchNormalization(256,activation='sigmoid',
                 
                 ),
    layers.Dense(128,activation=keras.activations.relu),layers.MaxPooling2D(3,strides=12, padding='same'),
    layers.Dense(10)
    ])
''' what is sequential api?

The Sequential API is the simplest way to build neural networks in Keras. Key points:

1. Layer Organization:
   - Layers are arranged in a straight line (like a stack)
   - Each layer has exactly one input and one output
   - Data flows sequentially from first layer to last

2. Characteristics:
   - Simple to understand and use
   - Perfect for basic neural networks
   - Each layer automatically connects to the previous one
   - Limited to single-input, single-output flows

3. Structure:
   layers.Dense(521)  →  layers.Dense(256)  →  layers.Dense(10)
   (input layer)         (hidden layer)        (output layer)

4. Limitations:
   - Cannot handle multiple inputs or outputs
   - No support for layer branching
   - No residual connections or complex architectures

Use Sequential when:
- Your network is simple and linear
- You don't need complex layer connections
- Single input → Single output
'''
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adadelta(lr=1,),
    metrics=['accuracy']
)
'''model.compile specify network configuration like:-
    Loss function: Measures how well the model performs

    Optimizer: Defines how the model should update based on the loss

    Metrics: What to measure during training/testing'''
model.fit(
    x_train, 
     ytrain,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    verbose=2,
    callbacks=[LiveTrainingPlotlive()]
)
''' model.fit actually trains your model on data with:-
    
    Training data (x_train and y_train)
    
    Batch size: How many samples to process before updating

    Epochs: How many times to go through the entire dataset

    Validation data: Data to evaluate model during training
'''




