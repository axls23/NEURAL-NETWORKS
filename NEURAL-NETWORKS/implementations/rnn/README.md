
## RNN: RECURRING NEURAL NETWORK

**RECURRING NEURAL NETWORK** is a sequence based model which is unlike other models dependent on multi-dimensions data type.

**inputs and outputs of rnn model** are sequentially dependent such as text, timeseries, biological signal data.

### Lets understand how this data is represented in brief

1.  **time series**: In a **time series**, values on successive discrete time stamps are closely related to one another.
    If we try to use values of these time stamps as individual features, one might lose their key information. For which we tend to use a **unidimensional sequence input model such as RNN**.

2.  **text data**: One can often represent **text data** in the form of a `bag of words [BOW]` format, but such representation may have limited scope of semantic representation. A sequenced ordered arrangement may provide a better processing quality. In day-to-day scenarios, **RNN** are mostly applied in **NATURAL LANGUAGE PROCESSING [NLP]**.

Data can be generalized into sequences with `real valued {timeseries}` and `symbolic{ text based }` nature.

 ### RNN  Structure ###

## RNN and Timestamps: Sequence Position Correspondence

RNNs exhibit a **one-to-one correspondence** between the **layers of the network** and a **specific position in a sequence**.

This position is known as a **timestamp**. Therefore, instead of using a variable number of inputs in a single input layer of a network, an RNN:

*   Contains a **variable number of layers**.
*   Each layer has a **single corresponding input timestamp**.

This representation allows each input to have its own interaction with hidden layers at different timeframes.


**Important Note:** RNN models contain the **same set of parameters** across each iteration or timestamp in the sequence. This parameter sharing across time is a key characteristic of RNNs, enabling them to process sequences of varying lengths while maintaining model efficiency.
Each  temporal  block can take in  or output  both single atributed and multi atribuite data points  


A fundamental characteristic of standard RNNs is that they utilize the **same set of parameters** (weights and biases) across each iteration or timestamp throughout the sequence.

**Benefits of Parameter Sharing:**

*   **Efficiency:** Significantly reduces the number of parameters the model needs to learn compared to having separate parameters for each timestamp. This makes training more feasible, especially for long sequences.

*   **Generalization:**  Allows the model to generalize learned patterns across different positions in the sequence. For example, if the RNN learns to recognize a pattern at the beginning of a sentence, it can apply the same knowledge to recognize similar patterns later in the sentence or in other sentences.  It learns features that are relevant *regardless* of position in the sequence.

limitation 
While parameter sharing is powerful, simple RNNs can struggle with capturing very long-range dependencies due to issues like vanishing gradients during training. this  limitation led to the development of more advanced RNN architectures like LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units), which incorporate mechanisms to better handle long-term dependencies while still benefiting from the core idea of recurrent processing and parameter sharing across time.


#  working of Basic RNN Implementation 

This document provides **easy-to-understand explanations** of the `BasicRNN` model, implemented in Python using TensorFlow and Keras.  It shows a simple Recurrent Neural Network (RNN), explaining its **structure**, how it **processes data**, and how we can **visualize** its work.

## Code Overview: Simple RNN Building Blocks

The Python code defines a `BasicRNN` model.  It's built using two main parts:

*   **`SimpleRNN` Layer:**  This is the **brain of the RNN**. It processes input data step-by-step and remembers past information using a **"hidden state" (memory)**.
*   **`Dense` Output Layer:** This is a standard **output layer**. It takes the RNN's **final memory** and turns it into the **model's prediction**.

The `main` function shows how to use this model:

1.  **Create the `BasicRNN` Model:** Set up the RNN with its basic structure.
2.  **Make Example Input Data:** Create sample data that looks like sequence data (like text or time series).
3.  **Run Data Through the Model (Forward Pass):** Feed the sample data into the RNN to get an output.
4.  **Check Output Shapes:** Look at the shape of the output to understand what the model produces.
5.  **Prepare Model for Training (Compile):**  Set up how the model will learn, even though we don't train it in this example.
6.  **See Model Structure (Summarize):** Get a summary of the model's layers and parameters.
7.  **See RNN in Action (Visualize):** Use a plot to see the input, output, and memory of the RNN for one example.

## `BasicRNN` Class: Understanding the RNN's Parts

### Setting Up the RNN

This part of the code sets up the RNN's basic structure.  It uses two main settings:

*   **`hidden_size` (Conceptual Memory):**  This is like the **size of the RNN's memory** at each step.
    *   **Larger `hidden_size`**:  More complex patterns can be remembered.

*   **`output_size` (Task Output):** This is the **size of the model's final prediction**. It depends on what you want to do.
    *   **Classification**:  `output_size` is usually the **number of categories**.
    *   **Regression**: `output_size` might be `1` (for one number prediction) or more.

**Layers Inside `__init__`:**

*   **`self.rnn = SimpleRNN(...)`: The RNN Memory Layer**
    *   `units=self.hidden_size`: Sets the **memory size**.
    *   `return_sequences=False`:  **Important: `False` means get only the final memory.** Useful for tasks where you want **one output for the whole sequence** (like analyzing sentence sentiment). If `True`, it would give output at each step (for tasks like translation).
    *   `return_state=True`: **Important: `True` means also give back the RNN's final memory.** Lets us see the "memory" after processing the input.
    *   `name='rnn_layer'`: Just a label for easier understanding.

*   **`self.dense_output = Dense(output_size, name='output_layer')`:  Outputting the Result**
    *   This layer takes the **RNN's final memory** and converts it into the **model's prediction**.

### How Data Flows Through the RNN

The `call` method explains how the RNN processes input data.

*   **`inputs` (tf.Tensor): Sequence Input Data:**  The input data is a batch of sequences. Shape: `(batch_size, sequence_length, input_size)`.
    *   `batch_size`: How many sequences processed at once.
    *   `sequence_length`: Length of each sequence.
    *   `input_size`:  Features per step in the sequence.

*   **`initial_state` (optional tf.Tensor): Starting Memory:**  You can set a starting memory for the RNN.
    *   `None` (default):  RNN starts with **zero memory**.
    *   You can give a specific starting memory, but less common for basic RNNs in this code.

**Steps Inside `call`:**

1.  **`rnn_output, state = self.rnn(inputs, initial_state=initial_state)`:  RNN Processing and Memory Update.** Input `inputs` goes into the `SimpleRNN` layer.
    *   **Step-by-Step Processing:** RNN processes the sequence **one step at a time**.  It uses the current input and **memory from the last step** to create a new memory and output. This is how RNNs handle sequences.
    *   `rnn_output`:  **Final memory** after processing the whole sequence. Shape: `(batch_size, hidden_size)`. This memory is a summary of the input sequence.
    *   `state`:  Also the **final memory**, same as `rnn_output`. Just returned separately for easy access.

2.  **`output = self.dense_output(rnn_output)`:  Making the Prediction.** The **RNN's final memory** is passed to the `Dense` layer.
    *   `output`:  The **model's final prediction**. Shape: `(batch_size, output_size)`.

3.  **`return output, state`:**  The method returns the **prediction** and the **final memory**.

## `main()` Function: Putting the RNN to Work

The `main()` function shows how to use the `BasicRNN` model.

*   **Set Hyperparameters:** Define sizes for input, memory, output, sequence length, and batch size.  These decide the data shape and model size.

*   **Create the Model:**
    ```python
    inputs = Input(shape=(seq_length, input_size))
    model = BasicRNN(hidden_size, output_size)
    ```
    This creates an instance of the `BasicRNN` model.

*   **Generate Example Input:**
    ```python
    example_input = tf.random.normal([batch_size, seq_length, input_size])
    ```
    Creates random data to mimic sequence data for testing.

*   **Forward Pass and Shape Check:**
    ```python
    output, state = model(example_input)
    print(...) # Print shapes of 'output' and 'state'
    ```
    Runs the data through the model and checks the output shapes:
    *   `Output shape`: Should be a batch of predictions.
    *   `State shape`: Should be a batch of final memories.

*   **Compile the Model (for Training):**
    ```python
    model.compile(...)
    ```
    Prepares the model for learning:
    *   `optimizer`:  Sets how the model learns (Adam is popular). `learning_rate` controls learning speed.
    *   `loss`: Sets what the model tries to minimize (binary cross-entropy for binary tasks). **Important:** Choose the right loss for your task.

*   **Build and Summarize Model:**
    ```python
    model.build(input_shape=(None, seq_length, input_size))
    model.summary()
    ```
    *   `model.build`:  Sets up the model to handle different batch sizes.
    *   `model.summary()`: Shows the model's layers, shapes, and parameters.

*   **Visualize with `plot_rnn_working`:**
    ```python
    plot_rnn_working(example_input, output, state)
    ```
    Creates plots to show: Input Sequence, RNN Output, and RNN Memory for one example.

## Seeing the RNN Work Visually

This function makes plots to help understand the RNN. It plots:

1.  **Input Sequence :** Shows the input data sequence for one example.
2.  **RNN Output :** Shows the model's prediction for that sequence.
3.  **RNN State :** Shows the RNN's internal memory after processing the sequence.

**Purpose of Plots:**  Helps you see how the RNN processes input, makes an output, and how it uses its memory.

**What to Expect:**

*   **Output and State Shapes Printed:**  You'll see the shapes of the model's output.
*   **Model Summary Printed:** You'll see the model's structure in text form.
*   **Visualization Plot:** A window will pop up with plots of the input, output, and RNN memory.

**Experimenting**

To understand better, try changing:

*   **Hyperparameters:**  Change `hidden_size`, `output_size`, `seq_length` and see how the model changes.
*   **`return_sequences=True`:** Change `return_sequences` in `SimpleRNN` to `True`. Rerun and see how the output changes.
*   **Loss Functions and Optimizers:**  Try different settings in `model.compile(...)`.

This code is a basic RNN example.  Experimenting with it will help you learn the core ideas of Recurrent Neural Networks!



