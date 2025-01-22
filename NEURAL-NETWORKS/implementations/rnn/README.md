```markdown
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
```

