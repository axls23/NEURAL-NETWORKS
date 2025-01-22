import tensorflow as tf
from keras import Model
from keras.layers import Dense, SimpleRNN, Input
import matplotlib.pyplot as plt

class BasicRNN(Model):
    """
    A basic Recurrent Neural Network (RNN) model.

    This model uses a SimpleRNN layer followed by a Dense output layer.
    It's designed for sequence input and produces a single output vector
    and the final hidden state.
    """
    def __init__(self, hidden_size, output_size):
        """
        Initializes the BasicRNN model.

        Args:
            hidden_size (int): The number of units in the SimpleRNN layer (dimensionality of the hidden state).
            output_size (int): The size of the output vector (number of output classes or regression targets).
        """
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the RNN layer
        self.rnn = SimpleRNN(
            units=self.hidden_size,
            return_sequences=False,  # Return only the last output for sequence-to-vector task
            return_state=True,       # Return the final hidden state
            name='rnn_layer'         # Naming layers is good practice
        )

        # Define the output layer (Dense layer)
        self.dense_output = Dense(output_size, name='output_layer')

    def call(self, inputs, initial_state=None):
        """
        Performs a forward pass through the BasicRNN model.

        Args:
            Input tensor of shape (batch_size, sequence_length, input_size).
            
            initial_state: Initial hidden state for the RNN.
                If None, the RNN initializes its state to zero.
                Shape should be : (batch_size, hidden_size) . 
                while Defaults are None.

        Returns:
            tuple: A tuple containing:
                    output: Output tensor from the Dense layer, shape `(batch_size, output_size)
                    state  : The final hidden state of the RNN, shape `(batch_size, hidden_size)
        """
        # Forward pass through RNN layer
        # rnn_output shape: (batch_size, hidden_size) - last hidden state
        # state shape: (batch_size, hidden_size) - same as rnn_output when return_sequences=False
        rnn_output, state = self.rnn(inputs, initial_state=initial_state)

        # Pass the RNN output through the fully connected (Dense) layer
        output = self.dense_output(rnn_output)

        return output, state

def main():
    """
    Main function to demonstrate the BasicRNN model.
    Sets up the model, performs a forward pass with example input,
    prints output shapes, compiles the model, and generates a summary.
    """
    input_size = 30      # Size of input features at each time step
    hidden_size = 30    # Number of hidden units in the RNN layer
    output_size = 21    # Size of the output vector
    seq_length = 64    # Length of the input sequence
    batch_size = 10     # Number of sequences in a batch

    # lets,Create Model using Functional API for explicit input shape
    # While not strictly necessary for this simple model, it's  termed good  practice 
    # to define input shape explicitly, especially forcomplex models.
    inputs = Input(shape=(seq_length, input_size))
    model = BasicRNN(hidden_size, output_size)

    # Create Example Input Data
    #    Generate random input data for demonstration purposes.
    example_input = tf.random.normal([batch_size, seq_length, input_size])

     #Forward Pass   
     #  Pass the example input through the model to get the output and final state.
    output, state = model(example_input)

    # Print Output and State Shapes
    print("--- Output and State Shapes ---")
    print(f"Output shape: {output.shape} - (batch_size, output_size)")
    print(f"State shape: {state.shape}   - (batch_size, hidden_size)")
    print("---")

   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Using a more common learning rate
        loss=tf.keras.losses.binary_crossentropy # Or another suitable loss function
    )

    model.build(input_shape=(None, seq_length, input_size)) # None allows for variable batch size
    print("--- Model Summary ---")
    model.summary()
    print("---")

    # Plotting Function (Corrected to plot once after forward pass)
    def plot_rnn_working(inputs, outputs, states):
        """
        Plots the input sequence, RNN output, and RNN state for a single example from the batch.

        Args:
            inputs (tf.Tensor): Input tensor (batch_size, sequence_length, input_size).
            outputs (tf.Tensor): Output tensor (batch_size, output_size).
            states (tf.Tensor): State tensor (batch_size, hidden_size).
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        # Plot the inputs for the first example in the batch
        axs[0].plot(inputs[0].numpy())
        axs[0].set_title('Input Sequence (Example 1)')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Input Value')
        '''
        
        '''
        # Plot the RNN outputs for the first example in the batch
        axs[1].plot(outputs[0].numpy())
        axs[1].set_title('RNN Output (Example 1)')
        axs[1].set_xlabel('Output Dimension Index') # x-label  reflects output vector
        
        axs[1].set_ylabel('Output Value')

        # Plot the RNN states for the first example in the batch
        axs[2].plot(states[0].numpy())
        axs[2].set_title('RNN State (Example 1)')
        axs[2].set_xlabel('State Dimension Index') #  x-label reflects state vector
        axs[2].set_ylabel('State Value')

        plt.tight_layout()
        plt.show()

    # Plot the working of the RNN (after forward pass)
    print("--- Plotting RNN Working ---")
    plot_rnn_working(example_input, output, state)
    print("--- Plotting Completed ---")



if __name__ == "__main__":
    main()
