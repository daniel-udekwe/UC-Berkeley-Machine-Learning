import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"        
        # Compute the linear product of the input tensor with the learned weights
        linear_output = self.run(x)
        
        # Convert the (1,1) tensor to a scalar value
        linear_output_scalar = nn.as_scalar(linear_output)
        
        # Apply a binary classification decision rule based on the sign of the linear output
        if linear_output_scalar >= 0:
            classification_label = 1
        else:
            classification_label = -1
        
        return classification_label
        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
             
        # Set batch size to 1 and initialize change flag to True
        batch_size = 1
        change_flag = True
        
        # Loop until convergence
        while change_flag:
            # Reset change flag to False at the beginning of each iteration
            change_flag = False
            
            # Iterate over the dataset one example at a time
            for x, y in dataset.iterate_once(batch_size):
                # Compute prediction for input x
                prediction = self.get_prediction(x)
                
                # Update weights if prediction does not match true label
                if prediction != nn.as_scalar(y):
                    # Compute constant vector
                    constant = nn.Constant(nn.as_scalar(y) * x.data)
                    
                    # Update weights with constant vector scaled by learning rate 1
                    self.w.update(constant, 1)
                    
                    # Set change flag to True to indicate weight update
                    change_flag = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
                
        # Set learning rate to 0.01
        self.lr = 0.01
        
        # Initialize weights and biases for three-layer neural network
        self.w1 = nn.Parameter(1, 128)   # Input layer -> First hidden layer weights
        self.b1 = nn.Parameter(1, 128)   # First hidden layer biases
        self.w2 = nn.Parameter(128, 64)  # First hidden layer -> Second hidden layer weights
        self.b2 = nn.Parameter(1, 64)    # Second hidden layer biases
        self.w3 = nn.Parameter(64, 1)    # Second hidden layer -> Output layer weights
        self.b3 = nn.Parameter(1, 1)     # Output layer bias
        
        # Collect all learnable parameters in a list
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
                
        # Define the first hidden layer using a linear transformation followed by ReLU activation
        first_layer_output = nn.Linear(x, self.w1)   # Perform linear transformation on input
        first_layer_output = nn.AddBias(first_layer_output, self.b1)   # Add bias term to the output
        first_layer_output = nn.ReLU(first_layer_output)   # Apply ReLU activation function
        
        # Define the second hidden layer using a similar approach as the first layer
        second_layer_output = nn.Linear(first_layer_output, self.w2)   # Perform linear transformation on first layer output
        second_layer_output = nn.AddBias(second_layer_output, self.b2)   # Add bias term to the output
        second_layer_output = nn.ReLU(second_layer_output)   # Apply ReLU activation function
        
        # Define the output layer using a linear transformation followed by a bias term
        output_layer = nn.Linear(second_layer_output, self.w3)   # Perform linear transformation on second layer output
        output_layer = nn.AddBias(output_layer, self.b3)   # Add bias term to the output
        
        # Return the output of the neural network
        return output_layer


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
       
        # Compute the predicted values for the given input using the neural network
        predicted_values = self.run(x)
        
        # Compute the squared loss between the predicted values and the actual values, and return it
        loss = nn.SquareLoss(predicted_values, y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
       
        # Set the batch size for training
        batch_size = 50
        
        # Initialize the loss to an infinitely large value to guarantee that the loop runs at least once
        loss = float('inf')
        
        # Iterate until the loss drops below a certain threshold
        while loss >= .015:
        
            # Iterate over the dataset, processing one batch of 50 samples at a time
            for x, y in dataset.iterate_once(batch_size):
        
                # Compute the loss for the current batch
                loss = self.get_loss(x, y)
        
                # Print the current value of the loss
                print(nn.as_scalar(loss))
        
                # Compute the gradients of the loss with respect to the neural network parameters
                grads = nn.gradients(loss, self.params)
        
                # Convert the loss tensor to a scalar value
                loss = nn.as_scalar(loss)
        
                # Update the neural network parameters using stochastic gradient descent
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"    
        
        # set the learning rate
        self.lr = 0.1
        
        # initialize the weights and biases for each layer
        # using nn.Parameter() and store them in variables
        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, 64)
        self.b3 = nn.Parameter(1, 64)
        self.w4 = nn.Parameter(64, 10)
        self.b4 = nn.Parameter(1, 10)
        
        # store all the parameters in a list
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]

        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # Create first hidden layer
        hidden_layer_1 = nn.Linear(x, self.w1)
        hidden_layer_1 = nn.AddBias(hidden_layer_1, self.b1)
        hidden_layer_1 = nn.ReLU(hidden_layer_1)
        
        # Create second hidden layer
        hidden_layer_2 = nn.Linear(hidden_layer_1, self.w2)
        hidden_layer_2 = nn.AddBias(hidden_layer_2, self.b2)
        hidden_layer_2 = nn.ReLU(hidden_layer_2)
        
        # Create third hidden layer
        hidden_layer_3 = nn.Linear(hidden_layer_2, self.w3)
        hidden_layer_3 = nn.AddBias(hidden_layer_3, self.b3)
        hidden_layer_3 = nn.ReLU(hidden_layer_3)
        
        # Create output layer
        output_layer = nn.Linear(hidden_layer_3, self.w4)
        output_layer = nn.AddBias(output_layer, self.b4)
        
        return output_layer


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        
        # The given code is computing the softmax loss of the predicted class probabilities for a given input tensor x and target tensor y.
        # It returns the softmax loss.
        y_pred = self.run(x)
        loss = nn.SoftmaxLoss(y_pred, y)
        return loss
      
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
     
        # Set the batch size for training.
        batch_size = 100
        
        # Initialize the best validation accuracy to zero.
        best_valid_acc = 0.0
        
        # Train the model until the best validation accuracy is at least 0.98.
        while best_valid_acc < 0.98:
        
            # Iterate over the dataset in batches of the specified size.
            for x, y in dataset.iterate_once(batch_size):
                
                # Calculate the loss and gradients for the current batch.
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                
                # Update the model parameters using the calculated gradients and learning rate.
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.lr)
                    
            # Calculate the validation accuracy for the current model.
            valid_acc = dataset.get_validation_accuracy()
            
            # Update the best validation accuracy if the current validation accuracy is higher.
            best_valid_acc = max(best_valid_acc, valid_acc)
