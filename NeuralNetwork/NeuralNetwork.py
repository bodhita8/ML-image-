"""This code implements a Neural Network from scratch in Python

Yathartha Tuladhar
03/15/2019
"""

from __future__ import division
from __future__ import print_function
import numpy as np
from utils import RELU, SigmoidCrossEntropy, iterate_minibatches, extrude
import pickle as pkl
import matplotlib.pyplot as plt


class MLP:
    """This is a class for a two layered neural network"""
    def __init__(self, n_input, n_output, n_hidden):
        eps = 0.001     # In order to randomly initialize weight with zero mean
        self.LR = 0.005 # learning rate
        # Initialize weights and biases for all the layers
        self.L1_weights = np.random.uniform(low=0-eps, high=0+eps, size=(n_input,n_hidden))
        self.L1_biases = np.random.uniform(low=0-eps, high=0+eps, size=(1, n_hidden))

        self.L2_weights = np.random.uniform(low=0-eps, high=0+eps, size=(n_hidden,n_output))
        self.L2_biases = np.random.uniform(low=0-eps, high=0+eps, size=(1, n_output))
        self.SigmoidCE = SigmoidCrossEntropy()

    def train(self, x_batch, y_batch):
        # --- Forward Propagation of Layer 1--- #

        b1 = extrude(np.copy(self.L1_biases), batch_size=16)#--ask yathu abt this#
        b2 = extrude(np.copy(self.L2_biases), batch_size=16)

        # z = weight*input + bias
        z1 = x_batch.dot(self.L1_weights) + b1  #--b is added individually here--#
        # Pass z through the activation function a=f(z). This is the output of hidden-layer
        a1 = RELU(z1)#----what if use other activation functions--#

        # --- Forward Propagation of Layer 2--- #
        # z = weight*input + bias
        z2 = a1.dot(self.L2_weights) + b2
        # Activation for Layer2 will be sigmoid, which is implemented inside SigmoidCrossEntropy function
        #ask --why sigmoid in 2nd layer and relu in 1st layer-#
        # Now that we have passed it through Layer 1 and 2, we need to generate an output, and calculate loss
        # We will do this in the SigmoidCrossEntropy function, just to keep it clean
        loss, prediction = self.SigmoidCE.forward(z2, y_batch) #--ask yathu abt actual outputs of P and Loss--#

        avg_loss = sum(loss)

        # --- Forward Pass is done! Now do backward pass --- #
        # Gradient of output (a-y). "d" means derivative
        d_output = prediction - y_batch  #--what#

        d_L2_weights = self.SigmoidCE.backward(d_output, a1)
        d_L2_biases = d_output    #TODO: fix this???  ask
        # Output layer backpropagation done

        # Now, do Hidden-layer backpropagation
        # TODO: is this called loss for hidden layer?
        # As in loss = output_gradient*hidden_layer_weights
        loss_hidden = np.dot(d_output, self.L2_weights.T)  # RELU backprop
        loss_hidden[a1<=0] = 0

        d_L1_weights = np.dot(x_batch.T, loss_hidden)
        d_L1_biases = loss_hidden

        # Update weights and biases
        self.L2_weights = self.L2_weights - self.LR*d_L2_weights
        self.L2_biases = self.L2_biases - self.LR*np.reshape(np.mean(d_L2_biases, axis=0), (1, len(d_L2_biases[0])))

        self.L1_weights = self.L1_weights - self.LR*d_L1_weights
        self.L1_biases = self.L1_biases - self.LR*np.reshape(np.mean(d_L1_biases, axis=0), (1, len(d_L1_biases[0])))

        return avg_loss

    def evaluate(self, x_batch, y_batch):
        '''Do the same forward pass as during training
        It would have been cleaner to put the forward pass for the training and evaluation
        both into a common forward function
        '''

        # --- Forward Propagation of Layer 1--- #
        # z = weight*input + bias
        z1 = x_batch.dot(self.L1_weights) + self.L1_biases
        # Pass z through the activation function a=f(z). This is the output of hidden-layer
        a1 = RELU(z1)

        # --- Forward Propagation of Layer 2--- #
        # z = weight*input + bias
        z2 = a1.dot(self.L2_weights) + self.L2_biases
        # Activation for Layer2 will be sigmoid, which is implemented inside SigmoidCrossEntropy function

        # Now that we have passed it through Layer 1 and 2, we need to generate an output, and calculate loss
        # We will do this in the SigmoidCrossEntropy function, just to keep it clean
        loss, prediction = self.SigmoidCE.forward(z2, y_batch)

        avg_loss = sum(loss)

        diff = prediction - y_batch  # if prediction is same as labels diff will be zero
        is_correct = (np.abs(diff)) <= 0.49

        accuracy = np.mean(is_correct) * 100.0
        return accuracy, avg_loss

if __name__=="__main__":
    # Load CIFAR data
    #data = pkl.load(open('cifar_2class_py2.p', 'rb'))  # This was throwing error
    with open('cifar_2class_py2.p', 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # Training samples
    train_x = data['train_data']
    train_y = data['train_labels']
    # Tesing samplies
    test_x = data['test_data']
    test_y = data['test_labels']

    # Get dimensions
    num_examples, INPUT_DIMS = train_x.shape
    _, OUTPUT_DIMS = train_y.shape

    # PARAMETERS
    NUM_EPOCHS = 50
    NUM_BATCHES = 16
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.005
    
    # --- Start training --- #
    # Instantiate neural network (multi-layer perceptron)
    neural_network = MLP(INPUT_DIMS, OUTPUT_DIMS, HIDDEN_UNITS)
    neural_network.LR = LEARNING_RATE

    # Tracking
    loss_per_epoch = []
    train_accuracy_per_epoch = []
    test_accuracy_per_epoch = []

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        # Create batches of data
        for batch in iterate_minibatches(train_x, train_y, NUM_BATCHES, shuffle=True):
            avg_loss =0.0
            x_batch,y_batch = batch
            x_batch = x_batch/255.0

            avg_loss = neural_network.train(x_batch,y_batch)
            # Update total loss for epoch
            total_loss = total_loss + avg_loss

        #print("Epoch ="+str(epoch)+"    Epoch batch Loss="+str(total_loss))
        loss_per_epoch.append(total_loss)

        # Now, calculate train accuracy for the whole dataset
        train_accuracy, train_loss = neural_network.evaluate(train_x, train_y)
        train_accuracy_per_epoch.append(train_accuracy)
        #print("Train accuracy="+str(train_accuracy)+" Train loss="+str(train_loss))
        #
        # Now, calculate test accuracy for the whole dataset
        test_accuracy, test_loss = neural_network.evaluate(test_x, test_y)
        test_accuracy_per_epoch.append(test_accuracy)
        print("Epoch ="+str(epoch)+" Epoch batch Loss="+str(round(total_loss[0],2)) +
              "  Train accuracy="+str(round(train_accuracy,2))+" Train loss="+str(round(train_loss[0],2)) +
              "  Test accuracy="+str(round(test_accuracy,2)) + " Test loss=" + str(round(test_loss[0],2)))
        #print("\n")

    # plotting after all epochs are done
    plt.plot(loss_per_epoch)
    plt.title('Average Loss (' + "Ep:" + str(NUM_EPOCHS) + " Batches: "+str(NUM_BATCHES)+" H-units:"+str(HIDDEN_UNITS)+" LR:"+str(LEARNING_RATE))
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.show()

    plt.plot(train_accuracy_per_epoch)
    plt.title('Training Accuracy (' + "Ep:" + str(NUM_EPOCHS) + " Batches: "+str(NUM_BATCHES)+" H-units:"+str(HIDDEN_UNITS)+" LR:"+str(LEARNING_RATE))
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.show()

    plt.plot(test_accuracy_per_epoch)
    plt.title('Test Accuracy (' + "Ep:" + str(NUM_EPOCHS) + " Batches: " + str(NUM_BATCHES) + " H-units:" + str(
        HIDDEN_UNITS) + " LR:" + str(LEARNING_RATE))
    plt.xlabel('Epoch')
    plt.ylabel('Testing Accuracy')
    plt.show()

print("Finished Plotting")
