"""This file contains utility functions for activation functions, loading data,
plotting and saving models

Yathartha Tuladhar
03/15/2019
"""
import numpy as np

def RELU(x):
    # If x>0, returns x, else returns zero. Implements a binary mask
    return x*(x>0)

# To generate mini-batches for training
# https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class SigmoidCrossEntropy:
    def __init__(self):
        pass

    def forward(self,x, z):

        """This function combines the sigmoid activation function,
        and calculated the CrossEntropy loss.

        x = logits, z = labels
        
        *** The reason this is put into a single layer is because it has a simple gradient form. ***

        The Sigmoid is used so that the prediction is within [0,1].
        CrossEntropy loss is used.
        https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        """
        # Pass through Sigmoid activation function

        prediction = 1.0/(1.0 + np.exp(-x))  # can throw a runtime error when exp(-x) is a very large number

        # Calulate loss. See formula. This version below takes care of stability and overflow.
        # It is derived from the actual
        loss = np.zeros(np.shape(prediction))  # since we're dealing with a batch
        for i in range(x.size):
            loss[i] = max(x[i], 0) - x[i] * z[i] + np.log(1 + np.exp(-abs(x[i])))
        return loss, prediction #--ask yathu is this also a matrix--#

    def backward(self, g_output, hidden_output):
        """ Returns loss back from the output layer so that L1 can calculate gradients """
        d_W2 = hidden_output.T.dot(g_output)
        return d_W2

def extrude(w, batch_size):
    ig = np.copy(w)
    for i in range(batch_size-1):
        w = np.concatenate((w, ig), axis=0)

    return w
