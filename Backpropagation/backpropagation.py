# Authors: Dimitrios Gagatsis // Christos Kaparakis

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function / Activation function
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
def dsigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))


# Define the Inputs
inputs = np.diag(np.ones(8, dtype=int)).tolist()

# we define bias and weights in such a way that we have a hidden layer consisting of 3 nodes and output layer of 8 nodes

# Bias
bias_inp = np.random.randn(3).tolist()
bias_hidden = np.random.randn(8).tolist()

# Weights
weights_l1_l2 = np.random.randn(8, 3)
weights_l2_l3 = np.random.randn(3, 8)

# Define the learning rate and the lambda value = weight decay parameter
# we can change these
# we got the best results after testing for different values of learning_rate and lambda_value

learning_rate = 2
lambda_value = 0.00001


# Activation of the hidden and output layer

def feed_forward(i, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden):
    hl = sigmoid(np.dot(i, weights_l1_l2) + bias_inp).tolist()
    ol = sigmoid(np.dot(hl, weights_l2_l3) + bias_hidden).tolist()
    return hl, ol


# By running the " feed_forward(inputs, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden) " we can see the
# first activation of hidden and output layer.


# find the deltas of the hidden and output layer
def deltas(hl, ol, inputs, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden):
    # inputs need to be passed one by one

    fz = dsigmoid(np.dot(hl, weights_l2_l3) + bias_hidden).tolist()
    delta = [-1 * (y - a) * f for y, a, f in zip(inputs, ol, fz)]
    fzh = dsigmoid(np.dot(inputs, weights_l1_l2) + bias_inp)
    deltah = np.dot(np.array(weights_l1_l2).transpose(), np.array(delta))
    deltah = [x * f for x, f in zip(deltah, fzh)]
    # deltah: hidden layer & delta: output layer

    pdW = np.dot(np.array(delta).reshape(8, 1), np.array(hl).reshape(1, 3))
    pdB = delta
    pdWH = np.dot(np.array(deltah).reshape(3, 1), np.array(inputs).reshape(1, 8))
    pdBH = deltah
    # partial derivatives of weights and bias

    return pdW.transpose(), pdB, pdWH.transpose(), pdBH


def updates(inputs, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden):
    dW = np.zeros((3, 8))
    dB = np.zeros((1, 8))
    dWH = np.zeros((8, 3))
    dBH = np.zeros((1, 3))

    for i in inputs:
        hl, ol = feed_forward(i, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden)
        pdW, pdB, pdWH, pdBH = deltas(hl, ol, i, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden)

        dW = dW + np.array(pdW)
        dB = dB + np.array(pdB)
        dWH = dWH + np.array(pdWH)
        dBH = dBH + np.array(pdBH)

    # update weights
    weights_l1_l2_new = weights_l1_l2 - learning_rate * (
            (1 / len(inputs)) * dWH + lambda_value * np.array(weights_l1_l2))
    weights_l2_l3_new = weights_l2_l3 - learning_rate * (
            (1 / len(inputs)) * dW + lambda_value * np.array(weights_l2_l3))
    # update bias
    bias_inp = np.array(bias_inp) - learning_rate * ((1 / len(inputs)) * np.array(dBH[0]))
    bias_hidden = np.array(bias_hidden) - learning_rate * ((1 / len(inputs)) * np.array(dB[0]))

    return weights_l1_l2_new, weights_l2_l3_new, bias_inp, bias_hidden


def backpropagation(inputs, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden, learning_rate, lambda_value):
    count = 0
    correct_count = 0
    mse = []
    while count < 20000:
        avg_error = 0
        count += 1
        weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden = updates(inputs, weights_l1_l2, weights_l2_l3, bias_inp,
                                                                      bias_hidden)

        for i in inputs:
            hl, ol = feed_forward(i, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden)
            error = ((np.array(ol) - np.array(i)) ** 2) / 2
            avg_error = avg_error + error
            avg_error = sum(avg_error) / len(inputs)
        print('Average error:', avg_error)
        print('count', count)
        if avg_error < 0.005:
            print(count, 'iterations are needed for convergence')
            break
        mse.append(avg_error)

    for i in inputs:
        hl, ol = feed_forward(i, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden)
        print("Input layer:", i, "  Output layer", [round(x) for x in ol])

    return mse, weights_l1_l2, weights_l2_l3


m, w1, w2 = backpropagation(inputs, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden, learning_rate, lambda_value)

# Plot performance for different learning rates
# MSEplot = []
# x = np.linspace(1, 20000, 20000, endpoint=True)
# rates = [0.1, 0.25, 0.5, 1, 2, 5, 10]
# for learning_rate in rates:
#     MSE,w1,w2 = backpropagation(inputs, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden)
#     MSEplot.append(MSE)
#
# for i in range(7):
#     plt.plot(MSEplot[i], label=str(rates[i]))
# plt.legend(loc="upper right")
# plt.title("Convergence for different learning rates")
# plt.ylabel('Error')
# plt.xlabel("Iterations")
# plt.show()

# Plot performance for different lambdas
# MSEplot = []
# x = np.linspace(1, 50000, 50000, endpoint=True)
# ls = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
# for lambda_value in ls:
#     MSE, w1, w2 = backpropagation(inputs, weights_l1_l2, weights_l2_l3, bias_inp, bias_hidden)
#     MSEplot.append(MSE)
#
# for i in range(6):
#     plt.plot(MSEplot[i], label=str(ls[i]))
# plt.legend(loc="upper right")
# plt.title("Convergence for different lambdas")
# plt.ylabel('Error')
# plt.xlabel("Iterations")
# plt.show()
