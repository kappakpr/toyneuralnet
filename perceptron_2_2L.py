import  numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

class Perceptron:
    weights_hi = []
    weights_hh = []
    weights_oh = []
    num_hidden_layers = 1
    num_hidden_perceptrons = 2
    num_inputs = 2
    num_outputs = 2
    learning_rate = 0.1
    bias = 0.1
    hh1_layer = []
    hh2_layer = []
    output_layer = []
    def sigmoid (self,x):
        return 1 / (1 + np.exp(-x))
    def dsigmoid(self,y):
        # return y * (1-y)
        return np.multiply(y,(1 - y))
    def __init__(self, num_inputs = 2, num_outputs = 2, num_hidden_perceptrons = 2):
        # initialize random weight
        self.num_hidden_perceptrons = num_hidden_perceptrons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights_hi = np.random.uniform(-1, 1, (self.num_hidden_perceptrons, self.num_inputs))
        self.weights_hh = np.random.uniform(-1, 1, (self.num_hidden_perceptrons, self.num_hidden_perceptrons))
        self.weights_oh = np.random.uniform(-1, 1, (self.num_outputs , self.num_hidden_perceptrons))
    def guess(self,inputs):
        self.hh1_layer = np.add(np.dot(self.weights_hi, inputs), self.bias)
        self.hh1_layer = self.sigmoid(self.hh1_layer)

        self.hh2_layer = np.add(np.dot(self.weights_hh, self.hh1_layer), self.bias)
        self.hh2_layer = self.sigmoid(self.hh2_layer)

        self.output_result = np.add(np.dot(self.weights_oh, self.hh2_layer), self.bias)
        self.output_result = self.sigmoid(self.output_result)
        return self.output_result
    def train(self,inputs,target):
        inputs=np.array([inputs]).transpose()
        target=np.array([target]).transpose()
        output = self.guess(inputs)

        # delta weight = learning rate * error * gradient * weights transpose
        ds_output = self.dsigmoid(output)
        output_error = target - output
        gradient_ho = np.multiply(output_error,ds_output)
        gradient_ho = gradient_ho * self.learning_rate

        weight_ho_delta = np.dot(gradient_ho, self.hh2_layer.transpose())
        for i in range(self.weights_oh.shape[1]):
            self.weights_oh[:,i] = np.add(self.weights_oh[:,i], weight_ho_delta[:,i])

        ds_hh2_layer = self.dsigmoid(self.hh2_layer)
        hh2_error = np.dot(self.weights_oh.transpose(), output_error)
        gradient_hh = np.multiply(hh2_error,ds_hh2_layer)
        gradient_hh = gradient_hh * self.learning_rate

        weight_hh_delta = np.dot(gradient_hh,self.hh1_layer.transpose())
        for i in range(self.weights_hh.shape[1]):
            self.weights_hh[:,i] = np.add(self.weights_hh[:,i], weight_hh_delta[:,i])

        ds_hh1_layer = self.dsigmoid(self.hh1_layer)
        hh1_error = np.dot(self.weights_hh.transpose(), hh2_error)
        gradient_ih = np.multiply(hh1_error,ds_hh1_layer)
        gradient_ih = gradient_ih * self.learning_rate

        weight_hi_delta = np.dot(gradient_ih,inputs.transpose())
        for i in range(self.weights_hi.shape[1]):
            self.weights_hi[:,i] = np.add(self.weights_hi[:,i], weight_hi_delta[:,i])

def main():
    p1 = Perceptron(2,2, 2)

    inputs = np.array([1,0])
    targets = np.array([1,0])

    results_b = []
    results_b = p1.guess(inputs)
    print("before training guess... \n",results_b)

    for i in range(30000):
        if (i % 500 == 0):
            print("--------------- new iteration ------------ ", i)
        p1.train(inputs, targets)

    results_b = []
    results_b = p1.guess(inputs)
    print("After training guess... \n",results_b)

if __name__ == "__main__":
    main()