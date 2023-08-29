import  numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

class Perceptron:
    weights_hi = []
    weights_oh = []
    num_hidden_layers = 1
    num_hidden_perceptrons = 2
    num_inputs = 2
    num_outputs = 2
    learning_rate = 0.1
    bias = 0.1
    hidden_layer = []
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
        self.weights_oh = np.random.uniform(-1, 1, (self.num_outputs , self.num_hidden_perceptrons))
    def guess(self,inputs):
        self.hidden_layer = np.add(np.dot(self.weights_hi, inputs),self.bias)
        self.hidden_layer = self.sigmoid(self.hidden_layer)

        self.output_result = np.add(np.dot(self.weights_oh,self.hidden_layer), self.bias)
        self.output_result = self.sigmoid(self.output_result)
        return self.output_result

    def train(self,inputs,target):
        inputs=np.array([inputs]).transpose()
        output = self.guess(inputs)
        # delta weight = learning rate * error * gradient * weights transpose
        output_error = target - output
        ds_output = self.dsigmoid(output)
        gradient_ho = np.multiply(output_error,ds_output)
        gradient_ho = gradient_ho * self.learning_rate

        weight_ho_delta = np.dot(gradient_ho,self.hidden_layer.transpose())
        for i in range(self.weights_oh.shape[1]):
            self.weights_oh[:,i] = np.add(self.weights_oh[:,i], weight_ho_delta[:,i])

        ds_hidden_layer = self.dsigmoid(self.hidden_layer)
        hidden_error = np.dot(self.weights_oh.transpose(), output_error)
        gradient_ih = np.multiply(hidden_error,ds_hidden_layer)
        gradient_ih = gradient_ih * self.learning_rate

        weight_hi_delta = np.dot(gradient_ih,inputs.transpose())
        for i in range(self.weights_hi.shape[1]):
            self.weights_hi[:,i] = np.add(self.weights_hi[:,i], weight_hi_delta[:,i])

def main():

    #initialize perceptron
    p1 = Perceptron(2,1, 6)

    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    targets = np.array([0,1,1,0])

    print("Before training guesses......")
    for i in range(4):
        results_a = []
        results_a = p1.guess(inputs[i])
        print("Input ",inputs[i]," target ",targets[i]," Before training guess... ",results_a)

    print("----- training ---")
    for i in range(20000):
        if (i % 5000 == 0):
            print("--------------- new iteration ------------ ", i)
        subscript = np.random.randint(0,4)
        p1.train(inputs[subscript], targets[subscript])

    print("After training guesses......")
    for i in range(4):
        results_a = []
        results_a = p1.guess(inputs[i])
        print("Input ",inputs[i]," target ",targets[i]," After training guess... ",results_a)

if __name__ == "__main__":
    main()