'''
toy neural network using sigmoid activation function and derivative for back propagation
'''

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
    learning_rate = 0.07
    bias = 1
    hidden_layer = []
    output_layer = []

    def __init__(self, num_inputs = 2, num_outputs = 2, num_hidden_perceptrons = 2):
        # initialize random weight
        self.num_hidden_perceptrons = num_hidden_perceptrons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        '''

        # initialize weights with random values between -1 and 1, these weights will be later adjusted through backpropagation.
        # the weights are saved as a matrix as matrix maths can be used conveniently to perform neural network operations for forward facing and backpropagation steps
        # dimension of the weights will depend on number of inputs, hidden nodes, hidden layers and outputs
        # this toy NN has only one hidden layer, the number of perceptrons in the hidden layer can be configured when instantiating the neural network
        # weights_hi - weights from input to hidden layer, if there are 2 input and 3 hidden perceptrons then the dimension of the matrix will be 3x2
        # each weight is represented as weight value of h1 to i1 - w11 from input 1 to hidden 1, w12 from input 2 to hidden 1 and these two weights will form the first row of the weights_hi matrix
        # from each input to the hidden perceptron there will be two such weights - w21, w22 and w31, w32 and they will be the 2nd and 3rd row of the weights_hi matrix

        # similary there is one more weight weights_oh for the weights of the connections from hidden to output perceptrons, the individual subscript will be from output n to hidden n
        # so that they end up in the correct position in the matrix

        weights from input1 to hidden (subscript in opposite direction)
        i1   --w11--      h1     o1
             --w21--      h2     o2
             --w31--      h3

        weights from input 2 hidden
        i2   --w12--      h1     o1
             --w22--      h2     o2
             --w32--      h3

        calculate hidden perceptron values by using dot product
        | w11 w12 | | i1 |  => |  w11 * i1 + w12 * i2    |  = | h1 |
        | w21 w22 | | i2 |                                    | h2 |
        | w31 w32 |

        '''

        self.weights_hi = np.random.uniform(-1, 1, (self.num_hidden_perceptrons, self.num_inputs))
        self.weights_oh = np.random.uniform(-1, 1, (self.num_outputs , self.num_hidden_perceptrons))
        print("Inital weights input to hidden: \n" , self.weights_hi, self.weights_hi.shape)
        print("Inital weights hidden to output: \n", self.weights_oh, self.weights_oh.shape)
        print("----")

    def guess(self,inputs):
        # matrix multiplication of transposed weights and sum
        #  3,2 weights , 1,2 input , add bias , sum of all result elements
        print("Adjusted weights input to hidden: \n" , self.weights_hi, self.weights_hi.shape)
        print("Adjusted weights hidden to output: \n", self.weights_oh, self.weights_oh.shape)
        self.hidden_layer = np.add(np.dot(self.weights_hi, inputs),self.bias)
        # print("hidden layer \n", self.hidden_layer, self.hidden_layer.shape)
        self.hidden_layer = self.sigmoid(self.hidden_layer)
        # print("sigmod hidden layer \n", self.hidden_layer, self.hidden_layer.shape)
        self.output_result = np.add(np.dot(self.weights_oh,self.hidden_layer), self.bias)
        # print("output layer \n", self.output_result, self.output_result.shape)
        return (self.sigmoid(self.output_result))
        # return np.array([  self.activation(output_result[0,0],"sigmoid"), self.activation(output_result[1,0],"sigmoid")])
        # return self.activation(np.sum(np.add(np.dot(self.weights, inputs),self.bias)), "sign")

    ''' sigmoid activation function  for forward facing network'''
    def sigmoid (self,x):
        return 1 / (1 + np.exp(-x))

    ''' derivate of sigmoid activatation function for backpropagation'''
    def dsigmoid(self,y):
        return y * (1-y)

    def train(self,inputs,target):
        print("--------------- new iteration ------------")
        guess = self.guess(inputs)
        print("guess result\n", guess, guess.shape, "expected target", target)

        # delta weight = learning rate * error * gradient * hidden transpose

        # output error = target - guess and multiply by learning rate
        # learning rate will made the adjustments smaller steps, otherwise neural network can never find the optimum weight values as it will overshoot the bottom (optimum)
        output_error =  (target - guess)
        print("output_error\n",output_error, output_error.shape)

        gradient_guess = self.dsigmoid(guess)
        gradient = np.multiply(output_error,gradient_guess)
        gradient = gradient * self.learning_rate
        print("after apply gradient output_error\n", gradient, gradient.shape)

        # matrix multiplication with the transpose of the weight matrix as backpropagation is processing in the opposite direction, weight matrix is transposed
        hidden_error = np.dot(self.weights_oh.transpose(),gradient)
        print("hidden_error\n",hidden_error, hidden_error.shape )

        gradient_hidden = self.dsigmoid(self.hidden_layer)
        gradient_hidden = self.learning_rate * gradient_hidden
        hidden_error = np.multiply(hidden_error, gradient_hidden)

        # loop through the columns as each column is multiplied with 1 input
        for i in range(self.weights_oh.shape[1]):
            #pay attention the correct weights are adjusted, adjust the looping subscript accordingly
            print("i self.weights_oh[i], hidden_error[i]",i, self.weights_oh[:,i], hidden_error[i])
            self.weights_oh[:,i] = np.add(self.weights_oh[:,i], hidden_error[i])
        print("updated weights: \n", self.weights_oh, self.weights_oh.shape)

        input_error = np.dot(self.weights_hi.transpose(),hidden_error)
        print("input_error\n",input_error, input_error.shape )

        for i in range(self.weights_hi.shape[1]):
            #pay attention the correct weights are adjusted
            print("i self.weights_hi[i], hidden_error[i]",i, self.weights_hi[:,i], input_error[i])
            self.weights_hi[:,i] = np.add(self.weights_hi[:,i], input_error[i])
        print("updated weights: \n", self.weights_hi, self.weights_hi.shape)


# end Perceptron



def main():
    #initialize perceptron
    p1 = Perceptron(2,2, 2)

    # points = []
    inputs = np.array([[3],[5]])
    print("inputs \n",inputs,inputs.shape)
    targets = np.array([[1],[0]])
    print("targets\n",targets,targets.shape)

    # results_b = []
    # results_b = p1.guess(inputs)
    # print("Before training guess... \n",results_b)

    print("----- training ---")
    for i in range(20000):
        p1.train(inputs, targets)

    # results_a = []
    # results_a = p1.guess(inputs)
    # print("After training guess... \n",results_a)



#main program
if __name__ == "__main__":
    main()


class Point:
    x = 0
    y = 0
    label = 0
    color = 0
    width = 0
    height = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x=np.random.randint(low=0,high=width,size=1)[0]
        self.y=np.random.randint(low=0,high=height,size=1)[0]
        if (self.x > self.y): # will work okay for a linear pattern like this
        # if(np.random.randint(low=0, high=2, size=1)[0] == 0): ## simple perceptron model will not work well for such a random pattern
            self.label = 1
            self.color = 'black'
        else:
            self.label = -1
            self.color = 'blue'

    @staticmethod
    def draw(points, results_a,results_b,width,height):
        #green outline to indicate guess matched with target
        plt.rcParams["figure.figsize"] = [7,7]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.title.set_text('Guess Before Training')
        l = mlines.Line2D([0, width], [0, height])
        ax1.add_line(l)
        for i in range(len(points)):
            # print(points[i].x,points[i].y,points[i].label, results_b[i])
            if (points[i].label == results_b[i]):
                ax1.plot(points[i].x,points[i].y,'ks',markerfacecolor = points[i].color,ms=5,markeredgecolor='green')
            else:
                ax1.plot(points[i].x, points[i].y, 'ks', markerfacecolor=points[i].color, ms=5,markeredgecolor='red')
        ax2 = fig.add_subplot(122)
        ax2.title.set_text('Guess After Training')
        l1 = mlines.Line2D([0, width], [0, height])
        ax2.add_line(l1)
        for i in range(len(points)):
            # print(points[i].x, points[i].y, points[i].label, results_a[i])
            if (points[i].label == results_a[i]):
                ax2.plot(points[i].x, points[i].y, 'ks', markerfacecolor=points[i].color, ms=5, markeredgecolor='green')
            else:
                ax2.plot(points[i].x, points[i].y, 'ks', markerfacecolor=points[i].color, ms=5, markeredgecolor='red')

        plt.show()
#end Point

    # results_a = []
    # width = 400
    # height = 400
    # #initialize n number of random points with a label
    # for i in range(500):
    #     points.append(Point(width,height))
    #
    # #make an intial guess using the random weighted perceptron
    # for point in points:
    #     inputs = np.array([[point.x],[point.y]])
    #     results_b.append(p1.guess(inputs))
    #
    # #train the perceptron, all data is used to train
    # for point in points:
    #     inputs = np.array([[point.x], [point.y]])
    #     print("----------next record---------------")
    #     print("inputs ", inputs, "target ", point.label)
    #     p1.train(inputs, point.label)
    #
    # #make guess with the trained perceptron
    # for point in points:
    #     inputs = np.array([[point.x], [point.y]])
    #     results_a.append(p1.guess(inputs))
    #
    # #plot the classification
    # Point.draw(points,results_a,results_b,width,height)