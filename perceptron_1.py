import  numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

class Perceptron:
    weights = []
    num_hidden_layers = 1
    num_hidden_perceptrons = 2
    num_inputs = 2
    learning_rate = 0.07
    bias = 1

    def __init__(self, num_inputs = 2,num_hidden_perceptrons = 2):
        # initialize random weight
        self.num_hidden_perceptrons = num_hidden_perceptrons
        self.num_inputs = num_inputs

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

        self.weights = np.random.uniform(-1, 1, (num_hidden_perceptrons, num_inputs))
        print("Inital weights: \n" , self.weights, self.weights.shape)
        print("----")

    def guess(self,inputs):
        # matrix multiplication of transposed weights and sum
        #  3,2 weights , 1,2 input , add bias , sum of all result elements
        return self.activation(np.sum(np.add(np.dot(self.weights, inputs),self.bias)), "sign")

    def activation(self,x,f):
        #if greater than negative return 1 else return -1
        if f == "sign":
            return np.sign(x) * 1
        elif f == "sigmoid":
            return 1 / (1 + np.exp(-x))

    def train(self,inputs,target):
        guess = self.guess(inputs)
        print("guess result", guess, "expected target", target)
        error = (target - guess) * self.learning_rate
        print("error",error)
        werr = np.dot(error,inputs)
        print("werr",werr, "len(self.weights) ", len(self.weights),self.weights.shape[1] )
        # loop through the columns as each column is multiplied with 1 input
        for i in range(self.weights.shape[1]):
            #pay attention the correct weights are adjusted
            print("i self.weights[i], werr[i]",i, self.weights[:,i], werr[i])
            self.weights[:,i] = np.add(self.weights[:,i], werr[i])
        print("updated weights: \n", self.weights, self.weights.shape)

# end Perceptron

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

def main():
    #initialize perceptron
    p1 = Perceptron(2,3)

    points = []
    inputs = np.array([])
    results_b = []
    results_a = []
    width = 400
    height = 400
    #initialize n number of random points with a label
    for i in range(500):
        points.append(Point(width,height))

    #make an intial guess using the random weighted perceptron
    for point in points:
        inputs = np.array([[point.x],[point.y]])
        results_b.append(p1.guess(inputs))

    #train the perceptron, all data is used to train
    for point in points:
        inputs = np.array([[point.x], [point.y]])
        print("----------next record---------------")
        print("inputs ", inputs, "target ", point.label)
        p1.train(inputs, point.label)

    #make guess with the trained perceptron
    for point in points:
        inputs = np.array([[point.x], [point.y]])
        results_a.append(p1.guess(inputs))

    #plot the classification
    Point.draw(points,results_a,results_b,width,height)

#main program
if __name__ == "__main__":
    main()