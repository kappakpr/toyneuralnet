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
        #self.weights = np.random.uniform(-1,1,(num_hidden_perceptrons,num_inputs)).transpose()
        self.weights = np.random.uniform(-1, 1, (num_hidden_perceptrons, num_inputs))
        print("Inital weights: \n" , self.weights, self.weights.shape)
        # print("----")
        # print(self.weights.transpose(),self.weights.transpose().shape)
        print("----")
        # print(self.weights[0])
        # print(self.weights[1])
        # print(len(self.weights))

    def guess(self,inputs):
        # matrix multiplication of transposed weights and sum
        # 1,2 input , 3,2 weights transpose , sum of all result elements
        # return self.activation(np.sum(np.dot(inputs,self.weights.transpose())),"sign")
        # print("guess ", inputs, inputs.shape)
        # return self.activation(np.sum(np.dot(inputs, self.weights)), "sign")
        # return self.activation(np.sum(np.dot(self.weights,inputs)), "sign")

        return self.activation(np.sum(np.add(np.dot(self.weights, inputs),self.bias)), "sign")

    def activation(self,x,f):
        # print("dot output to activation ", x, f)
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
        # for i in range(len(self.weights)):
        for i in range(self.weights.shape[1]):
            #pay attention the correct weights are adjusted
            print("i self.weights[i], werr[i]",i, self.weights[:,i], werr[i])
            self.weights[:,i] = np.add(self.weights[:,i], werr[i])
            # print(np.add(self.weights[i,], werr[i]))
            # print(self.weights[i], werr[0,i])
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
        # print(self.x, self.y, self.label,self.color)

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
    # inputs =np.array([-1,0.5])
    # target=1
    # print("inputs ", inputs, "target ",target)
    p1 = Perceptron(2,3) #(2,6);
    # result=p1.guess(inputs)
    # print("Guess: ", result)
    # result=p1.train(inputs,target)

    points = []
    inputs = np.array([])
    results_b = []
    results_a = []
    width = 400
    height = 400
    for i in range(500):
        points.append(Point(width,height))

    for point in points:
        inputs = np.array([[point.x],[point.y]])
        results_b.append(p1.guess(inputs))

    for point in points:
        # inputs = np.array([point.x,point.y])
        inputs = np.array([[point.x], [point.y]])
        print("----------next record---------------")
        print("inputs ", inputs, "target ", point.label)
        p1.train(inputs, point.label)

    for point in points:
        # inputs = np.array([point.x,point.y])
        inputs = np.array([[point.x], [point.y]])
        results_a.append(p1.guess(inputs))

    Point.draw(points,results_a,results_b,width,height)

if __name__ == "__main__":
    main()