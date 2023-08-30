import  numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import os

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
        target=np.array([target]).transpose()
        # print(inputs)
        output = self.guess(inputs)
        # delta weight = learning rate * error * gradient * weights transpose
        output_error = target - output
        # print("output_error ",target.shape , output.shape, output_error.shape)
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

        return self.weights_hi, self.weights_oh, self.bias
    def predict(self,inputs):

        op_bias = 0.1
        op_weights_hi = np.loadtxt('op_weights_hi.out')
        op_weights_oh = np.loadtxt('op_weights_oh.out')

        hidden_layer = np.add(np.dot(op_weights_hi, inputs),op_bias)
        hidden_layer = self.sigmoid(hidden_layer)

        output_result = np.add(np.dot(op_weights_oh,hidden_layer), op_bias)
        output_result = self.sigmoid(output_result)
        return output_result


def main():

    samp_size = 2000
    cat_class = 0
    num_classes = 8 #change depending on number of image classes
    img_size = 784
    img_size_xy = 28
    max_color = 255
    cat_class_name = []


    inputs = np.empty((0,img_size))
    targets = np.empty([0,num_classes])
    train_inputs = np.empty((0, img_size))
    train_targets = np.empty([0,num_classes])
    test_inputs = np.empty((0, img_size))
    test_targets = np.empty([0,num_classes])

    for x in os.listdir():
        if x.endswith(".npy"):
            print("loading file.. ", x)
            all_data = np.load(x)
            x_data = all_data[np.random.choice(all_data.shape[0], samp_size, replace=False), :]
            x_cat_cls = np.zeros([samp_size,num_classes])
            x_cat_cls[:,cat_class] = 1
            inputs=np.append(inputs,x_data,axis = 0)
            targets=np.append(targets,x_cat_cls.reshape((samp_size,num_classes)),axis=0)
            cat_class_name = np.append(cat_class_name, [np.char.replace(x,'.npy','')], 0)
            cat_class += 1

    train_size = int(.999 * inputs.shape[0])
    print("train_size ",train_size)
    train_idxs=np.random.choice(inputs.shape[0], train_size, replace=False)
    train_inputs = inputs[train_idxs,:]
    train_targets = targets[train_idxs]
    test_inputs = np.delete(inputs,train_idxs,axis=0)
    test_targets = np.delete(targets, train_idxs, axis=0)

    p1 = Perceptron(img_size,num_classes, img_size * 4)

    print("----- training ---", train_inputs.shape[0])
    for i in range(train_inputs.shape[0]):
        if (i % 50 == 0):
            print("--------------- new iteration ------------ ", train_inputs.shape[0] - i)
        op_weights_hi, op_weights_oh, op_bias = p1.train(train_inputs[i]/max_color, train_targets[i])

    # np.savetxt('op_bias.out',op_bias)
    np.savetxt('op_weights_hi_4.out',op_weights_hi)
    np.savetxt('op_weights_oh_4.out',op_weights_oh)

    # print("After training guesses......")
    # for i in range(test_inputs.shape[0]):
    #     results_a = []
    #     results_a = p1.guess(test_inputs[i]/max_color)
    #     print("Input ",i," target ",np.argmax(test_targets[i]), " ", test_targets[i]," After training guess... ",np.argmax(results_a)," ", results_a)
        # x = test_inputs[i].reshape((img_size_xy, img_size_xy))
        # plt.imshow(x, cmap='gray')
        # plt.show()

if __name__ == "__main__":
    main()