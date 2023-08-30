import  numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    def predict(self,inputs):

        op_bias = 0.1
        op_weights_hi = np.loadtxt('op_weights_hi_4.out')
        op_weights_oh = np.loadtxt('op_weights_oh_4.out')

        hidden_layer = np.add(np.dot(op_weights_hi, inputs),op_bias)
        hidden_layer = self.sigmoid(hidden_layer)

        output_result = np.add(np.dot(op_weights_oh,hidden_layer), op_bias)
        output_result = self.sigmoid(output_result)
        return output_result


def main():

    samp_size = 10
    cat_class = 0
    num_classes = 8
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
            all_data = np.load(x)
            x_data = all_data[np.random.choice(all_data.shape[0], samp_size, replace=False), :]
            x_cat_cls = np.zeros([samp_size,num_classes])
            x_cat_cls[:,cat_class] = 1
            inputs=np.append(inputs,x_data,axis = 0)
            targets=np.append(targets,x_cat_cls.reshape((samp_size,num_classes)),axis=0)
            cat_class_name = np.append(cat_class_name, [np.char.replace(x,'.npy','')], 0)
            cat_class += 1

    train_size = int(.1 * inputs.shape[0])
    print("train_size ",train_size)
    train_idxs=np.random.choice(inputs.shape[0], train_size, replace=False)
    # train_inputs = inputs[train_idxs,:]
    # train_targets = targets[train_idxs]
    test_inputs = np.delete(inputs,train_idxs,axis=0)
    test_targets = np.delete(targets, train_idxs, axis=0)

    p1 = Perceptron(img_size,num_classes, img_size * 4)

    target_class=[]
    predict_class=[]

    for i in range(test_inputs.shape[0]):
        results_a = []
        results_a = p1.predict(test_inputs[i]/max_color)
        target_class = np.append(target_class,[np.argmax(test_targets[i])])
        predict_class = np.append(predict_class,[np.argmax(results_a)])
        # print("Input ",i," target ",np.argmax(test_targets[i]), " ", test_targets[i]," After training guess... ",np.argmax(results_a)," ", results_a)
        print("Input ", i, " target ", np.argmax(test_targets[i]), " ", cat_class_name[np.argmax(test_targets[i])] ," guess... ",  np.argmax(results_a), " ", cat_class_name[np.argmax(results_a)], " 2nd best guess.. ", cat_class_name[np.argsort(results_a)[-2]])
        # if np.argmax(test_targets[i]) != np.argmax(results_a):
        #     x = test_inputs[i].reshape((img_size_xy, img_size_xy))
        #     plt.imshow(x, cmap='gray')
        #     plt.show()
    print(target_class)
    print(predict_class)
    cm=confusion_matrix(target_class, predict_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=cat_class_name)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    main()