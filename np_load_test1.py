import  numpy as np
from matplotlib import pyplot as plt

x=np.load('full_numpy_bitmap_cat.npy')
# print(x.shape)
print(x[0].shape)

# print(x[0])

# x=x[1].reshape((28,28))
# plt.imshow(x, cmap='gray')
# plt.show()

# idx = np.random.randint(10, size=2)
# data1=np.random.choice(x, size=100, replace=False)
data1=x[np.random.choice(x.shape[0], 100, replace=False), :]
print(data1.shape)

x=data1[10].reshape((28,28))
plt.imshow(x, cmap='gray')
plt.show()



# inputs = np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
# print(inputs)
# print(inputs[1])
# print(inputs[1].transpose())
# # print(inputs[1][0])
# # print(inputs[1][1])
# x=np.array([[0,1]])
# print(x, type(x))
# print(x.transpose())