import numpy as np
import math

sin=np.vectorize(math.sin)
cos=np.vectorize(math.cos)
weight1=np.array([])
weight2=np.array([])

class NN:
    def __init__(self, x, y):
        global weight1, weight2

        self.x=x
        self.y=y
        self.output=np.zeros(y.shape)
        
        self.weights1=np.random.rand(self.x.shape[1], 4)
        weight1=self.weights1[:]
        
        self.weights2=np.random.rand(4, 1)
        weight2=self.weights2[:]

    def sigmoid(x, deriv=False):
        if deriv:
            return x*(1-x)
        return np.exp(x)/(1+np.exp(x))
    
    def feedforward(self):
        global weight1, weight2

        self.layer1=NN.sigmoid(np.dot(self.x, weight1))
        self.output=NN.sigmoid(np.dot(self.layer1, weight2))

    def backprop(self):
        global weight1, weight2

        d_weights2=np.dot(self.layer1.T, 2*(self.y-self.output)*NN.sigmoid(self.output, True))
        
        d_weights1 = np.dot(self.x.T,  (np.dot(2*(self.y - self.output)*NN.sigmoid(self.output, True), self.weights2.T)*NN.sigmoid(self.layer1, True)))

        self.weights1 += d_weights1
        weight1=self.weights1[:]
        
        self.weights2 += d_weights2
        weight2=self.weights2[:]

c=[[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
d=[[1], [0], [1], [1], [0]]

c=np.array(c)
d=np.array(d)

train=NN(c, d)

for i in range(1500):
    NN.feedforward(train)
    NN.backprop(train)

print(train.output)
