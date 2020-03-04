import numpy as np
import matplotlib.pyplot as plt

np.random.seed(30)
X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float) 
Y = np.array(([0], [1], [1], [0]), dtype=float)

W1 = np.random.randn(2, 2)
b1 = np.random.randn(2,)
W2 = np.random.randn(2, 1)
b2 = np.random.randn(1,)

loss_values = []
iterations = []
accuracy = []

def forward(X,W1,b1,W2,b2):

    z1 = np.dot(X,W1) + b1
    a1 = np.maximum(0, z1)

    z2 = np.dot(a1,W2) + b2
    a2 = 1 / (1 + np.exp(-z2))
    return(z1, a1, a2) 


def loss(Y,a2):
    error = - (Y * np.log(a2) + (1-Y) * np.log(1-a2))
    #print(error)
    loss = error.mean()
    return loss

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == np.around(predicted[i], 2):
			correct += 1
	return correct / float(len(actual)) * 100.0

def backward_output(W2, b2, Y, a1, a2, l):
    dw = np.dot((a2 - Y).T, a1) 
    db = a2 - Y 

    W2 = W2 - (l * dw.T) 
    b2 = b2 - (l * db) 
    return W2, b2

  
def backward_hidden(W1, b1, z1, X, l):
    dw = np.zeros((4,2))
    db = np.zeros((4,2))

    for i in range(4):
        for j in range(2):
            if z1[i][j] <= 0:
                dw[i][j] = 0
            else:
                dw[i][j] = X[i][j]

    for i in range(4):
        for j in range(2):
            if z1[i][j] <= 0:
                db[i][j] = 0
            else:
                db[i][j] = 1

    W1 = W1 - (l * dw.mean())
    b1 = b1 - (l * db)
    return W1, b1



for i in range(10000):
    z1, a1, a2 = forward(X,W1,b1,W2,b2)
    loss_values.append(loss(Y, a2)) 
    accuracy.append(accuracy_metric(Y, np.round(a2)))
    iterations.append(i)
    W2, b2 = backward_output(W2, b2, Y, a1, a2, l=0.01)
    W1, b1 = backward_hidden(W1, b1, z1, X, l=0.01)
    
    
print("The predicted result is:") 
print(a2)

f1 = plt.figure(1)
plt.plot(iterations,loss_values)
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.show()


f2 = plt.figure(2)
plt.plot(iterations, accuracy)
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.show()



