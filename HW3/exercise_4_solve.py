import numpy as np

obs = np.array([2, 2])
W1 = np.array([[0.1, 0.1], [0.1, 0.2], [0.2, 0.1]])
b1 = np.array([0.1, 0, 0.1])
W2 = np.array([[1, 2, 2], [1, 2, 1], [1, 1, 1]])
b2 = np.array([1, 1, 1])
alpha = 0.5

## Forward Propagation

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

z1 = W1 @ obs.T + b1
print("z[1]: ", z1)
x1 = sigmoid(z1)
print("x[1]: ", x1)

z2 = W2 @ x1.T + b2
print("z[2]: ", z2)
x2 = softmax(z2)
print("x[2]: ", x2)


## Backward Propagation

aux = np.array([[softmax(z2)[0]*(1-softmax(z2)[0]), softmax(z2)[0]*softmax(z2)[1], softmax(z2)[0]*softmax(z2)[2]],
                   [softmax(z2)[1]*softmax(z2)[0], softmax(z2)[1]*(1-softmax(z2)[1]), softmax(z2)[1]*softmax(z2)[2]],
                   [softmax(z2)[2]*softmax(z2)[0], softmax(z2)[2]*softmax(z2)[1], softmax(z2)[2]*(1-softmax(z2)[2])]])
print("Jacobian: ", aux)
delta2 = np.array([-1/x2[0], 0, 0]) @ aux
print("delta[2]: ", delta2)
delta1 = (W2.T @ delta2) * (sigmoid(z1)*(1-sigmoid(z1)))
print("delta[1]: ", delta1)


## Update Weights

W1_new = W1 - alpha * np.outer(delta1, obs)
print("New W[1]: ", W1_new)

W2_new = W2 - alpha * np.outer(delta2, x1)
print("New W[2]: ", W2_new)

b1_new = b1 - alpha * delta1
print("New b1: ", b1_new)

b2_new = b2 - alpha * delta2
print("New b2: ", b2_new)



