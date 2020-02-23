import numpy as np
# might need to install 
import torch

# parameters
N = 2000
M = 100
H = 75

# generate data
np.random.seed(181)
W1 = np.random.random((H, M))
b1 = np.random.random(H)
W2 = np.random.random(H)
b2 = np.random.random(1)

X = np.random.random((N, M))
y = np.random.randint(0,2,size=N).astype('float')

# torch copies of data
tW1 = torch.tensor(W1, requires_grad=True)
tb1 = torch.tensor(b1, requires_grad=True)
tW2 = torch.tensor(W2, requires_grad=True)
tb2 = torch.tensor(b2, requires_grad=True)

tX = torch.tensor(X)
ty = torch.tensor(y)

# CAREFUL: if you run the code below w/o running the code above,
# the gradients will accumulate in the grad variables. Rerun the code above
# to reset

# torch implementation
def tforward(X):
  z1 = (torch.mm(tX, tW1.T) + tb1).sigmoid()
  X = (torch.mv(z1, tW2) + tb2).sigmoid()
  return X

tyhat = tforward(tX)
L = (ty * tyhat.log()) + (1-ty) * (1 - tyhat).log()
# the magic of autograd!
L.sum().backward()

# the gradients will be stored in the following variables
grads_truth = [tW1.grad.numpy(), tb1.grad.numpy(), tW2.grad.numpy(), tb2.grad.numpy()]

# Utils
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# use this to check your implementation
# you can pass in grads_truth as truth and the output of get_grads as our_impl
def compare_grads(truth, our_impl):
  for elt1, elt2 in zip(truth, our_impl):
    if not np.allclose(elt1, elt2, atol=0.001, rtol=0.001):
      return False
  
  return True

# Implement the forward pass of the data. Perhaps you can return some variables that 
# will be useful for calculating the gradients. 
def forward(X):
  return X

# Code the gradients you found in part 2.
# Can pass in additional arguments
def get_grads(y, yhat, X): 
  dLdb2 = None
  dLdW2 = None
  dLdb1 = None
  dLdW1 = None
  
  # make sure this order is kept for the compare function
  return [dLdW1, dLdb1, dLdW2, dLdb2]
  