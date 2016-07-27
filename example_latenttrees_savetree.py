import numpy as np
import latenttrees.lt_model as lt

# create model
model = lt.LatentTree()
# set most important parameters

# the prediction output of my model is a probability distribution. For
# continuous variables, the default output is the expected value of the
# predicted distribution. However, for discrete variables, there are
# several options:
# (a) 'exp': return the expected value when assuming that the class indices
# correspond to discrete intensities, i.e. class 0 corresponds to intensity
# 0, class 1 corresponds to intensity 1, etc. Use this, if
# there is an order of classes!
# (b) 'max': return the class index with the highest probability
model.inference.extract_samples_mode = 'max'
model.structure_update.k_default = 10 # this paramter needs to be cross-validated. a reasonable range is between 5 and 20
model.structure_update.lklhd_mindiff = 0.01
model.structure_update.lklhd_mindiff_siblings = -0.1

# Example data X with M=10 dimensions and N=100 samples
N = 100
M = 10
X = np.zeros((N, M))

# K describes the nature of each dimension of the data X: 1 for continuous variables and
# the number of states for discrete variables.
K = np.zeros((M,), dtype=np.int)
# dim 0-7: continuous, gaussian distributed
K[0:8] = 1
# dim 8: discrete with 3 levels: 0-2
K[8] = 3
# dim 9: discrete with 5 levels: 0-4
K[9] = 5

# fill X_train with random values
X[:,:8] = np.random.randn(N,8) # random Gaussian
X[:, 8] = np.random.randint(3, size=(N,)) # random integer between 0 and 2
X[:, 9] = np.random.randint(5, size=(N,)) # random integer between 0 and 4

# use the same data as for training and testing, just as example.
X_train = X
X_test = X

# Train the model.
# Training assumes that all dimensions are observed.
# Note: Missing features during training are not implemented yet!
model.training(X_train, K)
print('lklhd_train = {}'.format(model.lklhd))

# now test the model.
# assume that only the dimensions [0:6, 8] are observed and we try to infer
# dimensions [6 9]
ind_o = list(range(6)) + [8] # this are the indices of the observed dimensions
ind_u = [6, 9] # this are the indices of the unoberved dimensions

# testing gets as input only the observed dimensions (indexed by ind_o)
X_prediction, lklhd_test = model.testing(X_test[:, ind_o], ind_o, ind_u)
# the output are only the unobserved dimensions (indexed by ind_u), i.e.
# size of X_prediction is N x 2


adjlist = model.graph.get_adjlist()
for line in adjlist:
    print(line)

# draw the learned tree (needs pygraphviz installed)
model.graph.draw()