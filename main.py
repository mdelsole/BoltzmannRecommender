# import the libraries
import numpy as np
import pandas as pd
import torch
# for neural networks
import torch.nn as nn
# for parallel computations
import torch.nn.parallel as parallel
# for the optimizer
import torch.optim as optim
# for some utilities
import torch.utils.data
# for stochasitic gradient descent
from torch.autograd import Variable

# Prepare the data set

# Delimiter (separator) is \t for tab
training_set = pd.read_csv("ml-100k/u1.base", delimiter='\t')
# Easier to work with arrays, so lets convert it. Convert all the values to integers to work with it
training_set = np.array(training_set, dtype="int")
test_set = pd.read_csv("ml-100k/u1.test", delimiter='\t')
test_set = np.array(test_set, dtype="int")

# Get whatever the last user # is, and the highest of the two from the training_set of test_set for the loop later
num_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
num_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Boltzmann Machine requires specific input. We need input value and the feature


# Convert the data into an array with users as rows and movies in columns: input = user, features = movie watched
def convert(data):
    # Make a list of lists. Each of the 943 users in the list will have a list of which of the 1682 movies they watched
    # List instead of array because torch expects lists of lists (says it still works with arrays though)
    new_data = []
    for user_ids in range(1,num_users+1):
        # Second bracket is syntax for a conditional, says only get all movies for current user
        movie_ids = data[:,1][data[:,0] == user_ids]
        rating_ids = data[:,2][data[:,0] == user_ids]
        # Create a new empty array
        ratings = np.zeros(num_movies)
        # Fill the empty array with the ratings of the movies. Movie_ids holds all the indexes of the movies
        ratings[movie_ids-1] = rating_ids
        new_data.append(list(ratings))
    return new_data

# Convert the training set and the test set so that they are ready for the boltzmann machine
training_set = convert(training_set)
test_set = convert(test_set)

# Convert data into Torch tensor (tensor is just a multi-dimensional matrix)
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# On to the actual Boltzmann Machine

# Convert the ratings into binary ratings: 1 = liked, 0 = not liked, b/c we're predicting a binary "will they like it",
# and if you remember RBMs, the output and input are swapped as it works, so they must be the same format

# Get rid of all the missing values. Remember brackets are conditionals.
training_set[training_set == 0] = -1
# 1/2 stars means they didn't like it, 3/4/5 stars means they did like it
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Create the architecture of the Neural Network
class RBM():
    # Every class must have an __init__. It's like the constructor
    def __init__(self, num_visible_nodes, num_hidden_nodes):
        # Initialize weights, which are the probabilities of the visible nodes given the hidden nodes
        self.W = torch.randn(num_hidden_nodes,num_visible_nodes)
        # Initialize bias for hidden nodes
        self.a = torch.randn(1, num_hidden_nodes)
        # Initialize bias for visible nodes
        self.b = torch.randn(1, num_visible_nodes)

    # Sample the hidden nodes using bernoulli distribution according to the probability hidden given visible
    def sample_h(self, x):
        # wx from z = wx + b. Torch.nn makes the product of two tensors. .t() to transpose it
        wx = torch.mm(x, self.W.t())
        # z from z = wx + b, just like normal ANN
        # Each input vector is not treated individually like in an ANN, but rather in batches, as we use all of the
        # inputs to predict a singular output. So to apply the bias to all the parts of wx, the tensors must be the same
        # size. Use expand_as to expand the bias tensor to size of wx. Could also use a.expand(wx.size())
        activation = wx + self.a.expand_as(wx)
        # probability is given by the sigmoid activation function
        prob_hid_given_vis = torch.sigmoid(activation)
        # We're making a bernoulli RBM because we're predicting a binary outocme, so fit results to that distribution
        # Bernoulli sampling: determines the threshold, maybe 0.77, and everything the below that will activate the
        # neuron, everything above that won't
        return prob_hid_given_vis, torch.bernoulli(prob_hid_given_vis)

    # Sample the visible nodes using bernoulli distribution according to the probability visible given hidden
    def sample_v(self, y):
        # No transpose because x column of weights matches with hidden nodes (y is the hidden nodes)
        wy = torch.mm(y, self.W)
        # b bias instead of a because it's for converting to visible nodes
        activation = wy + self.b.expand_as(wy)
        prob_vis_given_hid = torch.sigmoid(activation)
        return prob_vis_given_hid, torch.bernoulli(prob_vis_given_hid)

    # Contrastive divergence: minimize the weights to minimize/optimize the energy i.e. maximize the log likelihood
    # v0 is the first input vector, vk is the visible nodes after k epochs (round trips to hidden then back to visible)
    # ph0 is probabilities at first input vector, phk is probabilities of yes after k epochs
    def train(self, v0, vk, ph0, phk):
        # The following is just the algorithm for contrastive divergence
        # ph0 = probability that the hidden nodes = 1 given the input vector
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(v0.t(), phk)).t()
        self.b += torch.sum((v0-vk), 0)
        self.a += torch.sum((ph0-phk), 0)

# num visible nodes (number of movies, because we have a node for each movie), could also say nv = num_movies, but it's
# safer to do this, based off our tensors, looking at the length of a line in training_set
num_vis = len(training_set[0])
# Try to detect 100 features, 100 is just the arbitrary number I chose
num_hid = 100
# Update the weights after several observations. 1 would be online learning, updating weights after each observation
# We'll try after 100 observations
batch_size = 100
rbm = RBM(num_vis,num_hid)

# Training the rbm
num_epochs = 10
for epoch in range(1, num_epochs+1):
    # Keep track of our loss (cost function i.e. residual)
    train_loss = 0
    # Need to normalize loss, we'll divide loss by this, . is to make it float
    s = 0.
    # Go through the users in batches (step is batch_size)
    for user_id in range(0, num_users-batch_size, batch_size):
        # Set vk to be values in the training set from current user_id to uesr_id+batch_size
        vk = training_set[user_id:user_id+batch_size]
        # Set v0 so we can compare later for the loss function. At the begining, it will be the same as the target,
        # hence why they are the same code
        v0 = training_set[user_id:user_id+batch_size]
        # Set the initial probabilities equal to the initial bernoulli sampling of v, v0
        # Use ,_ to say we only want the first variable, since our class returns 2 variables
        ph0,_ = rbm.sample_h(v0)
        # K steps of contrastive divergence, do gibbs sampling, make the sampling equal the next h/v (back and forth)
        for k in range(10):
            # hk = Second element returned from sampling hidden nodes (the bernoulli sampling
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            # Ignore the missing nodes (the -1 nodes) by freezing them (set them always equal to v0)
            vk[v0<0] = v0[v0<0]
        # Get the sample for the hidden node applied on the last sample of the visible node
        phk,_ = rbm.sample_h(vk)
        # Apply contrastive divergence
        rbm.train(v0,vk,ph0,phk)
        # Calculate the loss, the difference in the original and the current sample (mean b/c we do many at once)
        # Conditional so we only factor in movies for which ratings exist
        train_loss += torch.mean(torch.abs(v0[v0 >= 0]-vk[v0 >= 0]))
        # Update the counter, which is to normalize the loss
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: '+str(train_loss/s))


# Testing the rbm, basically just get rid of all code that has us do it multiple times, since only need one additional
test_loss = 0
s = 0.
for user_id in range(num_users):
    # Keep as training set because our input is from our training data
    v = training_set[user_id:user_id+1]
    # vt = visible nodes in target
    vt = test_set[user_id:user_id+1]
    # Make sure the tensor isn't empty (no rating), since that'll give an error
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))
