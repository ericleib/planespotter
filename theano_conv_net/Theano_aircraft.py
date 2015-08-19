
# coding: utf-8

# In[1]:

import theano.tensor as T
from theano import function
from theano import shared
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import os
import time
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import matplotlib.cm as cm
from PIL import Image
import pylab
from random import shuffle


# In[18]:

class AircraftDataset:
    def __init__(self, directory, width, height, batch_size=1000):        
        self.labels = []
        self.directory = directory
        self.image_size = (width, height)
        self.n_pixels = width*height
        self.batch_size = batch_size
        self.load()

    def get_label(self, i):
        return self.labels[i]
    
    def load(self):
        print "Loading datasets..."
        self.training_set = self._load_dataset("training.txt")
        self.testing_set = self._load_dataset("testing.txt")
        self.validating_set = self._load_dataset("validating.txt")

    def _load_dataset(self, filename):
        file = os.path.join(self.directory, filename)
        print "Loading dataset: "+file
        with open(file) as f:
            lines = f.readlines()
            aircraft_list = [];
            for line in lines:
                words = line.strip().strip(r'"').split(r'" "')
                if(len(words) != 4):
                    raise Exception('wtf?')
                label = words[1] + " " + words[2]
                if not label in self.labels:
                    self.labels.append(label)
                aircraft_list.append(Aircraft(words[0], self.labels.index(label), self))
            shuffle(aircraft_list)
            print str(len(aircraft_list))+" aircraft in dataset!"
            return aircraft_list
    
    def get_training_minibatch(self, index):
        #print "Getting training minibatch "+str(index)
        return self._get_minibatch(self.training_set, index)
    
    def get_validating_minibatch(self, index):
        #print "Getting validation minibatch "+str(index)
        return self._get_minibatch(self.validating_set, index)
    
    def get_testing_minibatch(self, index):
        #print "Getting testing minibatch "+str(index)
        return self._get_minibatch(self.testing_set, index)
    
    def _get_minibatch(self, list, index):
        sub_list = list[index * self.batch_size: (index + 1) * self.batch_size]
        data = np.zeros(shape=(len(sub_list),self.n_pixels),dtype=theano.config.floatX)
        target = np.empty([len(sub_list)], dtype=np.int32)
        for i, aircraft in enumerate(sub_list):
            data[i] = aircraft.load_image()
            target[i] = aircraft.get_label()
        return (data, target)
        
        """
        return theano.shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
        return  T.cast(theano.shared(np.asarray(target, dtype=theano.config.floatX), borrow=True), 'int32')
        """

class Aircraft:        
    def __init__(self, path, label, dataset):
        self.path = path
        self.label = label
        self.dataset = dataset

    def get_label(self):
        return self.label

    def get_label_text(self):
        return self.dataset.get_label(self.get_label())

    def load_image(self):
        img = Image.open(os.path.join(self.dataset.directory, self.path)).convert('L').resize(
            self.dataset.image_size, Image.ANTIALIAS)
        array = np.asarray(img, dtype=theano.config.floatX) / 256.
        #pylab.imshow(array, cmap = cm.Greys_r, vmin = 0., vmax = 1.); pylab.show()
        return array.flatten()
    
dataset_test = AircraftDataset(r"C:\Users\niluje\Documents\planespotter", 200, 133, 1)
dataset_test.get_training_minibatch(0)


# In[3]:

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
    def predict(self):
        return self.y_pred


# In[4]:

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]



# # LeNet Convolutional Neural Network

# In[5]:

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
    
    def return_output():
        return self.output


# ### Model Setup

# In[6]:

learning_rate=0.05
nkerns=[20, 50, 50, 50]
batch_size=125

directory = r"C:\Users\niluje\Documents\planespotter"
width = 196
height = 132
dataset = AircraftDataset(directory, width, height, batch_size)

""" Demonstrates lenet on Aircraft dataset

:type learning_rate: float
:param learning_rate: learning rate used (factor for the stochastic
                        gradient)

:type n_epochs: int
:param n_epochs: maximal number of epochs to run the optimizer

:type nkerns: list of ints
:param nkerns: number of kernels on each layer
"""

rng = np.random.RandomState(23455)  # number is the seed of random generator

# compute number of minibatches for training, validation and testing
n_train_batches = len(dataset.training_set) / batch_size
n_valid_batches = len(dataset.validating_set) / batch_size
n_test_batches =  len(dataset.testing_set) / batch_size
nclasses = len(dataset.labels)

print "Batch size: "+str(batch_size)
print "Training set size: "+str(len(dataset.training_set))+" ("+str(n_train_batches)+" batches)"
print "Validation set size: "+str(len(dataset.validating_set))+" ("+str(n_valid_batches)+" batches)"
print "Testing set size: "+str(len(dataset.testing_set))+" ("+str(n_test_batches)+" batches)"
print "Classes:"
for label in dataset.labels:
    print " -> "+label

# allocate symbolic variables for the data
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


x.tag.test_value = np.random.rand(batch_size, height*width).astype(np.float32)
y.tag.test_value = np.random.randint(0, nclasses-1, batch_size).astype(np.int32)
    
######################
# BUILD ACTUAL MODEL #
######################
print '... building the model'

# Reshape matrix of rasterized images of shape (batch_size,28*28) -> width*height
# to a 4D tensor, compatible with our LeNetConvPoolLayer TODO: Check reshape !
layer0_input = x.reshape((batch_size, 1, height, width))
print "Layer 0 input size: "+str(width)+"x"+str(height)

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
# maxpooling reduces this further to (24/2,24/2) = (12,12)
# 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
pool_size = (2,2)
filter_size = (9,8)
layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, height, width),
            filter_shape=(nkerns[0], 1, filter_size[0], filter_size[1]), poolsize=pool_size)
out_width = (width - filter_size[0] + 1) / 2
out_height = (height - filter_size[1] + 1) / 2
print "Layer 1 input size: "+str(out_width)+"x"+str(out_height) + " - "+str(layer0.output.shape.tag.test_value)

# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
# maxpooling reduces this further to (8/2,8/2) = (4,4)
# 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
filter_size = (7,6)
layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], out_height, out_width),
            filter_shape=(nkerns[1], nkerns[0], filter_size[0], filter_size[1]), poolsize=pool_size)
out_width = (out_width - filter_size[0] + 1) / 2
out_height = (out_height - filter_size[1] + 1) / 2
print "Layer 1.5 input size: "+str(out_width)+"x"+str(out_height) + " - "+str(layer1.output.shape.tag.test_value)

filter_size = (4,4)
layer15 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], out_height, out_width),
            filter_shape=(nkerns[2], nkerns[1], filter_size[0], filter_size[1]), poolsize=pool_size)
out_width = (out_width - filter_size[0] + 1) / 2
out_height = (out_height - filter_size[1] + 1) / 2
print "Layer 1.7 input size: "+str(out_width)+"x"+str(out_height) + " - "+str(layer15.output.shape.tag.test_value)

filter_size = (4,4)
layer17 = LeNetConvPoolLayer(rng, input=layer15.output,
            image_shape=(batch_size, nkerns[2], out_height, out_width),
            filter_shape=(nkerns[3], nkerns[2], filter_size[0], filter_size[1]), poolsize=pool_size)
out_width = (out_width - filter_size[0] + 1) / 2
out_height = (out_height - filter_size[1] + 1) / 2
print "Layer 2 input size: "+str(out_width)+"x"+str(out_height) + " - "+str(layer17.output.shape.tag.test_value)

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (20,32*4*4) = (20,512)
layer2_input = layer17.output.flatten(2)
print "Layer 2 flattened input size: "+str(layer2_input.shape.tag.test_value)


# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[3] * out_width * out_height,
                         n_out=batch_size, activation=T.tanh)
print "Layer 2 shape: "+str(nkerns[3] * out_width * out_height)+" - "+str(layer2.output.shape.tag.test_value)

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=nclasses)

# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
validate_model = theano.function([x,y], layer3.errors(y))

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer17.params + layer15.params + layer1.params + layer0.params
print "Network parameters length: "+str(len(params))

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i],grads[i]) pairs.
updates = []
for param_i, grad_i in zip(params, grads):
    updates.append((param_i, param_i - learning_rate * grad_i))

train_model = theano.function([x,y], cost, updates=updates)


# ### Model Training

# In[8]:

n_epochs=200
###############
# TRAIN MODEL #
###############
print '... training'
# early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                           # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

best_validation_loss = np.inf
best_iter = 0

epoch = 0
done_looping = False

def stop_looping():
    done_looping = True

#button = widgets.ButtonWidget(description = 'Stop looping')
#button.on_click(stop_looping)
#display(button)

#f = FloatProgress(min=0, max=n_train_batches)
#display(f)

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):
        
        #f.value = minibatch_index
        
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if iter % 100 == 0:
            print 'training @ iter = ', iter, ' patience = ', patience
            
        data,target = dataset.get_training_minibatch(minibatch_index)
        cost_ij = train_model(data,target)

        if (iter + 1) % validation_frequency == 0:

            # compute zero-one loss on validation set
            validation_losses = []
            for i in xrange(n_valid_batches):
                data,target = dataset.get_validating_minibatch(i)
                validation_losses.append(validate_model(data, target))
            this_validation_loss = np.mean(validation_losses)
            print('epoch %i, validation error %f %%' %                       (epoch, this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

        if patience <= iter:
            done_looping = True
            break


# ### Prediction and Submission File Creation

# In[14]:

model_predict = theano.function([x], layer3.y_pred)
model_soft_predict = theano.function([x], layer3.p_y_given_x)

test_results = np.zeros(shape=(nclasses,nclasses))
for i in xrange(n_test_batches):
    data,target = dataset.get_testing_minibatch(i)
    predictions = model_predict(data)
    for j in xrange(batch_size):
        t = target[j]
        p = predictions[j]
        test_results[t][p] = test_results[t][p] + 1
        
print "Test results:"
for i in xrange(nclasses):
    total = 0
    for j in xrange(nclasses):
        total = total + test_results[i][j]
    print " - "+dataset.get_label(i)+": "+str(test_results[i][i])+" predicted / "+str(total)+" ("+str(100*test_results[i][i]/total)+"%)"
print "All results:"
for i in xrange(nclasses):
    total = 0
    for j in xrange(nclasses):
        total = total + test_results[j][i]
    print " - "+dataset.get_label(i)+": "+str(test_results[i][i])+" predicted / "+str(total)+" ("+str(100*test_results[i][i]/total)+"%)"



# In[22]:

import cPickle
f = file('../webapp/results/model-values.save', 'wb')
#cPickle.dump(model_predict, f, protocol=cPickle.HIGHEST_PROTOCOL)
#cPickle.dump(model_soft_predict, f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump([p.get_value() for p in params], f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(dataset.labels, f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(dataset.image_size, f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(dataset.batch_size, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()


# In[ ]:



