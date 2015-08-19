
# coding: utf-8

# In[1]:

import theano
import theano.tensor as T
from theano import function
from theano import shared
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import os
import time
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

theano.config.compute_test_value = 'off'

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


# In[2]:

class Model:
    
    def __init__(self, labels, size, batch_size, values):
        
        nkerns=[20, 50, 50, 50]
        width, height = size
        
        rng = np.random.RandomState(23455)  # number is the seed of random generator

        nclasses = len(labels)

        # allocate symbolic variables for the data
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                                # [int] labels

        # Reshape matrix of rasterized images of shape (batch_size,28*28) -> width*height
        # to a 4D tensor, compatible with our LeNetConvPoolLayer TODO: Check reshape !
        layer0_input = x.reshape((batch_size, 1, height, width))

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

        filter_size = (4,4)
        layer15 = LeNetConvPoolLayer(rng, input=layer1.output,
                    image_shape=(batch_size, nkerns[1], out_height, out_width),
                    filter_shape=(nkerns[2], nkerns[1], filter_size[0], filter_size[1]), poolsize=pool_size)
        out_width = (out_width - filter_size[0] + 1) / 2
        out_height = (out_height - filter_size[1] + 1) / 2

        filter_size = (4,4)
        layer17 = LeNetConvPoolLayer(rng, input=layer15.output,
                    image_shape=(batch_size, nkerns[2], out_height, out_width),
                    filter_shape=(nkerns[3], nkerns[2], filter_size[0], filter_size[1]), poolsize=pool_size)
        out_width = (out_width - filter_size[0] + 1) / 2
        out_height = (out_height - filter_size[1] + 1) / 2

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        layer2_input = layer17.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[3] * out_width * out_height,
                                 n_out=batch_size, activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=nclasses)

        # create a list of all model parameters to be fit by gradient descent
        params = layer3.params + layer2.params + layer17.params + layer15.params + layer1.params + layer0.params
                
        for param, value in zip(params, values):
            param.set_value(value)
        
        self.predict_soft = theano.function([x], layer3.p_y_given_x)
        


# In[3]:

import cPickle
from PIL import Image
import os
import numpy as np

class AircraftSpotter:
    
    def __init__(self, model_file):
        with file(model_file, 'rb') as f:
            #self.predict = cPickle.load(f)
            #self.predict_soft = cPickle.load(f)
            values = cPickle.load(f)
            self.labels = cPickle.load(f)
            self.size = cPickle.load(f)
            self.batch_size = cPickle.load(f)
            self.model = Model(self.labels, self.size, self.batch_size, values)
            
    def resize_crop(self, img):
        img_ratio = img.size[0] / float(img.size[1])
        tgt_ratio = self.size[0] / float(self.size[1])
        if tgt_ratio > img_ratio:
            img = img.resize((self.size[0], self.size[0] * img.size[1] / img.size[0]), Image.ANTIALIAS)
            box = (0, (img.size[1] - self.size[1]) / 2, img.size[0], (img.size[1] + self.size[1]) / 2)        
            img = img.crop(box)
        elif tgt_ratio < img_ratio:
            img = img.resize((self.size[1] * img.size[0] / img.size[1], self.size[1]), Image.ANTIALIAS)
            box = ((img.size[0] - self.size[0]) / 2, 0, (img.size[0] + self.size[0]) / 2, img.size[1])
            img = img.crop(box)
        else :
            img = img.resize((self.size[0], self.size[1]), Image.ANTIALIAS)
        return img
    
    def predict_image(self, path):
        img = Image.open(path).convert('L')
        img = self.resize_crop(img)
        img.save(path+".jpg", "JPEG")
        array = np.asarray(img, dtype=np.float32) / 256.
        
        data = np.zeros(shape=(self.batch_size, self.size[0]*self.size[1]), dtype=np.float32)
        data[0] = array.flatten()
            
        probas = self.model.predict_soft(data)[0]
        res = []
        for i,p in enumerate(probas):
            #print " - {0}: {1:.3f}%".format(self.labels[i],p*100)
            res.append((self.labels[i], p*100, "{0}: {1:.3f}%".format(self.labels[i],p*100)))
        res.sort(key=lambda tup: tup[1], reverse=True)
        return res
        
spotter = AircraftSpotter(r'results/model-values.save')


# In[ ]:

import SimpleHTTPServer
import SocketServer
import re

class Handler(SimpleHTTPServer.SimpleHTTPRequestHandler):
            
    def do_GET(self):
        if 'tempfile.jpg' in self.path:
            print "GET pic..."
            SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
        else:
            print "GET main..."
            with open('server/get.html', 'r') as f:
                read_data = f.read()
                self.wfile.write(read_data)     
    
    def do_POST(self):
        print "POST main..."
        r, info, fn = self.deal_post_data()
        print r, info, "by: ", self.client_address
        res = spotter.predict_image(fn)
        with open('server/post.html', 'r') as f:
            read_data = f.read().replace("{{PLACEHOLDER}}", self.toHTML(res))
            self.wfile.write(read_data)

    def toHTML(self, results):
        s = "<ul>"
        for r in results:
            s = s + "<li>"+r[2]+"</li>"
        s = s + "</ul>"
        return s
            
    def deal_post_data(self):
        boundary = self.headers.plisttext.split("=")[1]
        remainbytes = int(self.headers['content-length'])
        line = self.rfile.readline()
        remainbytes -= len(line)
        if not boundary in line:
            return (False, "Content NOT begin with boundary", None)
        line = self.rfile.readline()
        remainbytes -= len(line)
        fn = re.findall(r'Content-Disposition.*name="file"; filename="(.*)"', line)
        if not fn:
            #return (False, "Can't find out file name...")
            fn = ["server/tempfile"]
        path = self.translate_path(self.path)
        fn = os.path.join(path, fn[0])
        line = self.rfile.readline()
        remainbytes -= len(line)
        line = self.rfile.readline()
        remainbytes -= len(line)
        try:
            out = open(fn, 'wb')
        except IOError:
            return (False, "Can't create file to write, do you have permission to write?", None)
                
        preline = self.rfile.readline()
        remainbytes -= len(preline)
        while remainbytes > 0:
            line = self.rfile.readline()
            remainbytes -= len(line)
            if boundary in line:
                preline = preline[0:-1]
                if preline.endswith('\r'):
                    preline = preline[0:-1]
                out.write(preline)
                out.close()
                return (True, "File '%s' upload success!" % fn, fn)
            else:
                out.write(preline)
                preline = line
        return (False, "Unexpect Ends of data.", None)
        
print "serving at port", 80
SocketServer.TCPServer(("", 80), Handler).serve_forever()


# In[ ]:



