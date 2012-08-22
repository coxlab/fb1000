import numpy as np

from scipy.optimize import fmin_l_bfgs_b

import theano
from theano import tensor


class SVML2HuberLBFGS(object):

    def __init__(self, n_classes, n_features):

        self.dtype = dtype = 'float32'

        self.n_features = n_features
        self.n_classes = n_classes

        self.W = np.zeros((n_features, n_classes), dtype=dtype)
        self.b = np.zeros(n_classes, dtype=dtype)

    def fit(self, X, y):

        # -- One-vs-Rest multi-class formulation
        y = 2. * (y[:, np.newaxis] == np.arange(self.n_classes)) - 1.
        y = np.asarray(y, dtype=self.dtype)
        X = np.asarray(X, dtype=self.dtype)

        # -- theano variables
        t_X = tensor.fmatrix()
        t_y = tensor.fmatrix()
        t_W = tensor.fmatrix()
        t_b = tensor.fvector()

        # -- margins
        t_M = t_y * (tensor.dot(t_X, t_W) + t_b)

        # -- modified huber loss (l2)
        t_loss = tensor.switch(
            t_M > -1,
            tensor.maximum(0, 1 - t_M) ** 2,
            -4 * t_M)
        t_loss = t_loss.mean()

        # -- gradients
        t_dloss_dW = tensor.grad(t_loss, t_W)
        t_dloss_db = tensor.grad(t_loss, t_b)

        print 'Compiling theano functions....'
        _f_df = theano.function(
            # inputs
            [t_X, t_y, t_W, t_b],
            # outputs
            [t_loss, t_dloss_dW, t_dloss_db],
            allow_input_downcast=True,
            )

        def unpack(params):
            W = params[:self.W.size].reshape(self.W.shape)
            b = params[self.W.size:].reshape(self.b.shape)
            return W, b

        def pack(W, b):
            params = np.concatenate([W.ravel(), b.ravel()])
            return params

        def func(params):
            W, b = unpack(params)
            loss, dloss_dW, dloss_db = _f_df(X, y, W, b)
            params = pack(dloss_dW, dloss_db)
            return loss.astype('float64'), params.astype('float64')

        print "Optimizing with L-BFGS..."
        params0 = pack(self.W, self.b)
        best, bestval, info_dct = fmin_l_bfgs_b(
            func, params0,
            m=10,
            iprint=1,
            factr=1e7,
            maxfun=1000,
            )

        self.W, self.b = unpack(best)

        return self

    def decision_function(self, X):
        X = np.asarray(X)
        return np.dot(X, self.W) + self.b

    def predict(self, X):
        return self.decision_function(X).argmax(axis=1)
