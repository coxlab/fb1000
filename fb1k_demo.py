import time
import numpy as np

import data
from svm_lbfgs import SVML2HuberLBFGS

DATA_STRIDE = 5
N_BAGS = 2
BAG_SIZE = 1e4


def main():

    rng = np.random.RandomState(42)

    print '>>> Loading data...'
    X_trn, y_trn, X_tst, y_tst = data.get(stride=DATA_STRIDE)

    classes = np.unique(y_trn)
    n_classes = len(classes)
    n_features = X_trn.shape[1]
    print 'n_classes:', n_classes

    print 'shapes:'
    print X_trn.shape, y_trn.shape
    print X_tst.shape, y_tst.shape

    print '>>> Training...'
    clf = SVML2HuberLBFGS(n_classes, n_features)

    start = time.time()
    avg_w = np.zeros_like(clf.W)
    avg_b = np.zeros_like(clf.b)
    for i in xrange(N_BAGS):
        print '-' * 80
        print '>>> Bag %d' % (i + 1)
        # resampling w/o replacement
        ridx = rng.permutation(len(X_trn))[:BAG_SIZE]
        xx, yy = np.asarray(X_trn[ridx]), np.asarray(y_trn[ridx])
        print 'size=%d' % len(xx)
        print '>>> Fitting...'
        clf.fit(xx, yy)
        # simple averaging
        lr = 1. / (1. + i)
        avg_w = lr * clf.W.copy() + (1. - lr) * avg_w
        avg_b = lr * clf.b.copy() + (1. - lr) * avg_b
    end = time.time()
    ttime = end - start
    print 'Total time (all bags):', ttime

    # replace final coefficients and predict
    print '>>> Testing...'
    clf.W = avg_w.copy()
    clf.b = avg_b.copy()
    y_pred = clf.predict(X_tst)

    # accuracy (unbalanced)
    acc = (y_pred == y_tst).mean()
    print
    print
    print 'Accuracy=%.2f' % acc
    print
    print

if __name__ == '__main__':
    main()
