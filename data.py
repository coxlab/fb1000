from os import path
import numpy as np

DATA_PATH = path.join(path.dirname(__file__), "fb1000_data")


def get(stride=5):

    X_trn = np.memmap(path.join(DATA_PATH, 'L3_Prime_X_trn_zscored.mm'),
                      mode='r', shape=(90000, 51200), dtype='float32')
    X_trn = X_trn[:, ::stride]
    y_trn = np.memmap(path.join(DATA_PATH, 'L3_Prime_Y_trn.mm'),
                      mode='r', shape=(90000,), dtype='float32')

    X_tst = np.memmap(path.join(DATA_PATH, 'L3_Prime_X_tst_zscored.mm'),
                      mode='r', shape=(10000, 51200), dtype='float32')
    X_tst = X_tst[:, ::stride]
    y_tst = np.memmap(path.join(DATA_PATH, 'L3_Prime_Y_tst.mm'),
                      mode='r', shape=(10000,), dtype='float32')

    return X_trn, y_trn, X_tst, y_tst
