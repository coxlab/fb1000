Facebook1000 Experiments
========================

Simple bagging with features from HT-L3s and 1000 individuals from Zak's
Facebook dataset.

Data
----

To get the data you'll first need to get access to the `aws@coxlab.org`
S3 account. Then you can use `s3cmd` to retrieve the precomputed numpy
memmap files:

    $ s3cmd get -r s3://fb1000/fb1000_data .

    $ cd fb1000_data
    $ cat L3_Prime_X_trn_zscored.mm.split?? > L3_Prime_X_trn_zscored.mm

    $ sha1sum *.mm
    395e45223b0012a7783947e32373920812563427  L3_Prime_X_trn_zscored.mm
    9817c3a8bf40ead293392c273689ecddef97ed45  L3_Prime_X_tst_zscored.mm
    7ae7f27b72ced3b953cc9d5922b9e8e473027bf7  L3_Prime_Y_trn.mm
    829107227ba9aa49a23dce7ea9d72aa19fe1a8eb  L3_Prime_Y_tst.mm


Code
----

    $ grep "^[[:upper:]].* = [[:digit:]].*$" fb1k_demo.py
    DATA_STRIDE = 5
    N_BAGS = 32
    BAG_SIZE = 1e4

    $ python fb1k_demo.py
    (...)
    Total time (all bags): 1609.31552291
    >>> Testing...
    Accuracy=0.81
