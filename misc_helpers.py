# some useful functions from the course that are not restricted
# to the scope of project 1
# Rémi Clerc, Jordan Metz, Jonas Morin
###############################################################

import numpy as np
from proj1_helpers import *

def load_reduced_training_data(data_path, delete_low_output_correlation=True, delete_high_feature_correlation=True):
    """
    This function will load the training data, with the possibility to remove features according to our results of Feature Selection
    """
    y, tX, ids = load_csv_data(data_path)
    
    features_low_output_correlation = [2, 7, 8, 14, 15, 16, 17, 18, 19, 20]
    features_high_feature_correlation = [5, 11, 23, 24, 25, 26, 27]
    
    if delete_low_output_correlation and delete_high_feature_correlation:
        tX_reduced = np.delete(tX, features_low_output_correlation + features_high_feature_correlation, axis=1)
    elif delete_high_feature_correlation:
        tX_reduced = np.delete(tX, features_high_feature_correlation, axis=1)
    elif delete_low_output_correlation:
        tX_reduced = np.delete(tX, features_low_output_correlation, axis=1)
    else:
        tX_reduced = np.copy(tX)
    
    return y, tX_reduced, ids
    
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
