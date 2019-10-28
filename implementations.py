# ==============================================================================
# implementations.py
# ------------------------------------------------------------------------------
# authors:                             Rémi Clerc, Jordan Metz, Jonas Morin
# date:                                28.10.2019
# ==============================================================================
# This file implements the functions required for project 1 of the Machine 
# Learning course. It also contains other utility functions used solve the
# problems and achieve a better grade in the competition.
# Most functions here works with the help of the homebrewed library ml_math.
# ==============================================================================

import numpy as np
from ml_math import * #homebrewed library
from misc_helpers import *


def least_squares_GD(y, tx, initial_w, max_iters = 100, gamma = 0.7, verbose = False):
    """
    ----------------------------------------------------------------------------
    Iteratively compute the model weights "w" from "y" and "tx" using the
    gradient descent algorithm starting at "initial_w" with a maximum of
    "max_iters" steps and step size "gamma".
    The function also returns the loss computed as the mean square error (mse).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - max_iters     # of iterations after which the procedure will stop, int>0,
                    (default = 100)
    - gamma         step size, scalar in ]0,1[, (default = 0.7)
    - verbose       boolean to enable/disable the display of the loss in the for
                    loop used for error handling, (default= False)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """
    # Define the loss function to use
    loss_function = "MSE"
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y,tx,w, loss_function)
        loss = compute_loss(y,tx,w, loss_function)
        
        # update w by gradient
        w = w - gamma * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        # print the loss if verbose is true
        if verbose == True:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
            

    return ws[-1],losses[-1]

def least_squares_SGD(y, tx, initial_w, batch_size = 1, max_iters = 100, gamma = 0.7, verbose=False):
    """
    ----------------------------------------------------------------------------
    Iteratively compute the model parameters "w" from "y" and "tx" using the
    stochastic gradient descent algorithm starting at "initial_w" with a
    maximum of "max_iters" steps and step size "gamma". Only use a portion of
    the gradient defined by "batch_use".
    The function also returns the loss computed as the mean square error.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - batch_size    defines the number of 'directions' for the gradient to be
                    computed (default = 1)
    - max_iters     # of iterations after which the procedure will stop, int>0,
                    (default = 100)
    - gamma         step size, scalar in ]0,1[, (default = 0.7)
    - verbose       boolean to enable/disable the display of the loss in the for
                    loop used for error handling, (default = False)
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """
    
    # Define the loss function to use
    loss_function = "MSE"
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        for minibatch_y, minibatch_tx in batch_iter(y,tx,batch_size):
            gradient = compute_gradient(minibatch_y,minibatch_tx,w, loss_function)
        loss = compute_loss(y,tx,w, loss_function)
        
        # update w by gradient
        w = w - gamma * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        # print the loss if verbose is true
        if verbose == True:
            print("Stoch Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1],losses[-1], 

def least_squares(y, tx):
    """
    ----------------------------------------------------------------------------
    Compute the model weights "w" from "y" and "tx" using the
    normal equations.
    The function also returns the loss computed as the mean square error (mse).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """
    # define the loss function to use
    loss_function = "MSE"
    
    # computes the weights and loss
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y,tx,w, loss_function)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    ----------------------------------------------------------------------------
    Compute the model weights "w" from "y" and "tx" using the
    normal equations with reagularization parameter "lambda_".
    The function also returns the loss computed as the mean square error (mse).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - lambda_       regularization parameter, scalar>0
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """
    
    # define the loss function to use
    loss_function = "MSE"
    
    # define dimension of the system
    D = np.shape(tx)[1]
    N = np.shape(tx)[0]
    
    
    # computes the weights and loss
    w = np.linalg.solve(tx.T.dot(tx)+2*N*lambda_*np.identity(D), tx.T.dot(y))
    loss = compute_loss(y,tx,w, loss_function)
    
    return w,loss

def logistic_regression(y, tx, initial_w, max_iters = 100, gamma = 0.7, verbose = False, use_SGD = True, batch_size = 1, loss_function = "LL2"):
    """
    ----------------------------------------------------------------------------
    Iteratively computes the model weights "w" from "y" and "tx" using
    logistic regression with an optional reagularization parameter "lambda_"
    in up "max_iters" iterations.
    The function also returns the loss computed according to the logistic
    regression loss function.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - max_iters     # of iterations after which the procedure will stop, int>0,
                    (default = 100)
    - gamma         step size, scalar in ]0,1[, (default = 0.7)
    - verbose       boolean to enable/disable the display of the loss in the for
                    loop used for error handling, (default = False)
    - use_SGD       boolean to choose between stochastic gradient descent (true)
                    or normal gradient descent (default = true)
    - batch_size    defines the number of 'directions' for the gradient to be
                    computed (default = 1)
    - loss_function choose the loss_function to use, (default = "LL2")
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        if use_SGD:
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                gradient = compute_gradient(minibatch_y, minibatch_tx, w, loss_function)
        else:
            gradient = compute_gradient(y, tx, w, loss_function)
        loss = compute_loss(y, tx, w, loss_function)
        
        # update w by gradient
        w -= gamma * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        # print the loss if verbose is true
        if verbose == True:
            print("Stoch Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters = 100, gamma = 0.7, verbose = False, use_SGD = True, batch_size = 1, loss_function = "LL2"):
    """
    ----------------------------------------------------------------------------
    Iteratively computes the model weights "w" from "y" and "tx" using
    regularized logistic regression with reagularization parameter "lambda_"
    in up "max_iters" iterations.
    The function also returns the loss computed according to the logistic
    regression loss function.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - lambda_       regularization parameter, scalar>0
    - initial_w     initial guess of model weights, (nfeatures,1) np.array
    - max_iters     # of iterations after which the procedure will stop, int>0,
                    (default = 100)
    - gamma         step size, scalar in ]0,1[, (default = 0.7)
    - verbose       boolean to enable/disable the display of the loss in the for
                    loop used for error handling, (default = False)
    - use_SGD       boolean to choose between stochastic gradient descent (true)
                    or normal gradient descent (default = true)
    - batch_size    defines the number of 'directions' for the gradient to be
                    computed (default = 1)
    - loss_function choose the loss_function to use, (default = "LL2")
    Output:
    - w             obtained weights, (nfeatures,1) np.array
    - loss          loss computed as the mean square error, scalar
    ----------------------------------------------------------------------------
    """
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        if use_SGD:
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = compute_gradient(minibatch_y, minibatch_tx, initial_w, loss_function) + 2 * lambda_ * w
        else:
            grad = compute_gradient(y, tx, w, loss_function) + 2 * lambda_ * w
        loss = compute_loss(y, tx, w, loss_function) + lambda_ * np.squeeze(w.T.dot(w))
        
        # update w by gradient
        w -= gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        # print the loss if verbose is true
        if verbose == True:
            print("Stoch Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]