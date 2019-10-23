# functions that were required to be implemented for project 1
# Rémi Clerc, Jordan Metz, Jonas Morin
##############################################################

import numpy as np
from ml_math import * #homebrewed library
from misc_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters=100, gamma = 0.7, verbose=False):
    """Gradient descent algorithm.
    Return: w, loss"""
    # Define parameters to store w and loss
    loss_function = "MSE"
    ws = [initial_w]
    losses = []
    w = np.copy(initial_w)
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y,tx,w, loss_function)
        loss = compute_loss(y,tx,w, loss_function)
        
        # update w by gradient
        w = w - (gamma * gradient)
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if verbose == True:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
            

    return ws[-1],losses[-1]

def least_squares_SGD(y, tx, initial_w, batch_size=1, max_iters=100, gamma=0.7, verbose=False):
    """Stochastic gradient descent algorithm for MSE
    Return: w, loss"""
    
    # Define parameters to store w and loss
    loss_function = "MSE"
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = np.zeros([30])
        for minibatch_y, minibatch_tx in batch_iter(y,tx,batch_size):
            gradient += compute_gradient(minibatch_y,minibatch_tx,w, loss_function)
            
        loss = compute_loss(y,tx,w, loss_function)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        if verbose == True:
            print("Stoch Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1],losses[-1], 

def least_squares(y, tx):
    """calculate the least squares solution using normal equation
       Return: weights, MSE"""
    w = np.linalg.solve(tx.transpose().dot(tx), tx.transpose().dot(y))
    n = len(y)
    e = y-tx.dot(w)
    MSE = 1/(2*n)*e.transpose().dot(e)
    return w, MSE


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    Return: w, loss"""
    # ridge regression:
    D = np.shape(tx)[1]
    N = np.shape(tx)[0]
    w = np.linalg.solve(tx.transpose().dot(tx)+2*N*lambda_*np.identity(D), tx.transpose().dot(y))
    loss = compute_loss(y,tx,w, "MSE")
    return w,loss


def logistic_regression(y, tx, initial_w, max_iters = 100, gamma = 0.7, verbose = False, use_SGD = True, batch_size = 1):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss_function = "LL" #log likelyhood loss
    ws = [initial_w]
    losses = []
    w = initial_w
    grad = []
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w, loss_function)
        loss = compute_loss(y, tx, w, loss_function)
        
        w = w -gamma * grad
        ws.append(w)
        losses.append(loss)
        if verbose == True:
            print("Stoch Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters = 100, gamma = 0.7, verbose = False, use_SGD = True, batch_size = 1):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss_function = "LL"
    ws = [initial_w]
    losses = []
    w = initial_w
    grad = []
    for n_iter in range(max_iters):
        if use_SGD:
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = compute_gradient(minibatch_y, minibatch_tx, initial_w, loss_function) + 2 * lambda_ * w
        else:
            grad = compute_gradient(y, tx, w, loss_function) + 2 * lambda_ * w
        loss = compute_loss(y, tx, w, loss_function) + lambda_ * np.squeeze(w.T.dot(w))
        w -= gamma * grad
        ws.append(w)
        losses.append(loss)
        if verbose == True:
            print("Stoch Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]

def ridge_regression_GD(y, tx, initial_w, lambda_, max_iters=100, verbose=False):
    """Gradient descent algorithm.
    Return: w, loss"""
    # Define parameters to store w and loss
    loss_function = "MSE"
    ws = [initial_w]
    losses = []
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = -1/N*tx.T.dot(y-tx.dot(w)) + lambda_*w
        loss = compute_loss(y,tx,w, loss_function)
        L = 1/N*np.linalg.norm(tx.T.dot(tx),2) + lambda_
        # update w by gradient
        w = w - (2/(L+lambda_)) * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if verbose == True:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
            

    return ws[-1],losses[-1]