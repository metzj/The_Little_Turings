# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
import math
from implementations import *
from proj1_helpers import *
from misc_helpers import *
from plot_functions import *
from ml_math import *

from numpy.random import randint
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0+np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    s = 0
    for n in range(len(y)):
        s += np.log(1 + np.exp(np.dot(tx[n,:].T,w))) - y[n]*np.dot(tx[n,:].T,w)
    return s 

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T,sigmoid(np.dot(tx,w))-y)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    N = len(y)
    S = np.eye(N)
    for n in range(N):
        pred = sigmoid(np.dot(tx[n,:].T,w))
        S[n,n] = pred*(1-pred)
    return tx.T.dot(S.dot(tx))

def logistic_regression(y, tx, w, newton = False):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    if newton:
        hess = calculate_hessian(y, tx, w)
    else:
        hess = 0
    return loss, gradient, hess

def penalized_logistic_regression(y, tx, w, lambda_, newton = False):
    """return the loss, gradient, and hessian."""
    loss, gradient, hess = logistic_regression(y, tx, w, newton)
    loss += lambda_/2*np.linalg.norm(w)**2
    gradient += lambda_*w
    if newton:
        hess += lambda_*np.eye(len(w)).dot(w)
    else:
        hess = 0
    return loss, gradient, hess

def logistic_regression_ADAM(y , tx, lambda_, maxit, w0, verbose = False):
    """return w using ADAM"""
    n , p =np.shape(tx)
    w = w0
    w_prev = w
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1E-8
    m_prev = 0
    v_prev = 0
    for k in range(maxit):
        g = calculate_gradient(y, tx, w_prev) + lambda_ * w
        m = beta1*m_prev + (1-beta1)*g
        v = beta2*v_prev + (1-beta2)*g**2
        m_hat = m/(1-beta1)
        v_hat = v/(1-beta2)
        H = np.sqrt(v_hat)+eps
        w_next = w - alpha*m_hat/H
        w_prev = w
        w = w_next
        m_prev = m
        v_prev = v
        #loss = calculate_loss(y, tx, w)
        #if not k%10 and verbose:
        #if verbose:
            #print ('%d : loss = %f, norm(g) = %f'%(k,loss,np.linalg.norm(w)) )
            #print(k)
    return w

def logistic_regression_GD(y , tx, gamma, lambda_, maxit, verbose = False):
    """return w using ADAM"""
    n , p =np.shape(tx)
    w = np.zeros(p)
    
    for k in range(maxit):
        g = calculate_gradient(y, tx, w) + lambda_ * w
        w = w - gamma*g
        
        #loss = calculate_loss(y, tx, w)
        #if not k%10 and verbose:
        if verbose:
            #print ('%d : loss = %f, norm(g) = %f'%(k,loss,np.linalg.norm(w)) )
            print(k)
    return w

def logistic_regression_SGD(y , tx, lambda_, maxit, verbose = False):
    """return w using ADAM"""
    n , p =np.shape(tx)
    w = np.zeros(p)
    
    for k in range(maxit):
        i = randint(0,n),
        alpha = (n/2)/(k+(n/2))
        g = calculate_gradient(y[i], tx[i,:], w) + lambda_ * w
        w = w - alpha * g
        
        #if not k%100 and verbose:
            #print(k)
    return w

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
def compute_error(y,tX,w):
    erre = np.dot(tX,w)
    e = y - np.dot(tX,w)
    return 0
    
def cross_validation(y, x, k_fold, solver = 'LS',stoch = True,lambda_ = 0, maxit = 1):
    """return the loss of ridge regression."""
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    
    mse_tr = 0
    mse_te = 0
    w_0 = 0
    p = np.shape(x)[1]
    w0 = np.zeros(p)
    w_s = np.zeros(p)
    
    for k in range(k_fold):
        # get k'th subgroup in test, others in train:
        test_indices = k_indices[k]
        train_indices = np.delete(k_indices,k,0).flatten()
        x_tr = x[train_indices]
        y_tr = y[train_indices]
        x_te = x[test_indices]
        y_te = y[test_indices]

        # Least squares:
        if solver == 'LS':
            w, loss = least_squares(y_tr, x_tr)
        elif solver == 'RR':
            w, loss = ridge_regression(y_tr, x_tr, lambda_)
        elif solver == 'LR':
            w = logistic_regression_ADAM(y_tr ,x_tr, lambda_, maxit, w0)
        else:
            raise('Error')

        # Train loss
        b = np.dot(x_tr, w)
        b[b>0] = 1.0
        b[b<0] = 0.0
        loss_tr = np.linalg.norm(y_tr-b,1)/ len(y_tr)
        #validation loss
        b = np.dot(x_te, w)
        b[b>0] = 1.0
        b[b<0] = 0.0
        loss_te = np.linalg.norm(y_te-b,1)/ len(y_te)
    
        mse_tr += loss_tr/k_fold
        mse_te += loss_te/k_fold
        w_s += w/k_fold
        w_0 = w
    
    return mse_tr, mse_te, w_s

def build_poly(x, degree, linear = False):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if linear == False:
        D = len(x[0,:])
        N = len(x[:,0])
        new_x = np.ones((N,1)) #add bias
        if degree>=1:
            for i in range(1,degree+1):
                new_x = np.append(new_x,x**i,axis=1) 
        return new_x
    else:
        m = np.zeros((len(x),degree+1))
        for j in range(degree+1):
            m[:,j] = x**j
        return m

#load data

y_test, tX_test, ids_test = load_csv_data('data/test.csv')



id_min_loss = np.array([ 0,  1,  2,  3,  4,  5,  6, 10, 11, 12, 13, 22, 24, 27])
min_deg = np.array([11., 12., 12.,  5.,  2., 12.,  2.,  1., 11.,  6., 12., 12.,  2.,
       11.,  1.,  1.,  1.,  1.,  1.,  8.,  1.,  2.,  3.,  9.,  3.,  1.,
        2.,  3.,  1.,  4.])
       
weights = np.load('w0.npy')

DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test_0 = build_poly(tX_test,0)
for i in  id_min_loss:
    deg = int(min_deg[i])
    tX_test_0 = np.append(tX_test_0, build_poly(tX_test[:,i], deg, linear=True),1)


OUTPUT_PATH = 'data/output.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tX_test_0)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

