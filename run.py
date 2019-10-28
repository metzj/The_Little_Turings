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
       
weights = np.array([-5.24641878e+07,  4.76664873e+07, -1.02408133e-01,  2.04453443e-03,
       -1.67652756e-05,  7.02643393e-08, -1.61126769e-10,  1.88143679e-13,
       -4.51438853e-17, -1.57669641e-19,  2.04078037e-22, -1.03661870e-25,
        1.99387855e-29,  4.79770068e+06,  3.96519083e-03, -4.25467032e-04,
        1.37349321e-05, -3.49939622e-07,  5.15984884e-09, -4.33175181e-11,
        2.19532675e-13, -6.97387319e-16,  1.39838831e-18, -1.71850940e-21,
        1.18133910e-24, -3.47750488e-28,  2.11655537e-03, -2.88468110e-02,
        1.29131928e-03, -2.16400529e-05,  1.78759587e-07, -8.48105663e-10,
        2.50629548e-12, -4.81229383e-15,  6.09911955e-18, -5.05978692e-21,
        2.64067345e-24, -7.85931865e-28,  1.01646842e-31,  2.11654549e-03,
        8.57846997e-04, -4.04969648e-06,  5.13738992e-09, -1.34362198e-12,
        8.74739641e-18,  2.11654741e-03,  8.34595522e-02, -1.26464838e-02,
        2.11655018e-03,  9.14743843e-03, -8.68058331e-05,  3.80433465e-07,
       -8.68438034e-10,  1.16940095e-12, -9.94795232e-16,  5.54021081e-19,
       -2.04466183e-22,  4.94858602e-26, -7.53989024e-30,  6.55180192e-34,
       -2.47325341e-38,  2.11654913e-03, -1.20427463e-03,  3.58998126e-03,
        2.11654904e-03, -9.67356853e-04, -4.47311616e-03, -7.18238483e-03,
       -4.67835844e-03,  1.03599690e-02, -4.96190745e-03,  1.15444418e-03,
       -1.53785850e-04,  1.23279896e-05, -5.88358406e-07,  1.54052692e-08,
       -1.70340793e-10,  2.11654860e-03,  2.87036299e-04,  7.06115244e-03,
        5.41549976e-03,  7.28305967e-03,  8.28379251e-03,  6.20449822e-03,
        9.59953392e-03,  3.90062959e-03,  7.00246420e-03,  6.41404543e-04,
       -6.26503474e-03, -1.26312190e-03,  2.11654860e-03,  4.72091808e-02,
       -3.58106456e-01,  2.11654860e-03,  1.74114598e-02, -7.71593593e-05,
        3.21169682e-06, -1.17957372e-07,  1.60265780e-09, -1.11997351e-11,
        4.53755213e-14, -1.10558138e-16,  1.59300962e-19, -1.24673305e-22,
        4.07212132e-26,  2.11654860e-03,  2.47585268e-03, -3.28927326e-04,
       -1.64898078e-02,  2.11654893e-03, -8.84954302e-04,  2.31009206e-02,
        2.31248452e-05,  2.11654893e-03,  2.87005713e-03, -6.62665519e-02,
       -4.38311565e-04])

DATA_TEST_PATH = 'data/test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_test_0 = build_poly(tX_test,0)
for i in  id_min_loss:
    deg = int(min_deg[i])
    tX_test_0 = np.append(tX_test_0, build_poly(tX_test[:,i], deg, linear=True),1)


OUTPUT_PATH = 'data/output.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tX_test_0)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

