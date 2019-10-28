# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math
from implementations import *
from proj1_helpers import *
from misc_helpers import *
from plot_functions import *
from ml_math import *
%load_ext autoreload
%autoreload 2

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
DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#Over sample
y1 = y[y==1.]
X1 = tX[y==1.,:]
y = np.append(y,y1)
tX = np.append(tX,X1,axis=0)

#change y
y = (y+1)/2
# normalize X
tX = normalize(tX)

model = np.array([[ 0.,  0.],
       [ 0.,  1.],
       [ 1.,  1.],
       [ 1.,  2.],
       [ 2.,  1.],
       [ 2.,  2.],
       [ 2.,  3.],
       [ 3.,  1.],
       [ 3.,  2.],
       [ 4.,  1.],
       [ 4.,  2.],
       [ 5.,  1.],
       [ 7.,  1.],
       [ 8.,  1.],
       [10.,  1.],
       [10.,  2.],
       [11.,  2.],
       [13.,  1.],
       [13.,  3.],
       [14.,  2.],
       [15.,  2.],
       [16.,  2.],
       [17.,  2.],
       [18.,  3.]])

weights = np.array([ 2.83007006e-01,  2.49392270e+02, -3.05333493e+02, -2.63708162e+02,
       -1.40522448e+02, -3.40884141e+02, -3.48412200e+02,  2.05541339e+02,
        1.21257943e+02, -7.02681862e+01,  7.15839251e+01,  2.12093744e+02,
        2.50868028e+02, -1.14420804e+02, -2.76352098e+02, -1.96605992e+02,
        3.05462512e+02,  2.84946992e+02, -1.48135429e+02, -3.36488342e+02,
       -3.49648218e+01, -3.81936723e+00, -3.39766665e+02,  3.39902283e+02])

#Now Build the tX train data set
#start with bias
tX_model_test = build_poly(tX_test[:,1],0,linear ='True')
tX_test = normalize(tX_test)
#create model
for feat, deg in model:
    feat = int(float(feat.item()))
    print(feat,deg)
    if deg == 'arctan':
        tX_model_test = np.append(tX_model_test, np.array([np.arctan(tX_test[:,feat])]).T,axis=1)
    elif deg == 'cos':
        tX_model_test = np.append(tX_model_test, np.array([np.cos(tX_test[:,feat])]).T,axis=1)
    elif deg =='sin':
        tX_model_test = np.append(tX_model_test, np.array([np.sin(tX_test[:,feat])]).T,axis=1)
    else :
        deg = float(deg)
        if deg == 0.5 or deg == 1/3 or deg == -1/2 or deg == -1/3:
            tX_model_test = np.append(tX_model_test, np.array([np.abs(tX_test[:,feat])**deg]).T,axis=1)
        elif deg != 0:
            tX_model_test = np.append(tX_model_test, np.array([tX_test[:,feat]**deg]).T,axis=1)
np.shape(tX_model_test)

OUTPUT_PATH = 'data/output.csv' # TODO: fill in desired name of output file for submission
y_pred = predict_labels(weights, tX_model_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

