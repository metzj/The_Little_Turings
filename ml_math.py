# Some loss functions, gradients, and ML-related math
# Rémi Clerc, Jordan Metz, Jonas Morin
#######################################################################

import numpy as np


####################### Loss functions #######################

def compute_loss(y, tx, w, loss_function):
    """
    ----------------------------------------------------------------------------
    General function to be called to calculate the loss with a loss_function 
    parameter that can either be "MSE", "MAE", "LL" or "LL2" (log likelyhood)
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    - loss_function choose the loss_function to use
    Output:
    - loss          loss computed, scalar
    ----------------------------------------------------------------------------
    """
    loss = []
    
    # choose correct loss_function
    if loss_function == "MSE":
        loss = compute_MSE(y, tx, w)
    elif loss_function == "MAE":
        loss = compute_MAE(y, tx, w)
    elif loss_function == "LL":
        loss = compute_LL(y, tx, w)
    elif loss_function == "LL2":
        loss = compute_LL2(y, tx, w)
    else:
        # error handling
        print("Loss function ", loss_function, "is not implemented")
        raise NotImplementedError
        
    return loss
        
def compute_MSE(y, tx, w):
    """
    ----------------------------------------------------------------------------
    Function that computes the loss via mean-square error (MSE).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    Output:
    - loss          loss computed, scalar
    ----------------------------------------------------------------------------
    """
    # define the error vector
    e = y - tx.dot(w)
    N = len(y)
    
    # calculate the loss
    MSE_loss = (1/(2*N))*np.transpose(e).dot(e)
    
    return MSE_loss

def compute_MAE(y, tx, w):
    """
    ----------------------------------------------------------------------------
    Function that computes the loss via mean absolute error (MAE).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    Output:
    - loss          loss computed, scalar
    ----------------------------------------------------------------------------
    """
    # define the error vector
    e = y - tx.dot(w)
    N = len(y)
    
    # calculate the loss
    loss = 1/(2*N)*sum(abs(e))

    return loss

def compute_LL(y, tx, w):
    """
    ----------------------------------------------------------------------------
    Function that computes the loss via negative log likelihood (LL).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    Output:
    - loss          loss computed, scalar
    ----------------------------------------------------------------------------
    """
    # define the sigmoid
    pred = sigmoid(tx.dot(w))
    
    # compute the likelyhood
    likelyhood = y.transpose().dot(np.log(pred)) + (1 - y).transpose().dot(np.log(1 - pred))
    
    return -likelyhood

def compute_LL2(y, tx, w):
    """
    ----------------------------------------------------------------------------
    Function that computes the loss for logistic regression using an alternative
    method that makes tx*w vector take only -1 and 1 values.
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    Output:
    - loss          loss computed, scalar
    ----------------------------------------------------------------------------
    """
    # define the error vector
    e = np.where(tx.dot(w)<0,-1,1)
    e = np.abs(y - e)/(2*len(e))
    
    # calculate the loss
    loss = np.sum(e)
    
    return loss

################### gradient computation ######################

def compute_gradient(y, tx, w, loss_function):
    """
    ----------------------------------------------------------------------------
    General function to be called to compute the gradient with a loss_function
    parameter that can either be "MSE", "MAE", "LL" or "LL2".
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    - loss_function choose the loss_function to use
    Output:
    - gradient      gradient, (tx.shape[0],1) np.array
    ----------------------------------------------------------------------------
    """
    gradient = []
    
    
    # choose correct loss_function
    if loss_function == "MSE":
        gradient = compute_gradient_MSE(y, tx, w)
    elif loss_function == "MAE":
        gradient = compute_gradient_MAE(y, tx, w)
    elif loss_function == "LL" or loss_function == "LL2":
        gradient = compute_gradient_LL(y, tx, w)
    else:
        # error handling
        raise NotImplementedError
          
    return gradient
        

def compute_gradient_MSE(y, tx, w):
    """
    ----------------------------------------------------------------------------
    Function that computes the gradient via mean-square error (MSE).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    Output:
    - gradient      gradient, (tx.shape[0],1) np.array
    ----------------------------------------------------------------------------
    """
    # define the error vector
    e = y - tx.dot(w)
    N = len(y)
    
    # computes gradient
    gradient = (-1/N)*tx.transpose().dot(e)

    return gradient

def compute_gradient_MAE(y, tx, w):
    """
    ----------------------------------------------------------------------------
    Function that computes the gradient via mean absolute error (MAE).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    Output:
    - gradient      gradient, (tx.shape[0],1) np.array
    ----------------------------------------------------------------------------
    """
    # define the error vector
    e = y - tx.dot(w)
    N = len(y)
        
    # computes gradient
    sign_e = (e>=0)*2-1
    gradient = -1/N*np.transpose(sign_e).dot(tx)

    return gradient

def compute_gradient_LL(y, tx, w):
    """
    ----------------------------------------------------------------------------
    Function that computes the gradient via negative log likelihood (LL).
    ----------------------------------------------------------------------------
    Input:
    - y             "measured" objective function, (nsamples,1) np.array
    - tx            features, (nsamples,nfeatures) np.array
    - w             calculated weights
    Output:
    - gradient      gradient, (tx.shape[0],1) np.array
    ----------------------------------------------------------------------------
    """
    # define the sigmoid
    prediction = sigmoid(tx.dot(w))
    
    # compute the gradient
    gradient = tx.transpose().dot(prediction - y)
    
    return gradient

######################## misc ######################
def sigmoid(t):
    """
    ----------------------------------------------------------------------------
    Function that computes the value of the sigmoid function for a parameter 
    "t".
    ----------------------------------------------------------------------------
    Input:
    - t          parameter, (nsamples,nfeatures) np.array
    Output:
    - sigma(t)   sigma=1/(1+exp(-t)), (tx.shape[0],1) np.array
    ----------------------------------------------------------------------------
    """
    return 1.0 / (1.0 + np.exp(-t))

# normalizing function courtesy of Cloistered Monkey
# https://necromuralist.github.io/neural_networks/posts/normalizing-with-numpy
def normalize(x: np.ndarray):
    """
    ----------------------------------------------------------------------------
    Function that normalizes each columns of the matrix x to have unit length.
    ----------------------------------------------------------------------------
    Input:
    - x         A numpy matrix of shape (n, m)

    Returns:
    - x         The normalized (by row) numpy matrix
    """
    return x/np.linalg.norm(x, ord=2, axis=0, keepdims=True)