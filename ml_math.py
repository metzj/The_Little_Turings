# Some loss functions, gradients, and ML-related math
# Rémi Clerc, Jordan Metz, Jonas Morin
#######################################################################

import numpy as np


####################### Loss functions #######################

#general function to be called with a loss_function parameter
#that can either be "MSE", "MAE" or "LL" (log likelyhood)
def compute_loss(y, tx, w, loss_function):
    loss = []
    
    if loss_function == "MSE":
        loss = compute_MSE(y, tx, w)
    elif loss_function == "MAE":
        loss = compute_MAE(y, tx, w)
    elif loss_function == "LL":
        loss = compute_LL(y, tx, w)
    else:
        print("Loss function ", loss_function, "is not implemented")
        raise NotImplementedError
        
    return loss
        
def compute_MSE(y, tx, w):
    """Calculate the mse loss."""

    e = y - tx.dot(w)
    N = len(y)
    MSE_loss = (1/(2*N))*np.transpose(e).dot(e)
    
    return MSE_loss

def compute_MAE(y, tx, w):
    """Calculate the mae loss."""

    e = y - tx.dot(w)
    N = len(y)
    loss = 1/(2*N)*sum(abs(e))

    return loss

def compute_LL(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    pred = sigmoid(tx.dot(w))
    #print("prediction ", pred)
    likelyhood = y.transpose().dot(np.log(pred)) + (1 - y).transpose().dot(np.log(1 - pred))
    
    return -likelyhood

################### gradient computation ######################

#general function to be called with a loss_function parameter
#that can either be "MSE" or "MAE"
def compute_gradient(y, tx, w, loss_function):
    gradient = []
    
    if loss_function == "MSE":
        gradient = compute_gradient_MSE(y, tx, w)
    elif loss_function == "MAE":
        gradient = compute_gradient_MAE(y, tx, w)
    elif loss_function == "LL":
        gradient = compute_gradient_LL(y, tx, w)
    else:
        raise NotImplementedError
          
    return gradient
        

def compute_gradient_MSE(y, tx, w):
    """Compute the gradient of the MSE loss"""
    e = y - tx.dot(w)
    N = len(y)
    
    gradient = (-1/N)*tx.transpose().dot(e)

    return gradient

def compute_gradient_MAE(y, tx, w):
    """Compute the gradient of the MAE loss"""
    e = y - tx.dot(w)
    N = len(y)
    
    sign_e = (e>=0)*2-1
    gradient = -1/N*np.transpose(sign_e).dot(tx)

    return gradient

def compute_gradient_LL(y, tx, w):
    """compute the gradient of loss."""
    prediction = sigmoid(tx.dot(w))
    gradient = tx.transpose().dot(prediction - y)
    
    return gradient

######################## misc ######################
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))

# normalizing function courtesy of Cloistered Monkey
# https://necromuralist.github.io/neural_networks/posts/normalizing-with-numpy
def normalize(x: np.ndarray):
    """
    function that normalizes each columns of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return x/np.linalg.norm(x, ord=2, axis=0, keepdims=True)