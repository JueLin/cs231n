import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_example = len(y)
  num_feat, num_class = W.shape
  
  
  for i in range(num_example):
    
    score = X[i].dot(W)
    score -= np.max(score) # Avoid numerical instatbility
    
    exp_score = np.exp(score)
    denom = np.sum(exp_score)
    inv_denom = 1./denom
    num = exp_score[y[i]]
    prob = num*inv_denom
    
    loss -= np.log(prob)
    
    d_prob = -1./prob
    d_num = d_prob * inv_denom
    d_inv_denom = d_prob * num
    d_denom = d_inv_denom * (-inv_denom**2 )
    
    for n in range(num_class):
      d_exp_score_n = d_denom * 1.
      d_score_n = d_exp_score_n * exp_score[n]
      
      dW[:, n] += d_score_n * X[i]
        
    d_exp_score_yi = d_num * 1
    d_score_yi = d_exp_score_yi * exp_score[y[i]]
   
    dW[:, y[i]] += d_score_yi * X[i]
  
  loss /= num_example
  loss += 0.5*reg*np.sum(W**2)
  
  dW /= num_example
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # Approach 1: staged computation
  
  #num_example = len(y)
  #num_feat, num_class = W.shape
  
  #score = X.dot(W)
  #score -= np.max(score, axis=1, keepdims=True)
  #exp_score = np.exp(score)
  #num = exp_score[np.arange(num_example), y][:, np.newaxis]
  #denom = np.sum(exp_score, axis=1, keepdims=True)
  #inv_denom = 1./denom
  #prob = num * inv_denom
  
  #loss = np.sum(-np.log(prob))/num_example
  #loss += 0.5*reg*np.sum(W**2)
  
  #d_loss = 1
  #d_prob = -d_loss/prob
  #d_num = d_prob * inv_denom
  #d_inv_denom = d_prob * num
  #d_denom = d_inv_denom * (-inv_denom**2)
  #d_exp_score = np.tile(d_denom, num_class)
  #d_exp_score[np.arange(num_example), y] += d_num.flatten()
  #d_score = d_exp_score * exp_score
  
  #dW = X.T .dot(d_score)
  
  #dW /= num_example
  #dW += reg * W
  
  # Approach 2: simplified 
  num_example = len(y)
  
  score = X.dot(W)
  score -= np.max(score, axis=1, keepdims=True)
  score = np.exp(score)
  score /= np.sum(score, axis=1, keepdims=True)
  
  loss = -np.sum(np.log(score[np.arange(num_example), y]))/num_example
  loss += 0.5*reg*np.sum(W**2)
  
  score[np.arange(num_example), y] -= 1
  
  dW = X.T .dot(score)
  dW /= num_example
  dW += reg * W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

