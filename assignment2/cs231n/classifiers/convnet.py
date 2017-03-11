import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ConvNet(object):
  """
  A Convolutional Neural Network with the following architecture:
  
  (conv - batchnorm - relu - pool) * M - (affine - batchnorm) * N - affine - softmax
  """
  def __init__(self, input_dim=(3,32,32), num_filters=None, filter_size=None,
               hidden_dim=None, num_classes=10, weight_scale=1e-3, reg=0.0,
              M=3, N=2,dtype=np.float32):
    
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.M = M
    self.N = N
    
    if hidden_dim is None:
      hidden_dim = [100]*N
      
    if num_filters is None:
      num_filters = [32]*M
      filter_size = [3]*M
    
    if len(hidden_dim) != N or len(num_filters) != M or len(filter_size) != M:
      print 'Network architecture errors, return'  
      return
      
    conv_params = [{'stride': 1, 'pad': (filter_size[i]-1)/2} for i in xrange(M)]
    bn_params = [{'mode':'train'} for i in xrange(M)]
    pool_params = [{'pool_height':2, 'pool_width':2, 'stride':2} for i in xrange(M)]
    
    self.conv_params = conv_params
    self.pool_params = pool_params
    self.bn_params = bn_params
    
    C, H, W = input_dim
    
    # Initialize weights of conv layers #####################################
    
    self.params['W_C_0'] = np.random.randn(num_filters[0], C, filter_size[0], filter_size[0])*weight_scale
    self.params['b_C_0'] = np.zeros(num_filters[0])
    self.gamma, self.beta = {}, {}
    self.gamma['C_0'] = np.ones(num_filters[0])
    self.beta['C_0'] = np.zeros(num_filters[0])
    for i in xrange(1, M):
      weight = 'W_C_%d' % i
      bias = 'b_C_%d' % i
      self.params[weight] = np.random.randn(num_filters[i], num_filters[i-1], filter_size[i], filter_size[i])*weight_scale
      self.params[bias] = np.zeros(num_filters[i])
      self.gamma['C_%d' % i] = np.ones(num_filters[i])
      self.beta['C_%d' % i] = np.zeros(num_filters[i])
      
    # Implement fully-connected layers as conv layers #######################
    
    fc_params = [{'stride':1, 'pad': 0} for i in xrange(N+1)]
    fc_bn_params = [{'mode':'train'} for i in xrange(N)]
    self.fc_params = fc_params
    self.fc_bn_params = fc_bn_params
    H_out = H/2**M
    W_out = W/2**M
    
    self.params['W_F_0'] = np.random.randn(hidden_dim[0], num_filters[-1], H_out, W_out)*weight_scale
    self.params['b_F_0'] = np.zeros(hidden_dim[0])
    self.gamma['F_0'] = np.ones(hidden_dim[0])
    self.beta['F_0'] = np.zeros(hidden_dim[0])
    for i in xrange(1, N):
      weight = 'W_F_%d' % i
      bias = 'b_F_%d' % i
      self.params[weight] = np.random.randn(hidden_dim[i], hidden_dim[i-1], 1, 1)*weight_scale
      self.params[bias] = np.zeros(hidden_dim[i])
      self.gamma['F_%d' % i] = np.ones(hidden_dim[i])
      self.beta['F_%d' % i] = np.zeros(hidden_dim[i])
      
    self.params['W_F_%d'%N] = np.random.randn(num_classes, hidden_dim[-1], 1, 1)*weight_scale
    self.params['b_F_%d'%N] = np.zeros(num_classes)
      
    #########################################################################
      
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
      
      
  def loss(self, X, y=None):
    
    # Forward Propagation ###################################################
    
    M, N = self.M, self.N
    conv_params, bn_params, pool_params, fc_params, fc_bn_params = self.conv_params, self.bn_params, self.pool_params, self.fc_params, self.fc_bn_params
    params = self.params
    gamma, beta = self.gamma, self.beta
    
    conv_layer_cache = []
    for i in xrange(M):
      weight = 'W_C_%d' % i
      bias = 'b_C_%d' % i
      X, cache = conv_batchnorm_relu_pool_forward(X, params[weight], params[bias], conv_params[i], gamma['C_%d'%i], beta['C_%d'%i], bn_params[i], pool_params[i])
      conv_layer_cache.append(cache)
    
    fc_layer_cache = []
    for i in xrange(N):
      weight = 'W_F_%d' % i
      bias = 'b_F_%d' % i
      X, cache = conv_batchnorm_forward(X, params[weight], params[bias], fc_params[i], gamma['F_%d'%i], beta['F_%d'%i], 
                                       fc_bn_params[i])
      fc_layer_cache.append(cache)
    
    weight = 'W_F_%d' % N
    bias = 'b_F_%d' % N
    X, cache = conv_forward_fast(X, params[weight], params[bias], fc_params[N])
    fc_layer_cache.append(cache)
    scores = X.reshape((X.shape[0], -1))
    if y is None:
      return scores
    
    # Backward Propagation ###################################################
    
    loss, grads = 0, {}
    loss, dscore = softmax_loss(scores, y)
    dx = dscore.reshape(X.shape)
    
    dx, grads[weight], grads[bias] = conv_backward_fast(dx, fc_layer_cache[-1])
    loss += 0.5*self.reg*(np.sum(params[weight])**2)
    fc_layer_cache.pop()
    
    for i in reversed(xrange(N)):
      weight = 'W_F_%d' % i
      bias = 'b_F_%d' % i
      loss += 0.5*self.reg*(np.sum(params[weight]**2))
      dx, grads[weight], grads[bias] = conv_batchnorm_backward(dx, fc_layer_cache[-1])
      grads[weight] += self.reg * params[weight]
      fc_layer_cache.pop()
    
    for i in reversed(xrange(M)):
      weight = 'W_C_%d' % i
      bias = 'b_C_%d' % i
      loss += 0.5*self.reg*(np.sum(params[weight]**2)) # gamma, beta
      dx, grads[weight], grads[bias] = conv_batchnorm_relu_pool_backward(dx, conv_layer_cache[-1])
      grads[weight] += self.reg * params[weight]
      conv_layer_cache.pop()
      
    return loss, grads
      
    
      
      
      
      
      
      
      