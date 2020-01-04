from torch.autograd import Variable
from torch.autograd import grad
import torch.nn as nn
import torch

import numpy as np
import itertools


class NeuralNet(nn.Module):
    
  def __init__(self, layer_sizes):
    super(NeuralNet, self).__init__()
    self.layers = nn.ModuleList(
        [ nn.Linear(layer_sizes[i], layer_sizes[i+1])
          for i in range(len(layer_sizes) - 1) ])
    
  def forward(self, x):
    for layer in self.layers[:-1]:
      x = torch.sin(layer(x))
    return self.layers[-1](x)


def jacobian(output, inputs, wrt=None, hess=False):
  if wrt is None:
    wrt = inputs
  jacob = grad(output, inputs, grad_outputs=wrt, create_graph=True)[0]  # / inputs
  if hess:
    return jacob, grad(jacob, inputs,
        grad_outputs=inputs, create_graph=True)[0]  # / inputs
  return jacob


def hessian(output, inputs, wrt=None):
  return jacobian(output, inputs, wrt=wrt, hess=True)


def chain_params(*models):
  return itertools.chain(*[m.parameters() for m in models])


def torch_em(*ts):
  return [ torch.tensor(t.values).float() for t in ts ]


def variable(*ts, requires_grad=True):
  return [ Variable(t, requires_grad=requires_grad) for t in ts ]


def tv(a, shape=None):
  if not isinstance(a, torch.Tensor):
    a = torch.tensor(a).float()
  tv_ = Variable(a, requires_grad=True)
  if shape is not None:
    tv_ = tv_.view(shape)
  return tv_


def tvs(*a, shape=None):
  return [ tv(ai, shape) for ai in a ]


def t(a):
  return torch.tensor(a).float()


def batch(*args, i, batch_size=100):
  return tuple([ arg[i * batch_size : (i + 1) * batch_size]
      for arg in args ])


def rand_batch(*args, batch_size=200):
  num_batches = len(args[0]) // batch_size
  # pick a batch
  return batch(*args, i=np.random.randint(num_batches),
      batch_size=batch_size)
