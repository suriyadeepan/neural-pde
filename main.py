from torch.autograd import Variable
from neuralpde import dhpm
from neuralpde import data
from neuralpde import nnutils
from neuralpde.nnutils import torch_em

import numpy as np
import torch


def validate_idn_net():
  # specs for sub-networks
  uv_layers = [2, 50, 50, 50, 50, 1]
  pde_layers = [6, 100, 100, 1]
  # define bounds
  bounds = { 'x' : (-5., 5.), 't' : (0., np.pi / 2) }
  # Instantiate Identification Net
  idn_net = dhpm.IDNnet(uv_layers, pde_layers, bounds)
  # make random inputs
  x = Variable(torch.randn(100,), requires_grad=True)
  t = Variable(torch.randn(100,), requires_grad=True)
  # forward
  u, v = idn_net.uv_net(t, x)
  f, g = idn_net.fg_net(t, x)
  print('f[0]', f[0])
  print('g[0]', g[0])
  # run prediction
  u, v, f, g = idn_net.predict(t, x)
  print('u[0], v[0], f[0], g[0]', u[0], v[0], f[0], g[0])
  # calculate UV loss
  u_truth, v_truth = torch.unbind(torch.randn(2, 100))
  uv_loss_ = idn_net.uv_loss(u, u_truth, v, v_truth)
  fg_loss_ = idn_net.fg_loss(f, g)
  print('uv loss', uv_loss_)
  print('fg loss', fg_loss_)
  # train UV net
  #  make random trainset
  u, v = torch.unbind(torch.randn(2, 100))
  idn_net.train_uv_net(t, x, u, v)


if __name__ == '__main__':
  # validate model with random inputs
  # validate_idn_net()
  # we need data
  #  - read dynamics data from .mat file
  #  - sample N data points
  N = 10000
  schrod_df, bounds = data.schrodinger(N=N)
  # specs for sub-networks
  uv_layers = [2, 50, 50, 50, 50, 1]
  pde_layers = [6, 100, 100, 1]
  # instantiate identification net
  idn_net = dhpm.IDNnet(uv_layers, pde_layers, bounds)
  # make trainable data
  t, x, u, v = torch_em(schrod_df.t, schrod_df.x, schrod_df.u, schrod_df.v)
  t, x = nnutils.variable(t, x)
  # train uv sub-network
  idn_net.train_uv_net(t, x, u, v)
