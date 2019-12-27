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


def train_idn_net():
  # we need data
  #  - read dynamics data from .mat file
  #  - sample N data points
  N = 10000
  schrod_df, bounds = data.schrodinger()
  # sub-sample from data frame
  trainset = schrod_df.sample(N)
  # specs for sub-networks
  uv_layers = [2, 50, 50, 50, 50, 1]
  pde_layers = [6, 100, 100, 1]
  # set training specs
  lbfgs_max_iter, lbfgs_max_eval = 30, 40
  # instantiate identification net
  idn_net = dhpm.IDNnet(uv_layers, pde_layers, bounds,
      lbfgs_max_iter=lbfgs_max_iter,
      lbfgs_max_eval=lbfgs_max_eval)
  # make trainable data
  t_train, x_train, u_train, v_train = torch_em(
      trainset.t, trainset.x, trainset.u, trainset.v)
  t_train, x_train = nnutils.variable(t_train, x_train)
  for i in range(2):  # 30
    # train uv sub-network
    idn_net.train_uv_net(t_train, x_train, u_train, v_train)
    # train fg sub-network
    idn_net.train_fg_net(t_train, x_train)
    # run post-training prediction
    #  gather data from dynamics
    t, x, u, v = torch_em(schrod_df.t, schrod_df.x, schrod_df.u, schrod_df.v)
    t, x, = nnutils.variable(t, x)
    # predict (u, v, f, g)
    u_pred, v_pred, f_pred, g_pred = idn_net.predict(t, x)
    u_error = torch.norm(u - u_pred, p=2) / torch.norm(u, p=2)
    v_error = torch.norm(v - v_pred, p=2) / torch.norm(v, p=2)
    print(f'[{i+1}] Error (u, v) : ({u_error.item()}, {v_error.item()}')

  return idn_net


if __name__ == '__main__':
  # sample N data points
  N = 10000
  schrod_df, bounds = data.schrodinger()
  # sub-sample from data frame
  trainset = schrod_df.sample(N)
  # specs for sub-networks
  uv_layers = [2, 50, 50, 50, 50, 1]
  pde_layers = [6, 100, 100, 1]
  # set training specs
  lbfgs_max_iter, lbfgs_max_eval = (20, 20), (25, 25)
  # DeepHPM < model specs and space-time bounds
  model = dhpm.DeepHPM(uv_layers, pde_layers, bounds)
  # make train set
  t_train, x_train, u_train, v_train = torch_em(  # convert to torch tensors
      trainset.t, trainset.x, trainset.u, trainset.v)
  # make evaluation set
  t_eval, x_eval, u_eval, v_eval = torch_em(      # convert to torch tensors
      schrod_df.t, schrod_df.x, schrod_df.u, schrod_df.v)
  # train IDN net
  model.train_idn_net(
      trainset=(t_train, x_train, u_train, v_train),
      evalset=(t_eval, x_eval, u_eval, v_eval), epochs=1)
