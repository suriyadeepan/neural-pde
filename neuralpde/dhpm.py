from neuralpde import nnutils

import torch
import numpy as np


def tx_norm(t, x, bounds):
  tx = torch.stack([t, x]).transpose(1, 0)
  lower = torch.tensor([bounds['t'][0], bounds['x'][0]])
  upper = torch.tensor([bounds['t'][1], bounds['x'][1]])
  return 2. * (tx - lower) / (upper - lower) - 1.


class IDNnet:

  def __init__(self, uv_layers, pde_layers, bounds):
    self.bounds = bounds
    self.u_net = nnutils.NeuralNet(uv_layers)
    self.v_net = nnutils.NeuralNet(uv_layers)
    self.pde_u_net = nnutils.NeuralNet(pde_layers)
    self.pde_v_net = nnutils.NeuralNet(pde_layers)
    # optimizers
    self.lbfgs_optim_uv = torch.optim.LBFGS(
        nnutils.chain_params(self.u_net, self.v_net),
        # max_iter=50000,
        # max_eval=50000,
        tolerance_change=1.0 * np.finfo(float).eps)
    self.adam_optim_uv = torch.optim.Adam(
        nnutils.chain_params(self.u_net, self.v_net))
    # self.lbfgs_optim.step()

  def uv_net(self, t, x):
    x_norm = tx_norm(t, x, self.bounds)
    u, v = self.u_net(x_norm).reshape(-1), self.v_net(x_norm).reshape(-1)
    return u, v

  def pde_net(self, *args):
    inputs = torch.stack(args).transpose(1, 0)
    pde_u = self.pde_u_net(inputs).reshape(-1)
    pde_v = self.pde_v_net(inputs).reshape(-1)
    return pde_u, pde_v

  def fg_net(self, t, x):
    # get u, v
    u, v = self.uv_net(t, x)
    # get first derivatives
    u_t = nnutils.jacobian(u, t)
    v_t = nnutils.jacobian(v, t)
    u_x, u_xx = nnutils.hessian(u, x)
    v_x, v_xx = nnutils.hessian(v, x)
    pde_u, pde_v = self.pde_net(u, v, u_x, v_x, u_xx, v_xx)

    # TODO : rewrite/rename this abomination!!
    f = u_t - pde_u
    g = v_t - pde_v

    return f, g

  def predict(self, t, x):
    u, v = self.uv_net(t, x)
    f, g = self.fg_net(t, x)
    return u, v, f, g

  def uv_loss(self, u_pred, u_truth, v_pred, v_truth):
    return ((u_pred - u_truth)**2).sum() + ((v_pred - v_truth)**2).sum()

  # TODO : rename this abomination ffs!!
  def fg_loss(self, f, g):
    return (f**2).sum() + (g**2).sum()

  def train_uv_net(self, *trainset):

    def closure():
      # get inputs
      t, x, u, v = trainset
      # zero grad
      self.lbfgs_optim_uv.zero_grad()
      # run prediction
      u_pred, v_pred, f_pred, g_pred = self.predict(t, x)
      # calculate loss
      uv_loss_ = self.uv_loss(u_pred, u, v_pred, v)
      # backward 
      uv_loss_.backward()
      # return loss
      return uv_loss_

    self.lbfgs_optim_uv.step(closure)
