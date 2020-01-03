from neuralpde.nnutils import jacobian
from neuralpde import nnutils

import torch
import torch.nn as nn
import numpy as np


class PiDiscoveryNet:

  def __init__(self, layers, bounds,
      lbfgs_max_iter=10, lbfgs_max_eval=12):
    self.bounds = bounds
    self.net = nnutils.NeuralNet(layers)
    self.lambda_1 = nn.Parameter(torch.zeros(1,))
    self.lambda_2 = nn.Parameter(torch.zeros(1,))
    # (FG Sub-network) L-BFGS optimizer
    self.lbfgs_optim = torch.optim.LBFGS(
        [ self.lambda_1, self.lambda_2 ] + list(self.net.parameters()),
        max_iter=lbfgs_max_iter, max_eval=lbfgs_max_eval,
        tolerance_change=1.0 * np.finfo(float).eps)

  def __call__(self, x, y, t):
    # create torch variables from tensors
    x, y, t = nnutils.tvs(x, y, t, shape=(-1, 1))
    # run neural network given (x, y, t)
    psi, p = self.net(torch.cat([x, y, t], dim=-1)).unbind(dim=-1)
    psi, p = psi.view(-1, 1), p.view(-1, 1)
    # calculate first derivative
    u, u_y = jacobian(psi, y, hess=True)
    v, v_x = jacobian(psi, x, hess=True)
    u_x, u_xx = jacobian(u, x, hess=True)
    v_y, v_yy = jacobian(v, y, hess=True)
    #   w.r.t `t`
    u_t = jacobian(u, t)
    v_t = jacobian(v, t)
    # remaining calculate second derivatives
    v_xx = jacobian(v_x, x)
    u_yy = jacobian(u_y, y)
    # w.r.t `p`
    p_x = jacobian(p, x)
    p_y = jacobian(p, y)
    # calculate f, g from eqn(6) pg.6
    f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + \
        p_x - self.lambda_2 * (u_xx + u_yy)
    f_v = v_t + self.lambda_2 * (u * v_x + v * v_y) + \
        p_y - self.lambda_2 * (v_xx + v_yy)

    return u, v, p, f_u, f_v

  def loss_fn(self, prediction, groundtruth):
    # expand items
    u_pred, v_pred, f_u, f_v = prediction
    u_truth, v_truth = groundtruth
    # calculate UV loss
    uv_loss = ((u_pred - u_truth)**2).sum() + ((v_pred - v_truth)**2).sum()
    # calculate f loss
    f_loss = (f_u**2).sum() + (f_v**2).sum()
    loss = (uv_loss + f_loss)  # sum up losses
    return loss

  def train_epoch(self, trainset):

    def closure():
      # get inputs
      x, y, t, u, v, p = trainset
      # zero grad
      self.lbfgs_optim.zero_grad()
      # run prediction
      u_pred, v_pred, p_pred, f_u, f_v = self(x, y, t)
      # calculate loss
      loss = self.loss_fn(
          prediction=(u_pred, v_pred, f_u, f_v), groundtruth=(u, v))
      loss.backward()  # backward
      return loss

    self.lbfgs_optim.step(closure)

  def train(self, trainset, epochs=1):
    # expand trainset
    x, y, t, u, v, p = trainset
    # make variables
    x, y, t, nnutils.tvs(x, y, t, shape=x.size(0))
    # keep track of losses
    losses = []
    for i in range(epochs):
      self.train_epoch((x, y, t, u, v, p))
