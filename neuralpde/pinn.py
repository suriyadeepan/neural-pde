from neuralpde.nnutils import jacobian
from neuralpde import nnutils

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm


class PiDiscoveryNet:

  def __init__(self, layers, bounds,
      lbfgs_max_iter=10, lbfgs_max_eval=12):
    self.bounds = bounds
    self.net = nnutils.NeuralNet(layers)
    self.lambda_1 = nn.Parameter(torch.zeros(1,) + 0.001)
    self.lambda_2 = nn.Parameter(torch.zeros(1,) + 0.001)
    # list of parameters
    self.params = [ 
        self.lambda_1, self.lambda_2 ] + list(self.net.parameters())
    # L-BFGS optimizer
    self.lbfgs_optim = torch.optim.LBFGS(self.params,
        max_iter=lbfgs_max_iter, max_eval=lbfgs_max_eval,
        tolerance_change=1.0 * np.finfo(float).eps)
    # Adam optimizer
    self.adam = torch.optim.Adam(self.params)

  def __call__(self, x, y, t):
    # create torch variables from tensors
    x, y, t = nnutils.tvs(x, y, t, shape=(-1, 1))
    # run neural network given (x, y, t)
    psi, p = self.net(torch.cat([x, y, t], dim=-1)).unbind(dim=-1)
    psi, p = psi.view(-1, 1), p.view(-1, 1)
    # calculate first derivative
    u, u_y = jacobian(psi, y, hess=True)
    # v, v_x = jacobian(psi, x, hess=True)
    v = -jacobian(psi, x)
    v_x = jacobian(v, x)
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
    f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + \
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

  def train(self, trainset, evalset, epochs=20000, batch_size=512):
    # unpack trainset, evalset
    x, y, t, u, v, p = trainset
    xe, ye, te, ue, ve, pe = evalset
    # make variables
    x, y, t = nnutils.tvs(x, y, t, shape=x.size(0))
    xe, ye, te = nnutils.tvs(xe, ye, te, shape=xe.size(0))
    # repack trainset and evalset
    trainset, evalset = (x, y, t, u, v, p), (xe, ye, te, ue, ve, pe)
    # define closures for optimizers
    def get_closure(trainset, optim, random=False, batch_size=512):
      def closure():
        if random:  # select a random batch
          x, y, t, u, v, p = nnutils.rand_batch(*trainset,
              batch_size=batch_size)
        else:       # entire training set
          x, y, t, u, v, p = trainset
        # clear gradients
        optim.zero_grad()
        # run prediction
        u_pred, v_pred, p_pred, f_u, f_v = self(x, y, t)
        # calculate loss
        loss = self.loss_fn(
            prediction=(u_pred, v_pred, f_u, f_v), groundtruth=(u, v))
        loss.backward()  # backward
        return loss
      return closure

    # get closure fn for Adam optimizer
    adam_closure = get_closure(trainset, self.adam,
        random=True, batch_size=batch_size)
    pbar = tqdm(range(epochs))
    # run Adam steps `epochs` times
    for t in pbar:
      self.adam.step(adam_closure)
      # calculate errors
      _, err_str = self.evaluate(evalset)
      # add to progress bar
      pbar.set_description(err_str)

    # get closure fn for LBFGS optimization
    lbfgs_closure = get_closure(trainset, self.lbfgs_optim)
    self.lbfgs_optim.step(lbfgs_closure)
    _, err_str = self.evaluate(evalset)
    print('Error Measures : ', err_str)

  def evaluate(self, evalset):
    # unpack evalset
    xe, ye, te, ue, ve, pe = evalset
    # run prediction
    u_pred, v_pred, p_pred, f_u, f_v = self(xe, ye, te)
    # calculate errors
    u_error = torch.norm(ue - u_pred, p=2) / torch.norm(ue, p=2)
    v_error = torch.norm(ve - v_pred, p=2) / torch.norm(ve, p=2)
    p_error = torch.norm(pe - p_pred, p=2) / torch.norm(pe, p=2)
    # errors in lambda
    lambda_1_error = (torch.abs(self.lambda_1 - 1.) * 100).item()
    lambda_2_error = (torch.abs(self.lambda_2 - 0.01)/0.01 * 100).item()
    # summary of errors
    err_str = f'(u : {u_error:3.2f}, v : {v_error:3.2f}, p : {p_error:3.2f})'
    err_str += f' (l1 : {lambda_1_error:.2f}%, l2 : {lambda_2_error:.2f}%)'

    return { 'u' : u_error, 'v' : v_error, 'p' : p_error,
        'lambda_1' : lambda_1_error,
        'lambda_2' : lambda_2_error }, err_str
