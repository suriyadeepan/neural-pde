from neuralpde import nnutils
import numpy as np
import torch
import os


def tx_norm(t, x, bounds):
  tx = torch.stack([t, x]).transpose(1, 0)
  lower = torch.tensor([bounds['t'][0], bounds['x'][0]])
  upper = torch.tensor([bounds['t'][1], bounds['x'][1]])
  return 2. * (tx - lower) / (upper - lower) - 1.


class IDNnet:

  def __init__(self, uv_layers, pde_layers, bounds,
      lbfgs_max_iter=20, lbfgs_max_eval=25,
      u_net=None, v_net=None,
      pde_u_net=None, pde_v_net=None):
    self.bounds = bounds
    # setup sub-networks
    self.u_net = nnutils.NeuralNet(uv_layers) if u_net is None else u_net
    self.v_net = nnutils.NeuralNet(uv_layers) if v_net is None else v_net
    self.pde_u_net = nnutils.NeuralNet(pde_layers) if pde_u_net is None else pde_u_net
    self.pde_v_net = nnutils.NeuralNet(pde_layers) if pde_v_net is None else pde_v_net
    # (UV Sub-network) L-BFGS optimizer
    self.lbfgs_optim_uv = torch.optim.LBFGS(
        nnutils.chain_params(self.u_net, self.v_net),
        max_iter=lbfgs_max_iter, max_eval=lbfgs_max_eval,
        tolerance_change=1.0 * np.finfo(float).eps)
    # (FG Sub-network) L-BFGS optimizer
    self.lbfgs_optim_fg = torch.optim.LBFGS(
        nnutils.chain_params(self.pde_u_net, self.pde_v_net),
        max_iter=lbfgs_max_iter, max_eval=lbfgs_max_eval,
        tolerance_change=1.0 * np.finfo(float).eps)
    # Adam Optimizer
    self.adam_optim_uv = torch.optim.Adam(
        nnutils.chain_params(self.u_net, self.v_net))

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

  def train_fg_net(self, t, x):

    def closure():
      # zero grad
      self.lbfgs_optim_fg.zero_grad()
      # run prediction
      u_pred, v_pred, f_pred, g_pred = self.predict(t, x)
      # calculate loss
      fg_loss_ = self.fg_loss(f_pred, g_pred)
      # backward 
      fg_loss_.backward()
      # return loss
      return fg_loss_

    self.lbfgs_optim_fg.step(closure)


class PiNeuralNet:

  def __init__(self, bounds, u_net, v_net, pde_net,
      lbfgs_max_iter=20, lbfgs_max_eval=25):
    self.bounds = bounds
    self.u_net = u_net
    self.v_net = v_net
    self.pde_net = pde_net
    # (UV Sub-network) L-BFGS optimizer
    self.lbfgs_optim_uv = torch.optim.LBFGS(
        nnutils.chain_params(self.u_net, self.v_net),
        max_iter=lbfgs_max_iter, max_eval=lbfgs_max_eval,
        tolerance_change=1.0 * np.finfo(float).eps)

  def uv_net(self, t, x):
    x_norm = tx_norm(t, x, self.bounds)
    u, v = self.u_net(x_norm).reshape(-1), self.v_net(x_norm).reshape(-1)
    u_x, v_x = nnutils.jacobian(u, x), nnutils.jacobian(v, x)
    return u, v, u_x, v_x

  def fg_net(self, t, x):
    # get u, v
    u, v, _, _ = self.uv_net(t, x)
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

  def train(self, initial, boundary, collocation):

    def closure():
      # zero grad
      self.lbfgs_optim_uv.zero_grad()
      # run predictions
      #  [1] initial data
      u0, v0, _, _ = self.uv_net(initial['t'], initial['x'])  # (0, x)
      #  [2] boundary data [ (t, -8), (t, +8) ]
      u_lb, v_lb, u_x_lb, v_x_lb = self.uv_net(boundary['t_lb'], boundary['x_lb'])
      u_ub, v_ub, u_x_ub, v_x_ub = self.uv_net(boundary['t_ub'], boundary['x_ub'])
      #  [3] collocation data
      f, g = self.fg_net(collocation['t'], collocation['x'])
      # calculate loss
      loss_initial = ((u0 - initial['u'])**2).mean() + ((v0 - initial['v'])**2).mean()
      loss_boundary = ((u_lb - u_ub)**2).mean() + ((v_lb - v_ub)**2).mean() + \
          ((u_x_lb - u_x_ub)**2).mean() + ((v_x_lb - v_x_ub)**2).mean()
      loss_collocation = (f**2).mean() + (g**2).mean()
      # sum up
      loss = loss_initial + loss_boundary + loss_collocation
      # backward
      loss.backward()
      return loss

    self.lbfgs_optim_uv.step(closure)

  def predict(self, t, x):
    u, v, _, _ = self.uv_net(t, x)
    f, g = self.fg_net(t, x)
    return u, v, f, g


class DeepHPM:

  def __init__(self, uv_layers, pde_layers, bounds,
      lbfgs_max_iter=(20, 20), lbfgs_max_eval=(25, 25),
      path='saved_models/'):
    self.bounds = bounds
    # initialize sub-networks
    self.u_net = nnutils.NeuralNet(uv_layers)
    self.v_net = nnutils.NeuralNet(uv_layers)
    self.pde_u_net = nnutils.NeuralNet(pde_layers)
    self.pde_v_net = nnutils.NeuralNet(pde_layers)
    # create Identification Network
    self.idn_net = IDNnet(uv_layers, pde_layers, bounds,
        lbfgs_max_iter=lbfgs_max_iter[0],    # lbfgs options
        lbfgs_max_eval=lbfgs_max_eval[0],    # ..
        u_net=self.u_net, v_net=self.v_net,  # share sub-networks
        pde_u_net=self.pde_u_net, pde_v_net=self.pde_v_net)
    # create Physics-informed Neural Network
    self.pinn = PiNeuralNet(bounds,
        u_net=self.u_net, v_net=self.v_net,  # shared sub-networks
        pde_net=self.idn_net.pde_net,        # use method of idn_net
        lbfgs_max_iter=lbfgs_max_iter[1],    # ..
        lbfgs_max_eval=lbfgs_max_eval[1])    # lbfgs options
    # keep track of sub-networks
    self.subnets = {
        'u' : self.u_net, 'v' : self.v_net,
        'pde_u' : self.pde_u_net, 'pde_v' : self.pde_v_net
        }
    # set path
    self.path = path

  def train_idn_net(self, trainset, evalset, epochs=2):
    # expand trainset
    t, x, u, v = trainset
    # make variables
    t, x = nnutils.variable(t, x)
    #  gather eval data
    te, xe, ue, ve = evalset
    # make variables
    te, xe = nnutils.variable(te, xe)
    # keep track of losses
    losses = []
    for i in range(epochs):
      # train uv sub-network
      self.idn_net.train_uv_net(t, x, u, v)
      # train fg sub-network
      self.idn_net.train_fg_net(t, x)
      # run post-training prediction
      #  predict (u, v, f, g)
      u_pred, v_pred, f_pred, g_pred = self.idn_net.predict(te, xe)
      u_error = torch.norm(ue - u_pred, p=2) / torch.norm(ue, p=2)
      v_error = torch.norm(ve - v_pred, p=2) / torch.norm(ve, p=2)
      print(f'[{i+1}] Error (u, v) : ({u_error.item()}, {v_error.item()}')
      losses.append((u_error, v_error))
    # return loss history
    return losses

  def train_pinn(self, trainset, evalset, epochs=2):
    # expand eval set
    te, xe, ue, ve = evalset
    # make variables
    te, xe = nnutils.variable(te, xe)
    for i in range(epochs):
      self.pinn.train(*trainset)
      # run post-trianing prediction
      #  predict (u, v, f, g)
      u_pred, v_pred, f_pred, g_pred = self.pinn.predict(te, xe)
      uv_pred = torch.sqrt(u_pred**2 + v_pred**2)
      uv_e = torch.sqrt(ue**2 + ve**2)
      u_error = torch.norm(ue - u_pred, p=2) / torch.norm(ue, p=2)
      v_error = torch.norm(ve - v_pred, p=2) / torch.norm(ve, p=2)
      uv_error = torch.norm(uv_e - uv_pred, p=2) / torch.norm(uv_e, p=2)
      print(f'[{i+1}] Error (u : {u_error}), (v : {v_error}), (uv : {uv_error})')

  def load_subnets(self, path=None):
    if path is None:
      path = self.path
    for name in self.subnets.keys():
      print(f'Loading {os.path.join(path, name)}.pth')
      self.subnets[name].load_state_dict(
          torch.load(os.path.join(path, f'{name}.pth')))

  def save_subnets(self, path=None):
    if path is None:
      path = self.path
    for name, net in self.subnets.items():
      print(f'Saving {os.path.join(path, name)}.pth')
      torch.save(net.state_dict(), os.path.join(path, f'{name}.pth'))
