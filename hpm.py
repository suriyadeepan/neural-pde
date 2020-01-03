from torch.autograd import Variable
from neuralpde import dhpm
from neuralpde import data
from neuralpde import nnutils
from neuralpde.nnutils import torch_em, tv
from neuralpde.visualize import plot_dynamics

import numpy as np
import argparse
import torch


def to_list(s, type_=int):
  return [ type_(item) for item in s.replace(' ', '').split(',') ]


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


def visualize(args):
  # get bounds from schrodinger data
  df, bounds = data.schrodinger()
  # specs for sub-networks
  uv_layers = to_list(args.uv_layers, int)
  pde_layers = to_list(args.pde_layers, int)
  # set training specs
  model = dhpm.DeepHPM(uv_layers, pde_layers, bounds)  # model specs and space-time bounds
  model.load_subnets(args.model)
  te, xe, ue, ve = torch_em(df.t, df.x, df.u, df.v)
  u, v = model.idn_net.uv_net(te, xe)
  # visualize dynamics
  plot_dynamics(bounds, u, v)


def predict(args, t=None, x=None):
  # get bounds from schrodinger data
  df, bounds = data.schrodinger()
  # specs for sub-networks
  uv_layers = to_list(args.uv_layers, int)
  pde_layers = to_list(args.pde_layers, int)
  # set training specs
  lbfgs_max_iter = to_list(args.lbfgs_max_iter, int)
  lbfgs_max_eval = to_list(args.lbfgs_max_eval, int)
  model = dhpm.DeepHPM(uv_layers, pde_layers, bounds,  # model specs and space-time bounds
            lbfgs_max_iter=lbfgs_max_iter, lbfgs_max_eval=lbfgs_max_eval)
  model.load_subnets(args.model)
  if t is None and x is None:
    te, xe, ue, ve = torch_em(df.t, df.x, df.u, df.v)
    u, v = model.idn_net.uv_net(te, xe)
    u_error = torch.norm(ue - u, p=2) / torch.norm(ue, p=2)
    v_error = torch.norm(ve - v, p=2) / torch.norm(ve, p=2)
    print(f'[ error_u : {u_error.item()} ]')
    print(f'[ error_v : {v_error.item()} ]')
    return

  u, v = model.idn_net.uv_net(tv(t).view(1,), tv(x).view(1,))
  print(f'uv({args.time}, {args.space}) = {u.item()} + i{v.item()}')


def train(args):
  # sample N data points
  N = args.trainset_size
  schrod_df, bounds = data.schrodinger()
  # sub-sample from data frame
  trainset = schrod_df.sample(N)
  # specs for sub-networks
  uv_layers = to_list(args.uv_layers, int)
  pde_layers = to_list(args.pde_layers, int)
  # set training specs
  lbfgs_max_iter = to_list(args.lbfgs_max_iter, int)
  lbfgs_max_eval = to_list(args.lbfgs_max_eval, int)
  model = dhpm.DeepHPM(uv_layers, pde_layers, bounds,  # model specs and space-time bounds
            lbfgs_max_iter=lbfgs_max_iter, lbfgs_max_eval=lbfgs_max_eval)
  # make train set
  t_train, x_train, u_train, v_train = torch_em(  # convert to torch tensors
      trainset.t, trainset.x, trainset.u, trainset.v)
  # make evaluation set
  t_eval, x_eval, u_eval, v_eval = torch_em(      # convert to torch tensors
      schrod_df.t, schrod_df.x, schrod_df.u, schrod_df.v)
  # train IDN net
  epochs = to_list(args.epochs)[0]
  model.train_idn_net(
       trainset=(t_train, x_train, u_train, v_train),
       evalset=(t_eval, x_eval, u_eval, v_eval), epochs=epochs)
  # train PiNN
  #  feed Schrodinger Constraints data
  epochs = to_list(args.epochs)[1]
  model.train_pinn(
      trainset=data.schrodinger_constraints(torched=True),
      evalset=(t_eval, x_eval, u_eval, v_eval), epochs=epochs)
  # After training, save trained models
  model.save_subnets(args.savepath)


# [ .... config .... ]
parser = argparse.ArgumentParser(
    description='neuralbec : Neural Network based simulation of BEC'
    )
parser.add_argument('--train', default=False, action='store_true',
    help='Train DeepHPM model')
parser.add_argument('--model', type=str, default=None, help='Path to Saved Model')
parser.add_argument('--predict', default=False, action='store_true',
    help='Path to Saved Model')
parser.add_argument('--visualize', default=False, action='store_true',
    help='Plot dynamics')
parser.add_argument('-t', '--time', type=float, default=None,
    help='Time step for Prediction')
parser.add_argument('-x', '--space', type=float, default=None,
    help='Space step for Prediction')
parser.add_argument('-N', '--trainset-size', type=int, default=10000,
    help='Training Set size')
parser.add_argument('--uv-layers', type=str, default='2, 50, 50, 50, 50, 1',
    help='Model spec for UV sub-network')
parser.add_argument('--pde-layers', type=str, default='6, 100, 100, 1',
  help='Model spec for PDE sub-network')
parser.add_argument('--lbfgs-max-iter', type=str, default='30, 30',
    help='LBFGS max_iter option')
parser.add_argument('--lbfgs-max-eval', type=str, default='35, 35',
    help='LBFGS max_eval option')
parser.add_argument('--epochs', type=str, default='30, 30',
    help='Number of training epochs')
parser.add_argument('--savepath', type=str, default='saved_models',
    help='Number of training epochs')


if __name__ == '__main__':
  args = parser.parse_args()
  if args.train:          # train model
    train(args)
  elif args.model:        # load model from disk
    if args.predict:
      assert args.space and args.time
      predict(args, args.time, args.space)  # predict (u, v) given (t, x)
    elif args.visualize:
      visualize(args)     # visualize dynamics
    else:
      predict(args)      # evaluate trained model
  else:
    print('Try getting --help')
