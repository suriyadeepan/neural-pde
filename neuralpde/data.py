import numpy as np
import pandas as pd

from scipy.io import loadmat
from pyDOE import lhs as latin_hyper_cube
from neuralpde.nnutils import tv, t


def schrodinger(filepath="data/NLS.mat", N=None):
  # load matrix from file
  mat = loadmat(filepath)
  # read t, x
  t, x = mat['t'], mat['x']
  # make a mesh grid
  t_grid, x_grid = np.meshgrid(t, x) 
  # get u, v (separate out real and imaginary components)
  u = np.real(mat['usol'])
  v = np.imag(mat['usol'])
  # calculate uv
  uv = np.sqrt(u**2 + v**2)
  # flatten t, x, u, v
  t_flat = t_grid.flatten()
  x_flat = x_grid.flatten()
  u_flat = u.flatten()
  v_flat = v.flatten()
  # make a data frame
  schrod_df =  pd.DataFrame({'t' : t_flat, 'x' : x_flat, 'u' : u_flat, 'v' : v_flat })
  bounds = { 'x' : (-5., 5.), 't' : (0., np.pi / 2) }
  if N is not None:  # sample from data frame
    schrod_df = schrod_df.sample(N)

  return schrod_df, bounds


def navierstokes_wake(filepath="data/cylinder_nektar_wake.mat", N=None):
  # load matrix from file
  mat = loadmat(filepath)
  vec = mat['U_star']  # N x 2 x 200
  p = mat['p_star']  # N x 200
  t = mat['t']       # 200
  x = mat['X_star']  # N x 2
  x_len, t_len = x.shape[0], t.shape[0]
  xx = np.tile(x[:, 0].reshape(-1, 1), (1, t_len))
  yy = np.tile(x[:, 1].reshape(-1, 1), (1, t_len))
  tt = np.tile(t, (1, x_len)).T
  u = vec[:, 0, :].reshape(-1)
  v = vec[:, 1, :].reshape(-1)
  nsw_df = pd.DataFrame({
    'x' : xx.reshape(-1), 'y' : yy.reshape(-1),
    't' : tt.reshape(-1), 'p' : p.reshape(-1),
    'u' : u, 'v' : v
    })
  # sample `N` points from data frame
  if N is not None:
    nsw_df = nsw_df.sample(N)
  # construct bounds
  X = np.concatenate(
      [ nsw_df.x.values.reshape(-1, 1), nsw_df.y.values.reshape(-1, 1), 
        nsw_df.t.values.reshape(-1, 1) ], axis=1)
  bounds = (X.min(0), X.max(0))
  return nsw_df, bounds, { 't' : t, 'x' : x, 'p' : p, 'vec' : vec }


def schrodinger_constraints(filepath="data/NLS.mat", torched=False):
  df, bounds = schrodinger()
  # read mat file
  mat = loadmat(filepath)
  u, v = np.real(mat['usol']), np.imag(mat['usol'])
  # [1] initial conditions
  # (0, x)
  x_len, t_len = len(df.x.unique()), len(df.t.unique())
  # shuffle (0, 512)
  idx_x = np.random.choice(x_len, x_len, replace=False)
  initial = pd.DataFrame({
    'x' : df.x[idx_x], 't' : np.zeros_like(df.x[idx_x]),
    'u' : u[idx_x, 0], 'v' : v[idx_x, 0]})
  # [2] boundary conditions
  # shuffle (0, 501)
  idx_t = np.random.choice(t_len, t_len, replace=False)
  tb = df.t[idx_t]
  boundary = pd.DataFrame({
    'x_lb' : bounds['x'][0] + np.zeros_like(tb),
    'x_ub' : bounds['x'][1] + np.zeros_like(tb),
    't_lb' : tb,
    't_ub' : tb
    })
  # [3] Collocation points
  Nco = 20000  # number of points
  lb = np.array([bounds['t'][0], bounds['x'][0]])
  ub = np.array([bounds['t'][1], bounds['x'][1]])
  # sample from latin hypercube
  Xco = lb + (ub - lb) * latin_hyper_cube(2, Nco)
  collocation = pd.DataFrame({ 'x' : Xco[:, 1], 't' : Xco[:, 0] })

  if torched:
    return torch_constraints(initial, boundary, collocation)

  return initial, boundary, collocation


def torch_constraints(initial, boundary, collocation):
  # create torch tensors and variables for training
  #  [1] Initial Data
  initial = {
      't' : tv(initial.t), 'x' : tv(initial.x),  # torch variables
      'u' : t(initial.u), 'v' : t(initial.v)     # torch tensors
      }
  #  [2] Boundary Data
  boundary = {
      't_lb' : tv(boundary.t_lb), 'x_lb' : tv(boundary.x_lb),
      't_ub' : tv(boundary.t_ub), 'x_ub' : tv(boundary.x_ub)
      }
  #  [3] Collocation Points
  collocation = { 't' : tv(collocation.t), 'x' : tv(collocation.x) }
  return initial, boundary, collocation
