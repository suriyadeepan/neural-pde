import numpy as np
import pandas as pd

from scipy.io import loadmat
from pyDOE import lhs as latin_hyper_cube


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


def schrodinger_constraints():
  df, bounds = schrodinger()
  # [1] initial conditions
  # (0, x)
  x_len, t_len = len(df.x.unique()), len(df.t.unique())
  # shuffle (0, 512)
  idx_x = np.random.choice(x_len, x_len, replace=False)
  initial = pd.DataFrame({
    'x' : df.x[idx_x], 't' : np.zeros_like(df.x),
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
  # return constraints
  return initial, boundary, collocation
