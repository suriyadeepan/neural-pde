from scipy.io import loadmat
import numpy as np
import pandas as pd


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
