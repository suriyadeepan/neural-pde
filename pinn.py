from neuralpde.pinn import PiDiscoveryNet
from neuralpde.nnutils import torch_em
from neuralpde import data


if __name__ == '__main__':
  N_train, N_eval = 5000, 1000
  # get data < (training data, bounds, raw mat)
  df, bounds, nsw = data.navierstokes_wake(N=N_train)
  layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
  pidn = PiDiscoveryNet(layers, bounds,
      lbfgs_max_iter=10000, lbfgs_max_eval=10000)
  # u, v, p, f_u, f_v = pidn(1., 1., 1.)
  # print(u[0], v[0], p[0], f_u[0], f_v[0])
  # make trainset
  trainset = torch_em(df.x, df.y, df.t, df.u, df.v, df.p)
  # make evaluation set
  edf, _, _ = data.navierstokes_wake(N=N_eval)
  evalset = torch_em(edf.x, edf.y, edf.t, edf.u, edf.v, edf.p)
  pidn.train(trainset, evalset, epochs=200000, batch_size=512)
