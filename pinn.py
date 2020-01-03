from neuralpde.pinn import PiDiscoveryNet
from neuralpde.nnutils import torch_em
from neuralpde import data


if __name__ == '__main__':
  training_set_size = 5000
  # get data < (training data, bounds, raw mat)
  df, bounds, nsw = data.navierstokes_wake(N=training_set_size)
  layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
  pidn = PiDiscoveryNet(layers, bounds)
  # u, v, p, f_u, f_v = pidn(1., 1., 1.)
  # print(u[0], v[0], p[0], f_u[0], f_v[0])
  # make trainset
  trainset = torch_em(df.x, df.y, df.t, df.u, df.v, df.p)
  pidn.train(trainset, epochs=1)
