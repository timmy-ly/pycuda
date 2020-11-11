import numpy as np

# precipiti attributes
class AttributesPRECIPITI:
  def __init__(self):
    # numerical
    self.dtype = 'double'
    self.Nx = 256
    self.Ny = 256
    self.dt = 0.001
    # problem
    self.Lx = 256.
    self.Ly = 256.
    self.t = 0.0
    # model
    self.nof = 1
    self.v = 0.2
    self.ups0 = 0.0
    self.chi = 1.5
    self.ups1 = 0.0
    self.ups2 = 1.0
    self.ups3 = -5.0
    self.g = 10**-3
    self.beta = 2.0
    self.lamb = 1.8
    self.LAMB = 1.4
    self.sigma = 1.8
    # model/BC
    self.alpha = 0.0
    # BC
    self.bc = 2
    # BC/IC
    self.h0 = 20.0
    self.c0 = 0.3
    self.phi0 = 1.0
    # IC
    self.noise = 0.0
    self.h1 = 0.0
    # print('hello world')

  @property
  def Nx(self):
    return self._Nx

  @Nx.setter
  def Nx(self, value):
    self._Nx = value

  @property
  def Ny(self):
    return self._Ny

  @Ny.setter
  def Ny(self, value):
    self._Ny = value

  @property
  def Lx(self):
    return self._Lx

  @Lx.setter
  def Lx(self, value):
    self._Lx = value

  @property
  def Ly(self):
    return self._Ly

  @Ly.setter
  def Ly(self, value):
    self._Ly = value

  @property
  def nof(self):
    return self._nof

  @nof.setter
  def nof(self, value):
    self._nof = value

