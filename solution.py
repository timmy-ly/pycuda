import numpy as np

class solution:
  def __init__(self):
    #
    self.OneFName = 'test'
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
    self.nof = 4
      
  @property
  def OneFName(self):
    return self._OneFName

  @OneFName.setter
  def OneFName(self, value):
    if(value.endswith(('.bin','.dat'),-4)):
      value = value[:-4]
    self._OneFName = value

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, value):
    self._dtype = value

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