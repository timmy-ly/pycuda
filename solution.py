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
    self.fields = None
      
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
  
  def dx(self, Lx=None, Nx=None):
    if Lx is not None:
      Lx = Lx
    else:
      Lx = self.Lx
    if Nx is not None:
      Nx = Nx
    else:
      Nx = self.Nx
    return Lx/Nx

  def dy(self, Ly=None, Ny=None):
    if Ly is not None:
      Ly = Ly
    else:
      Ly = self.Ly
    if Ny is not None:
      Ny = Ny
    else:
      Ny = self.Ny
    return Ly/Ny
  
  # 2d arrays of y and x
  def coordinates(self, Lx=None, Ly=None, Nx=None, Ny=None):
    if Lx is not None:
      Lx = Lx
    else:
      Lx = self.Lx
    if Nx is not None:
      Nx = Nx
    else:
      Nx = self.Nx
    if Ly is not None:
      Ly = Ly
    else:
      Ly = self.Ly
    if Ny is not None:
      Ny = Ny
    else:
      Ny = self.Ny
    return np.meshgrid(np.arange(Ny)*Ly/Ny, np.arange(Nx)*Lx/Nx)
  # density
  def mean(self):
    # if(self.fields==None):
    #   raise TypeError('fields is None, check if data is properly read. ')
    # else:
    # sum over first and second axis (x and y axis)
    mean = np.sum(self.fields, axis = (1,2))/self.Nx/self.Ny
    return mean















