import numpy as np

# class for a solution (for some time t) from cuda data
class solution:
  def __init__(self):
    #
    self.OneFName = None
    # numerical
    self.dtype = 'double'
    self.Nx = None
    self.Ny = None
    self.dt = None
    # problem
    self.Lx = None
    self.Ly = None
    self.t = None
    # string by which BC is chosen
    self.BC = None
    # model
    self.nof = None
    self.fields = None
    # dummy field, used when using finite difference methods on fields outside of
    # class object
    self.DummyFields = None
      
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
  
  def dx(self):
    return self.Lx/self.Nx
  def dx2(self):
    return self.dx()*self.dx()
  def dx3(self):
    return self.dx()*self.dx()*self.dx()
  def dy(self):
    return self.Ly/self.Ny
  def dy2(self):
    return self.dy()*self.dy()
  def dy3(self):
    return self.dy()*self.dy()*self.dy()
  
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
  # get crosssection in y
  def get_crosssection_y(self, field, IdxX=None):
    if(IdxX is None):
      IdxX = self.Nx//2
    return field[IdxX]
  # wrapper for numpy argsort
  def ArgSort1DField(self, field, IdxX=None):
    self.Sorted1DIndices = np.argsort(self.get_crosssection_y(field, IdxX))
  # finite difference methods
  # for the stencils
  def f(self,ix,iy):
    fields = self.DummyFields
    # GPRight and GPBottom should have negative signs
    fields, GPTop, GPBottom, GPLeft, GPRight = self.ApplyBC(fields)
    result = np.roll(fields,(-ix,-iy),(-2,-1))
    return result[...,GPTop:GPBottom,GPLeft:GPRight]
  # overwrite this function in ChildClasses
  def ApplyBC(self, fields):
    return fields, None, None, None, None
  # 4th order
  # dy fdm 4th order, forward, copied from cuda code
  def dy4_04(self, FieldsOutsideOfInstance=None, dx=None):
    f = self.f
    if(FieldsOutsideOfInstance is not None):
      self.DummyFields = FieldsOutsideOfInstance
    else:
      self.DummyFields = self.fields
    if(dx is None):
      dx = self.dx()
    return ( -25.0*f(0,0) + 48.0*f(0,1) - 36.0*f(0,2) + 16.0*f(0,3) - 3.0*f(0,4) )/12.0/dx
  # dy fdm 4th order, center, copied from cuda code
  def dy4_m22(self, FieldsOutsideOfInstance=None, dx=None):
    f = self.f
    if(FieldsOutsideOfInstance is not None):
      self.DummyFields = FieldsOutsideOfInstance
    else:
      self.DummyFields = self.fields
    if(dx is None):
      dx = self.dx()
    return (-f(0,2) + 8.0*f(0,1) - 8.0*f(0,-1) + f(0,-2) )/12.0/dx
















