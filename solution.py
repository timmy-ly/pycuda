import numpy as np
from scipy.signal import find_peaks

class field:
  def __init__(self, values):
    self.array = values
  def set_max(self):
    self.max = np.nanmax(self.array)

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
  
  def CheckSetAttr(self, *args):
    # AttributeList is a list of strings or a string
    for arg in args:
      if not hasattr(self, arg):
        getattr(self, "set_" + arg)()

  def dx(self):
    return self.Lx/self.Nx
  def dx2(self):
    return self.dx()*self.dx()
  def dx3(self):
    return self.dx()*self.dx()*self.dx()
  def dx4(self):
    return self.dx()*self.dx()*self.dx()*self.dx()
  def dy(self):
    return self.Ly/self.Ny
  def dy2(self):
    return self.dy()*self.dy()
  def dy3(self):
    return self.dy()*self.dy()*self.dy()
  def dy4(self):
    return self.dy()*self.dy()*self.dy()*self.dy()
  
  # 2d arrays of y and x
  def set_coordinates(self, Lx=None, Ly=None, Nx=None, Ny=None):
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
    self.y2D, self.x2D = np.meshgrid(np.arange(Ny)*Ly/Ny, np.arange(Nx)*Lx/Nx)
    # careful since this overwrites self.y from set_coordinates
  def set_SpatialGrid1Dy(self):
    self.y = np.arange(self.Ny)*self.Ly/self.Ny
  # density
  def mean(self):
    # if(self.fields==None):
    #   raise TypeError('fields is None, check if data is properly read. ')
    # else:
    # sum over first and second axis (x and y axis)
    mean = np.sum(self.fields, axis = (1,2))/self.Nx/self.Ny
    return mean
  # legacy function, just a wrapper now
  def transform_crosssection(self, field):
    return self.get_crosssection_y(field)
  # general crosssection method in y
  def get_crosssection_y(self, field, IdxX=None):
    # in case field is the name of an existing class object attribute
    if(isinstance(field, str)):
      field = getattr(self, field)
    # in case field is not numpy array
    field = np.array(field)
    # in case field is already one dimensional
    if field.ndim == 1:
      return field
    # 2D case
    else:
      # if no crosssection index is provided
      if(IdxX is None):
        # take index in the center of the first axis
        IdxX = np.shape(field)[0]//2
      return field[IdxX]
  # wrapper for numpy argsort
  def ArgSort1DField(self, field, IdxX=None):
    self.Sorted1DIndices = np.argsort(self.get_crosssection_y(field, IdxX))
  # check if the provided field has negative values
  # def CheckIfNegative(self, field):
    # return np.any(np.array(field)<0)
  def CheckIfBelowThreshold(self, field, Threshold):
    # requires field to be "rectangular" shape, otherwise np.array does not work
    # and therefore comparison operation wouldnt work either
    return np.any(np.array(field)<Threshold)

  # get indices where >y
  def DomainRightPartIndices1D(self, y):
    return self.y>y
  # get values where >y
  def DomainRightPart1D(self, data, y):
    MaskedIndices = self.DomainRightPartIndices1D(y)
    Maskedy = self.y[MaskedIndices]
    return data[MaskedIndices], Maskedy, MaskedIndices

  # finds peaks with similar height to the highest peak for y>yMeasure (default: half of domain)
  def FindHighestPeaksMasked1D(self, Field, FractionOfMaximumProminence = None, yMeasure = None):
    # does nothing if Field is already 1D
    data = self.get_crosssection_y(Field)
    if(yMeasure == None):
      yMeasure = self.Ly/2
    # mask domain according to ymeasure
    MaskedData, Maskedy, MaskedIndices = self.DomainRightPart1D(data, yMeasure)
    # find highest peaks
    PeakIndices = self.FindHighestPeaks1D(MaskedData, FractionOfMaximumProminence)
    return PeakIndices, MaskedData, Maskedy, MaskedIndices
  # finds peaks with similar height to the highest peak for y>yMeasure (default: half of domain)
  def FindHighestPeaks1D(self, Field, FractionOfMaximumProminence = None):
    if(FractionOfMaximumProminence == None):
      FractionOfMaximumProminence = 0.9
    # does nothing if Field is already 1D
    data = self.get_crosssection_y(Field)
    PeakIndices, properties = find_peaks(data, prominence = 0)
    # prominence may not be intuitive for people who think of peaks as superpositions of
    # peaks. For a found peak: the nearest minima on the left and right are the bases. 
    # however for the calculation of the prominence the higher one of the two is chosen. 
    Prominences = properties['prominences']
    # get MaximumProminence, corresponding to the most significant peak
    # using FractionOfMaximum... makes most sense with prominence rather than keyword height
    # since one period may contain two peaks of very similar height. 
    # since the smaller peak will have a much smaller prominence due to it neighboring the other
    # peak, the smaller peak can be excluded by filtering out small prominences
    MaximumProminence = np.max(Prominences)
    PeakIndices = PeakIndices[Prominences>MaximumProminence*FractionOfMaximumProminence]
    return PeakIndices


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
  # dyy fdm 4th order, center
  def dyy4_m22(self, FieldsOutsideOfInstance=None, dx2=None):
    f = self.f
    if(FieldsOutsideOfInstance is not None):
      self.DummyFields = FieldsOutsideOfInstance
    else:
      self.DummyFields = self.fields
    if(dx2 is None):
      dx2 = self.dx2()
    return ( -f(0,-2) + 16.0*f(0,-1) - 30.0*f(0,0) + 16.0*f(0,1) -f(0,2) )/12.0/dx2
  # dyyy fdm 4th order, center, copied from cuda code
  def dyyy4_m33(self, FieldsOutsideOfInstance=None, dx3=None):
    f = self.f
    if(FieldsOutsideOfInstance is not None):
      self.DummyFields = FieldsOutsideOfInstance
    else:
      self.DummyFields = self.fields
    if(dx3 is None):
      dx3 = self.dx3()
    return (-f(0,3) + 8.0*f(0,2) - 13.0*f(0,1) + 13.0*f(0,-1) - 8.0*f(0,-2) + f(0,-3))/8.0/dx3
















