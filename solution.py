import numpy as np
from scipy.signal import find_peaks
import cuda
from cuda import convert
class NoExtremaError(Exception):
  pass

class SolutionMeasures:
  def __init__(self):
    self.PeakIndex = None
    self.Height = None
    self.Prominence = None
    self.Base = None
class FieldProps:
  def __init__(self, data = None, FieldName = None):
    self.FieldName = FieldName
    self.data = data
    self.dim = len(np.shape(data))
    self.Peak = None
  def GetMaxIdx(self):
    return np.argmax(self.Peak)
  def GetMax(self):
    return self.Peak[self.GetMaxIdx()]
  def GetMaxPos(self):
    return self.yPeak[self.GetMaxIdx()]
  # minimum should be the left most position of the peak
  def GetMin(self):
    return self.Peak[0]
  def GetMinPos(self):
    return self.yPeak[0]



# class for a solution (for some time t) from cuda data
class solution:
  def __init__(self):
    #
    self.OneFName = None
    # numerical
    self.dtype = 'double'
    self.Nx = None
    self.Ny = None
    self.dim = None
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

  def readparams(self, filepath=None):
    if filepath is None:
      filepath = self.path
    filepath = str(cuda.dat(filepath))
    with open(filepath,'r') as f:
      for line in f:
        entries = line.split()
        # pass
        if len(entries)>1:
          setattr(self, entries[0], convert(entries[1]))
      
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
  
  def set_SpatialDimensions(self):
    if(self.Nx == 1):
      self.dim = 1
    else:
      self.dim = 2

  # 2d arrays of y and x
  def set_coordinates(self):
    Nx = self.Nx
    Ny = self.Ny
    Lx = self.Lx
    Ly = self.Ly
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

  # finds peaks with similar height to the highest peak for y>ymin (default: half of domain)
  # Arrays are masked once to cut down the domain
  # then indices/peaks that do not correspond to high enough prominence are filtered out 
  # Note that index values do not change only array sizes change
  def FindHighestPeaksRight1D(self, data, FractionOfMaximumProminence = None, RelativeMeasurePt = 0.9, RelativeYMin = 0.3, **kwargs):
    # does nothing if Field is already 1D
    data1D = self.get_crosssection_y(data)
    # Take Reference Prominence from between the left boundary and the measurepoint
    # set the required indices for these boundaries
    self.set_YMinIndex(RelativeYMin)
    self.set_MPIndex(RelativeMeasurePt)
    # find highest peaks in full domain
    PeakIndices, properties = self.FindPeaks1D(data1D, **kwargs)
    if(len(PeakIndices)==0):
      raise NoExtremaError('Could not find extrema or missing left boundary of the peak')
    # get the reference prominence as stated above
    self.GetMaximumProminenceLeft(PeakIndices, properties)
    # exclude the peaks that have too small prominence
    ProminenceMask = cuda.MaskFromValue(properties, 'prominences', self.MaxProminenceLeft, FractionOfMaximumProminence)
    cuda.ExcludePeaksFromProperties(properties, PeakIndices, ProminenceMask)
    # Update PeakIndices
    PeakIndices = PeakIndices[ProminenceMask]
    # add peak positions
    properties["positions"] = self.y[PeakIndices]
    return PeakIndices, properties

  # wrapper of FindHighestPeaksRight1D but reversed for minima
  def FindSmallestMinimaRight1D(self, data, FractionOfMaximumProminence = None, RelativeMeasurePt = 0.9, RelativeYMin = 0.3, height = (None, 0)):
    MinimaIndices, properties = self.FindHighestPeaksRight1D(-data, FractionOfMaximumProminence, RelativeMeasurePt, RelativeYMin, height = height)
    return MinimaIndices, properties

  # wrapper for scipy's find_peaks
  def FindPeaks1D(self, data, height = 0, prominence = 0):
    # does nothing if data is already 1D
    data1D = self.get_crosssection_y(data)
    # provide prominence and height keywords in order to save their values in properties
    # left/right_bases seem to be useless since they are only relevant for prominence
    # in general they correspond to non-local minima except for when wlen is specified well
    # PeakIndices are relative to the shape of data1D
    # see FindMinima1D for minima algorithm
    PeakIndices, properties = find_peaks(data1D, height = height, prominence = prominence)
    # if(len(PeakIndices)==0):
      # raise NoExtremaError('Could not find extrema or missing left boundary of the peak')
    return PeakIndices, properties
  # # wrapper for scipy's find_peaks
  # def FindMinima1D(self, data):
  #   # does nothing if data is already 1D
  #   data1D = self.get_crosssection_y(data)
  #   # negate data, find peaks-> find minima
  #   # only consider heights: (-infty, 0). height=0 would filter out all "heights" below 0
  #   # height is simply the value of -data1D
  #   MinimaIndices, properties = find_peaks(-data1D, height = (None, 0), prominence = 0)
  #   return MinimaIndices, properties


  # 
  def GetMaximumProminenceLeft(self, PeakIndices, properties, CutLeft = True):
    mask = (PeakIndices<=self.MPIndex)
    if(CutLeft):
      mask = mask & (PeakIndices>self.YMinIndex)
    self.MaxProminenceLeft = np.max(properties['prominences'][mask])

  def set_YMinIndex(self, RelativeYMin):
    self.YMinIndex = int(self.Ny*RelativeYMin)
  # set index of measurepoint (round down if float)
  def set_MPIndex(self, RelativeMeasurePt = 0.9):
    # domain size Ly maps exactly to Ny. Grid points only go to index Ny-1 and thus domain
    # Ly-dx but that doesnt change the mapping
    self.MPIndex = int(self.Ny*RelativeMeasurePt)
  # get indices where >ymin
  def SplitDomainRightIndices1D(self, ymin= None):
    if(ymin == None):
      ymin = self.Ly/2
    # relevant indices after splitting the domain
    self.SplitDomainMask = self.y>ymin
    self.Maskedy = self.y[self.SplitDomainMask]
  # get values where >ymin
  def SplitDataRight1D(self, data, ymin=None):
    if(ymin == None):
      ymin = self.Ly/2
    self.SplitDomainRightIndices1D(ymin)
    return data[self.SplitDomainIndices]

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
















