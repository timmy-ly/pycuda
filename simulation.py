import numpy as np
from solution import solution
from pathlib import Path,PurePath
import cuda
import time
from memory_profiler import profile

# default values
attribute = 'imagenumber'
def convert(val):
  constructors = [int, float, str]
  for c in constructors:
    try:
      return c(val)
    except ValueError:
      pass

class SimulMeasures:
  def __init__(self):
    self.t = None

# class for calculated field of a simulation, usually contains meta data and norms of the field values over a simulation
class FieldsMetaAndNorm:
  def __init__(self, AttributeName, OuterPointsToDropPerSide = 0):
    self.array = None
    self.max = None
    self.min = None
    self.name = AttributeName
    # drop points if you calculate spatial derivatives since they are
    # caluculated by np.roll and we usually do not consider boundary conditions
    self.OuterPointsToDropPerSide = OuterPointsToDropPerSide
  def set_max(self, Iterable):
    istart, iend = 0 + self.OuterPointsToDropPerSide, -1 - self.OuterPointsToDropPerSide
    self.max = np.nanmax(np.array(Iterable)[:,:,istart:iend])
  def set_min(self, Iterable):
    istart, iend = 0 + self.OuterPointsToDropPerSide, -1 - self.OuterPointsToDropPerSide
    self.min = np.nanmin(np.array(Iterable)[:,:,istart:iend])
    
# simulation class
class Simulation:
  # constructor
  def __init__(self, path = None, start = None, end = None, file = 'frame_0000.dat', 
               objectclass = solution, attribute = attribute):
    self.pattern = 'frame_[0-9]*bin'
    self.params = {}
    self.filepaths = None
    self.file = file
    self.attribute = attribute
    self.NumberOfFilepaths = None
    self.sols = None
    self.nof = None
    self.objectclass = objectclass
    self.start = start
    self.end = end
    self.Stationary = None
    # read simulation parameters if path is present
    if path is not None:
      self.path = Path(PurePath(path))
      self.readparams(filepath = self.path)
      self.set_solutions(pattern = self.pattern)
      self.sort_solutions(attribute = self.attribute)
      self.set_time()
    else:
      self.path = None
  # read simulation parameters as attribute of type dict, default file is 0th frame
  def readparams(self, filepath=None, file = None):
    if filepath is None:
      filepath = self.path
    if file is None:
      file = self.file
    filepath = self.path / file
    try:
      with open(str(filepath),'r') as f:
        for line in f.readlines():
          # handle IndexError that occurs when a line only has one column
          try:
            self.params[line.split()[0]] = convert(line.split()[1])
          except IndexError:
            continue
    except FileNotFoundError:
      print(self.path, filepath)
      raise FileNotFoundError
  # set paths of all simulation frames
  def set_filepaths(self, pattern = None):
    if(pattern is None):
      pattern = self.pattern
    self.Filepaths = list(self.path.glob(pattern))
    self.NumberOfFilepaths = len(self.Filepaths)
  # get solution objects of all frames
  # @profile()
  def set_solutions(self, pattern = None):
    if(pattern is None):
      pattern = self.pattern
    self.set_filepaths(pattern = pattern)
    Filepaths = self.Filepaths
    objectclass = self.objectclass
    self.sols = [objectclass(Filepath) for Filepath in Filepaths]
    # self.sols = map(objectclass, Filepaths)
  # sort solution objects and crop if start/end are provided, default attribute is time
  def sort_solutions(self, attribute = attribute):
    ObjectClass = self.objectclass
    self.sols = sorted(self.sols, key = lambda ObjectClass:getattr(ObjectClass,attribute))[self.start:self.end]
  def ExtractSolsSubset(self, ListOfIndices):
    SolsSubset = []
    for i in ListOfIndices:
      if(i<len(self.sols)):
        SolsSubset.append(self.sols[i])
      else:
        break
    return SolsSubset
  # ExtractSolsSubset and adjust Filepaths, numberoffilepaths attributes
  def FilterSolsFromSubset(self, ListOfIndices):
    self.sols = self.ExtractSolsSubset(ListOfIndices)
    self.Filepaths = [sol.path for sol in self.sols]
    self.NumberOfFilepaths = len(self.Filepaths)
  def set_BCType(self, BCType):
    for sol in self.sols:
      sol.BC = BCType
  # wrapper to get sort-indices of field with name attribute for each sol object
  def ArgSort1DField(self, attribute='h'):
    for i in np.arange(self.NumberOfFilepaths)[self.start:self.end]:
      self.sols[i].ArgSort1DField(getattr(self.sols[i], attribute))
  # set number of fields
  def set_nof(self):
    if(self.sols is None):
      print("sols is None, call set_solutions first or check self.path / self.Filepaths")
    self.nof = self.sols[0].nof
  # set spatial grid in 1D in y direction
  def set_SpatialGrid1Dy(self):
    Ny = self.params['Ny']
    Ly = self.params['Ly']
    self.y = np.arange(Ny)*Ly/Ny
  def set_SpatialGrid2D(self):
    Nx = self.params['Nx']
    Lx = self.params['Lx']
    Ny = self.params['Ny']
    Ly = self.params['Ly']
    self.y2D, self.x2D = np.meshgrid(np.arange(Ny)*Ly/Ny, np.arange(Nx)*Lx/Nx)
  def set_time(self):
    self.t = np.array([sol.t for sol in self.sols])
  # set fields you calculate from the original fields according to method of sol 
  # you can use getattr for methods too!
  def set_CalculatedFields(self, MethodNames):
    if self.sols is not None:
      # can pass multiple methods
      if(isinstance(MethodNames, list)):
        for MethodName in MethodNames:
          for sol in self.sols:
            getattr(sol, MethodName)()
      # for single method
      else:
        for sol in self.sols:
          getattr(sol, MethodNames)()
    else:
      print("solution objects have not been set yet. ")
  # create multiple FieldsMetaAndNorm objects according to ListOfAttributeNames
  # then determine the maximum and minimum values of the fields
  # technically loops through whole simulation for each AttributeName
  # naively this should be fastest possible since looping once through simulation but looping through ListOfAttributeNames should be equal
  def set_FieldsMetaAndNorm(self, ListOfAttributeNames, OuterPointsToDropPerSide=0):
    self.FieldsMetaAndNorm = [FieldsMetaAndNorm(AttributeName, OuterPointsToDropPerSide) for AttributeName in ListOfAttributeNames]
    for fieldobj in self.FieldsMetaAndNorm:
      fieldobj.set_max([getattr(sol, fieldobj.name) for sol in self.sols])
      fieldobj.set_min([getattr(sol, fieldobj.name) for sol in self.sols])

  def get_field(self,fieldname):
    return np.array([getattr(sol, fieldname) for sol in self.sols])

  # wrapper to apply method to all solutions, functionhandle has to be string
  def ApplyToAll(self,FunctionHandle,*args):
    for sol in self.sols:
      getattr(sol, FunctionHandle)(*args)
  def GetMax(self, Attribute):
    return np.nanmax([getattr(sol, Attribute) for sol in self.sols])
  # check if the solution is stationary
  # eps: threshold / maximum error
  # n: last n frames to use
  def set_Stationary(self, n = 200, eps = 1e-10):
    # initialize maximumerror
    self.MaximumError = 0
    self.Stationary = True
    # loop through last n solutions
    for i in np.arange(n):
      # map i such that indices go from small to big
      index = i - n
      # take previous solution as reference
      sol1 = self.sols[index - 1]
      sol2 = self.sols[index]
      # maximum of norm is enough
      Norm = np.max(cuda.DifferenceNorm(sol1, sol2))
      # save maximumerror
      if(Norm > self.MaximumError):
        self.MaximumError = Norm
      # check if any error surpasses threshold
      if(Norm > eps):
        self.Stationary = False
        break

    

  
    




  

    

    















