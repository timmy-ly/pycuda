import numpy as np
from solution import solution
from pathlib import Path,PurePath

# default values
pattern = 'frame_[0-9]*bin'
attribute = 't'
def convert(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass

# class for calculated field of a simulation, usually contains meta data and norms of the field values over a simulation
class FieldsMetaAndNorm:
  def __init__(self, AttributeName):
    self.array = None
    self.max = None
    self.min = None
    self.name = AttributeName
  def set_max(self, Iterable):
    self.max = np.nanmax(Iterable)
  def set_min(self, Iterable):
    self.min = np.nanmin(Iterable)
# simulation class
class Simulation:
  # constructor
  def __init__(self, path = None, start = None, end = None):
    self.params = {}
    self.filepaths = None
    self.NumberOfFilepaths = None
    self.sols = None
    self.nof = None
    self.objectclass = solution
    self.start = start
    self.end = end
    # read simulation parameters if path is present
    if path is not None:
      self.path = Path(PurePath(path))
      self.readparams(filepath = self.path)
    else:
      self.path = None
  # read simulation parameters as attribute of type dict, default file is 0th frame
  def readparams(self, filepath=None, file = 'frame_0001.dat'):
    if filepath is None:
      filepath = self.path
    filepath = self.path / file
    with open(str(filepath),'r') as f:
      for line in f.readlines():
        # handle IndexError that occurs when a line only has one column
        try:
          self.params[line.split()[0]] = convert(line.split()[1])
        except IndexError:
          continue
  # set paths of all simulation frames
  def set_filepaths(self, pattern = pattern):
    self.Filepaths = list(self.path.glob(pattern))
    self.NumberOfFilepaths = len(self.Filepaths)
  # get solution objects of all frames
  def set_solutions(self, pattern = pattern):
    self.set_filepaths(pattern = pattern)
    self.sols = [self.objectclass(self.Filepaths[i]) for i in range(self.NumberOfFilepaths)[self.start:self.end]]
  # sort solution objects and crop if start/end are provided, default attribute is time
  def sort_solutions(self, attribute = attribute):
    ObjectClass = self.objectclass
    self.sols = sorted(self.sols, key = lambda ObjectClass:getattr(ObjectClass,attribute))[self.start:self.end]
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
  def set_time(self):
    self.t = [sol.t for sol in self.sols]
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
  def set_FieldsMetaAndNorm(self, ListOfAttributeNames):
    self.FieldsMetaAndNorm = [FieldsMetaAndNorm(AttributeName) for AttributeName in ListOfAttributeNames]
    for fieldobj in self.FieldsMetaAndNorm:
      fieldobj.set_max([getattr(sol, fieldobj.name) for sol in self.sols])
      fieldobj.set_min([getattr(sol, fieldobj.name) for sol in self.sols])
  

  
    



  
    




  

    

    















