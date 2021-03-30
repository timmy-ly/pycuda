import numpy as np
from precipitation import precipiti
from pathlib import Path,PurePath

# default values
pattern = 'frame_[0-9]*bin'
objectclass = precipiti
attribute = 't'
def convert(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass

class Simulation:
  # constructor
  def __init__(self, path = None):
    self.params = {}
    self.filepaths = None
    self.NumberOfFilepaths = None
    self.sols = None
    self.nof = None
    # read simulation parameters if path is present
    if path is not None:
      self.path = Path(PurePath(path))
      self.readparams(filepath = self.path)
    else:
      self.path = None
  # read simulation parameters as attribute of type dict, default file is 0th frame
  def readparams(self, filepath=None, file = 'frame_0000.dat'):
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
  def set_solutions(self, objectclass = objectclass, pattern = pattern):
    self.set_filepaths(pattern = pattern)
    self.sols = [objectclass(self.Filepaths[i]) for i in range(self.NumberOfFilepaths)]
  # sort solution objects and crop if start/end are provided, default attribute is time
  def sort_solutions(self, objectclass = objectclass, attribute = attribute, start=None, end=None):
    self.sols = sorted(self.sols, key = lambda objectclass:getattr(objectclass,attribute))[start:end]
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



  
    




  

    

    















