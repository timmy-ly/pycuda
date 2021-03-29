import numpy as np
from pathlib import Path,PurePath

def convert(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass

class Simulation:
  def __init__(self, path = None):
    self.params = {}
    if path is not None:
      self.path = PurePath(path)
      self.readparams(filepath = self.path)
    else:
      self.Path = None

  def readparams(self, filepath=None, file = 'frame_0000.dat'):
    if filepath is None:
      filepath = self.path
    filepath = self.path / file
    with open(str(filepath),'r') as f:
      for line in f.readlines():
        try:
          self.params[line.split()[0]] = convert(line.split()[1])
        except IndexError:
          continue
    

    















