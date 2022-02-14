from solution import solution
import cuda
from pathlib import Path, PurePath

class LB(solution):
  def __init__(self, path):
    super().__init__()
    self.dtype = 'float'
    if path is not None:
      self.path = Path(path)
      self.readparams(self.path)
      self.set_SpatialDimensions()
      self.set_coordinates()
      self.set_SpatialGrid1Dy()
      self.fields = cuda.readbin(self)
      self.nof = len(self.fields)