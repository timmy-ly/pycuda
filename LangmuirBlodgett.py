from continuation import Continuation
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

class LBCont(Continuation):
  def __init__(self, ParamsCMDArgs, ParamsOther):
    super().__init__(ParamsCMDArgs, ParamsOther)
  def get_Subdir(self, args):
    SubdirSuffix = ("Lx{Lx:g}_Ly{Ly:g}_Nx{Nx:d}_Ny{Ny:d}_eps{epsilontol:g}_noise{noise:g}_ys{ys:g}_ls{ls:g}_c0{c0:g}_v{v:g}"
    .format(**args))
    return self.SubdirPrefix + "_" + SubdirSuffix