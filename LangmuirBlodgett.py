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

class LBBranch:
  # def __init__(self, ControlParam, Top, SimulPattern, SolPattern = 'frame.bin'):
  def __init__(self, ControlParam, Top, SimulPattern, SolPattern = 'frame_[0-9]*.dat'):
    self.ControlParam = ControlParam
    self.Top = Path(PurePath(Top))
    self.SimulPattern = SimulPattern
    self.SolPattern = SolPattern
    self.sols = []
    self.set_sols()
    self.sort_sols()

  def set_sols(self):
    self.set_SimulPaths()
    self.FilePathsToSols()

  # set paths of simulation directories
  def set_SimulPaths(self):
    from PathManipulation import GetSubdirs
    self.SimulPaths = GetSubdirs(self.SimulPattern, self.Top)
    self.NumberOfFilepaths = len(self.SimulPaths)
  def FilePathsToSols(self):
    for SimulPath in self.SimulPaths:
      FilePaths = list(SimulPath.glob(self.SolPattern))
      # sort by time
      FilePath = sorted(FilePaths, key = lambda filepath:cuda.GetOnePropertyFromParamFile(cuda.dat(filepath), "t"))[-1]
      self.sols.append(LB(FilePath))
  def sort_sols(self):
    self.sols = sorted(self.sols, key = lambda obj:getattr(obj,self.ControlParam))

  # def sort_solutions(self, attribute = attribute):
  #   ObjectClass = self.objectclass
  #   self.sols = sorted(self.sols, key = lambda ObjectClass:getattr(ObjectClass,attribute))[self.start:self.end]
class LBCont(Continuation):
  def __init__(self, ParamsCMDArgs, ParamsOther):
    super().__init__(ParamsCMDArgs, ParamsOther)
  def get_Subdir(self, args):
    SubdirSuffix = ("Lx{Lx:g}_Ly{Ly:g}_Nx{Nx:d}_Ny{Ny:d}_eps{epsilontol:g}_noise{noise:g}_ys{ys:g}_ls{ls:g}_c0{c0:g}_v{v:g}"
    .format(**args))
    return self.SubdirPrefix + "_" + SubdirSuffix