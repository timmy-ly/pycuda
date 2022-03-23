import numpy as np
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
  def add_dirichlet(self):
    self.fields = np.insert(self.fields, 0, self.c0, axis = -1)
  def interp_1D(self, array, Nx, Lx, newNx):
    from scipy.interpolate import interp1d
    dx = Lx/Nx
    n = len(array)
    x = dx*np.arange(n)
    newdx = Lx/newNx
    if(n == Nx):
      newx = newdx*np.arange(newNx)
    elif (n == Nx + 1):
      newx = newdx*np.arange(newNx + 1)
    interpolatedobject = interp1d(x,array)
    # evaluate interpolatedobject at newx, newy points
    return interpolatedobject.__call__(newx)

class LBBranch:
  # def __init__(self, ControlParam, Top, SimulPattern, SolPattern = 'frame.bin'):
  def __init__(self, ControlParam, Top, SimulPattern, SolPattern = 'frame.dat'):
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
      if not FilePaths:
        print("no finish frame, using SolPattern frame_[0-9]*.dat")
        FilePaths = list(SimulPath.glob("frame_[0-9]*.dat"))
      # sort by time
      FilePath = sorted(FilePaths, key = lambda filepath:cuda.GetOnePropertyFromParamFile(cuda.dat(filepath), "t"))[-1]
      self.sols.append(LB(FilePath))
  def sort_sols(self):
    self.sols = sorted(self.sols, key = lambda obj:getattr(obj,self.ControlParam))
  def set_l2(self, newNx = None, newNy = None, dim = 1):
    # adding the dirichlet BC matches the nodes a lot better
    self.add_dirichlet()
    if(dim == 1):
      interpolated = False
      if newNy is not None:
        self.interp_1D(newNy)
        interpolated = True
      elif newNx is not None:
        self.interp_1D(newNx)
        interpolated = True
      if(interpolated):
        self.l2 = [cuda.l2norm(sol.InterpolatedFields) for sol in self.sols]
      else:
        self.l2 = [cuda.l2norm(sol.fields[0][0]) for sol in self.sols]
    else:
      self.l2 = [cuda.l2norm(sol.fields[0]) for sol in self.sols]
    self.v = [sol.v for sol in self.sols]
  def add_dirichlet(self):
    for sol in self.sols:
      sol.add_dirichlet()    
  def interp_1D(self, newNy):
    for sol in self.sols:
      data = sol.fields[0][0]
      newdata = sol.interp_1D(data, sol.Ny, sol.Ly, newNy)
      sol.InterpolatedFields = newdata


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