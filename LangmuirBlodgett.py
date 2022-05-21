import numpy as np
from continuation import Continuation
from simulation import Simulation
from solution import solution
import cuda
from pathlib import Path, PurePath

SortAttribute = 'imagenumber'


class LB(solution):
  def __init__(self, path= None):
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
      self.set_c()
      self.Ny_up = self.Ny
      self.Ny_down = self.Ny
      self.cut = 0
      self.cCut = self.c #c after cutoff
      self.cCutShift = self.cCut #c after shifting by its mean
  #FFT: source of 0-mode: non-zero mean of the pattern (k=0 is a constant fourier factor--> constant added to the pattern --> peak added in FT)
  def set_cCutShift(self):
    self.cCut = cuda.cut_field_y(self.c)
    self.cCutShift = cuda.shift_field(self.cCut)
  def add_dirichlet(self):
    self.fields = np.insert(self.fields, 0, self.c0, axis = -1)
  def set_c(self):
    self.c = self.fields[0]

  def get_k(self, field):
    dx = self.dx()
    dy = dx
    kx, ky = cuda.set_k(field, dx, dy)
    return kx, ky
  def get_fft(self, field):
    field_fft = cuda.fft_field(field)
    return cuda.normalize_fft(field_fft)



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
  def l2norm(self, field):
    l2path = self.path.parent / "frame_l2.dat"
    if(l2path.exists()):
      data = cuda.read_l2(l2path)
      l2 = cuda.mean_l2(data[0], data[1])
      return l2
    else:
      return cuda.l2norm(field)
class LBSimu(Simulation):
  def __init__(self, path, start = None, end = None, file = 'frame_0001.dat', 
              objectclass = LB, SortAttribute = SortAttribute):
    super().__init__(path, start = start, end = end, file = file, 
                    objectclass = objectclass, SortAttribute = SortAttribute)
    self.set_SpatialGrid1Dy()
  def set_l2(self):
    # adding the dirichlet BC matches the nodes a lot better
    self.add_dirichlet()
    self.l2path = list(self.path.glob("*l2.dat"))[0]
    # print(self.l2path.exists())
    if(self.l2path.exists()):
      _, self.l2 = cuda.read_l2(self.l2path)
    else:
      self.l2 = [sol.l2norm(sol.fields[0]) for sol in self.sols]
  def add_dirichlet(self):
    for sol in self.sols:
      sol.add_dirichlet()   

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
        print("no finish frame, using SolPattern frame_[0-9]*.dat: ", self.Top)
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
        self.l2 = [sol.l2norm(sol.InterpolatedFields) for sol in self.sols]
      else:
        self.l2 = [sol.l2norm(sol.fields[0][0]) for sol in self.sols]
    else:
      self.l2 = [sol.l2norm(sol.fields[0]) for sol in self.sols]
    self.l2 = np.array(self.l2)
    self.set_v()
  def set_dcdx_max(self):
    self.set_v()
    dx2_m11 = cuda.dx2_m11
    self.dcdx_max = [np.max(np.abs(dx2_m11(sol.fields[0], sol.dx()))) for sol in self.sols]
    self.dcdx_max = np.array(self.dcdx_max)
  def set_v(self):
    self.v = np.array([sol.v for sol in self.sols])
  def add_dirichlet(self):
    for sol in self.sols:
      sol.add_dirichlet()    
  def interp_1D(self, newNy):
    for sol in self.sols:
      data = sol.fields[0][0]
      newdata = sol.interp_1D(data, sol.Ny, sol.Ly, newNy)
      sol.InterpolatedFields = newdata

class LBBranchLegacy(LBBranch):
  # used for primitively continued DNS branches with older directory structure, aka no saved solution frames, only final frames + solution measures in the top folder
  def __init__(self, ControlParam, Top, SolPattern = 'Lx*.bin'):
    super().__init__(ControlParam, Top, "dummy", SolPattern = SolPattern)
  def set_sols(self):
    # print("overwritten set_sols")
    self.FilePathsToSols()
  def FilePathsToSols(self):
    # also, sorted by controlparam already, dont need sort_solutions method
    FilePaths = list(self.Top.glob(self.SolPattern))
    if not FilePaths:
      print("FilePaths empty, check SolPattern and Tops ", self.Top)
    # sort by ControlParam
    FilePaths = sorted(FilePaths, key = lambda filepath:cuda.GetOnePropertyFromParamFile(cuda.dat(filepath), self.ControlParam))
    self.sols = [LB(FilePath) for FilePath in FilePaths]
    # print(len(self.sols))

class LBCont(Continuation):
  def __init__(self, ParamsCMDArgs, ParamsOther):
    super().__init__(ParamsCMDArgs, ParamsOther)
  def get_Subdir(self, args):
    if(args["vamplitude"]==0):
      SubdirSuffix = ("Lx{Lx:g}_Ly{Ly:g}_Nx{Nx:d}_Ny{Ny:d}_eps{epsilontol:g}_noise{noise:g}_ys{ys:g}_ls{ls:g}_c0{c0:g}_v{v:g}".format(**args))
    else:
      SubdirSuffix = ("Lx{Lx:g}_Ly{Ly:g}_Nx{Nx:d}_Ny{Ny:d}_eps{epsilontol:g}_noise{noise:g}_ys{ys:g}_ls{ls:g}_c0{c0:g}_v{v:g}_A{vamplitude:g}_w{vperiod:g}".format(**args))
    return self.SubdirPrefix + "_" + SubdirSuffix