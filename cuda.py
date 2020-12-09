import numpy as np
import warnings
from scipy.interpolate import RectBivariateSpline as rbs

# methods that apply to all cuda problems
# add .dat to filepath
def dat(filepath):
  return filepath + '.dat'

# add .bin to filepath
def bin(filepath):
  return filepath + '.bin'

#Error of numerical dissipation
# @staticmethod
def E_diss(array_true, array_fd):
  std_T = np.std(array_true)
  std_D = np.std(array_fd)
  mean_T = np.mean(array_true)
  mean_D = np.mean(array_fd)
  return (std_T - std_D)**2 + (mean_T - mean_D)**2
  
#read cuda bin file
def readbin(solobj=None, filepath=None, dtype=None, nof=None, Nx=None, Ny=None):
  if (solobj is not None):
    filepath, dtype, nof, Nx, Ny = solobj.path, solobj.dtype, solobj.nof, solobj.Nx, solobj.Ny
  filepath = bin(filepath)
  if dtype == 'float':
    array = np.fromfile(filepath,np.dtype('float32'))
    array = array.astype('float')
  else:
    array = np.fromfile(filepath,np.dtype('float64'))
    array = array.astype('double')
  array = array.reshape((nof*Nx,Ny))
  return array

# read dat file and return value of param as a string
def readparam(param, solobj=None, filepath=None):
  if solobj is not None:
    filepath = solobj.path
  filepath = dat(filepath)
  with open(filepath,'r') as f:
    lines = f.readlines()
  for i in np.arange(len(lines)):
    if lines[i].split()[0] == param:
      param_value_str = lines[i].split()[1]
      break
  return param_value_str

# read balancedata
def readbalance(filepath, n=0):
  with open(filepath,'r') as f:
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      data = np.loadtxt(f, skiprows=1, ndmin=2)
  # create copy of data that keeps column 0 and removes columns
  # 1 to n, np.s_ is a slice, last argument is axis
  return np.delete(data,np.s_[1:-n],1)

# interpolate array with Nx, Ny, Lx, Ly to newNx, Newy
def interpolate(array, Nx, Ny, Lx, Ly, newNx, newNy):
  x = np.linspace(0,Lx,Nx)
  y = np.linspace(0,Ly,Ny)
  newx = np.linspace(0,Lx,newNx)
  newy = np.linspace(0,Ly,newNy)
  interpolatedobject = rbs(x,y,array,bbox=[0,Lx,0,Ly])
  # evaluate interpolatedobject at newx, newy points
  return interpolatedobject.__call__(newx,newy)

# finite difference stencils
def dy4_04(field, dy):
  # forward difference from node 0 to node 4
  return (-25*field + 4*12*np.roll(field,-1,1) - 3*12*np.roll(field,-2,1) + 4/3*12*np.roll(field,-3,1) - 12/4*np.roll(field,-4,1))/(12*dy)
def dy4_m13(solobj = None, field = None, dy = None):
  # node -1 to node 3
  if (field is None) or (dy is None):
    field = solobj.field, dy=solobj.dy
  return (-3*np.roll(field,1,1) - 10*field + 18*np.roll(field,-1,1) - 6*np.roll(field,-2,1) + np.roll(field,-3,1) )/(12*dy)
def dy4_m22(field, dy):
  # node -2 to node 2
  return (np.roll(field,2,1) - 8*np.roll(field,1,1) + 0 + 8*np.roll(field,-1,1) - np.roll(field,-2,1) )/(12*dy)
def dyyy4_06(field, dy3):
  # node 0 to 6
  return (-49*field + 29*8*np.roll(field,-1,1) - 461*np.roll(field,-2,1) + 62*8*np.roll(field,-3,1) - 307*np.roll(field,-4,1) + 13*8*np.roll(field,-5,1) - 15*np.roll(field,-6,1))/(8*dy3)
def dyyy4_m15(field, dy3):
  # node -1 to node 5
  return (-15*np.roll(field,1,1) + 56*field - 83*np.roll(field,-1,1) + 64*np.roll(field,-2,1) -29*np.roll(field,-3,1) + 8*np.roll(field,-4,1) - np.roll(field,-5,1) )/(8*dy3)
def dyyy4_m24(field, dy3):
  # node -1 to node 5
  return (-np.roll(field,2,1) - 8*np.roll(field,1,1) + 35*field - 48*np.roll(field,-1,1) + 29*np.roll(field,-2,1) -8*np.roll(field,-3,1) + np.roll(field,-4,1) )/(8*dy3)