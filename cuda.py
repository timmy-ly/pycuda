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
  print(filepath)
  if dtype == 'float':
    array = np.fromfile(filepath,np.dtype('float32'))
    array = array.astype('float')
  else:
    array = np.fromfile(filepath,np.dtype('float64'))
    array = array.astype('double')
  array = array.reshape((nof,Nx,Ny))
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
# forward difference from node 0 to node 4
def dy4_04(solobj = None, field = None, dy = None):
  if solobj is not None:
    field, dy = solobj.field, solobj.dy()
  return (-25*field + 4*12*np.roll(field,-1,1) - 3*12*np.roll(field,-2,1) + 4/3*12*np.roll(field,-3,1) - 12/4*np.roll(field,-4,1))/(12*dy)

# node -1 to node 3
def dy4_m13(solobj = None, field = None, dy = None):
  if solobj is not None:
    field, dy = solobj.field, solobj.dy()
  return (-3*np.roll(field,1,1) - 10*field + 18*np.roll(field,-1,1) - 6*np.roll(field,-2,1) + np.roll(field,-3,1) )/(12*dy)

# node -2 to node 2
def dy4_m22(solobj = None, field = None, dy = None):
  if solobj is not None:
    field, dy = solobj.field, solobj.dy()
  return (np.roll(field,2,1) - 8*np.roll(field,1,1) + 0 + 8*np.roll(field,-1,1) - np.roll(field,-2,1) )/(12*dy)

# node 0 to 6
def dyyy4_06(solobj=None, field=None, dy3=None):
  if solobj is not None:
    field, dy3 = solobj.field, solobj.dy()**3
  return (-49*field + 29*8*np.roll(field,-1,1) - 461*np.roll(field,-2,1) + 62*8*np.roll(field,-3,1) - 307*np.roll(field,-4,1) + 13*8*np.roll(field,-5,1) - 15*np.roll(field,-6,1))/(8*dy3)

# node -1 to node 5
def dyyy4_m15(solobj=None, field=None, dy3=None):
  if solobj is not None:
    field, dy3 = solobj.field, solobj.dy()**3
  return (-15*np.roll(field,1,1) + 56*field - 83*np.roll(field,-1,1) + 64*np.roll(field,-2,1) -29*np.roll(field,-3,1) + 8*np.roll(field,-4,1) - np.roll(field,-5,1) )/(8*dy3)

# node -2 to node 4  
def dyyy4_m24(solobj=None, field=None, dy3=None):
  if solobj is not None:
    field, dy3 = solobj.field, solobj.dy()**3
  return (-np.roll(field,2,1) - 8*np.roll(field,1,1) + 35*field - 48*np.roll(field,-1,1) + 29*np.roll(field,-2,1) -8*np.roll(field,-3,1) + np.roll(field,-4,1) )/(8*dy3)

def dyy4_m22(solobj=None, field=None, dy2=None):
  if solobj is not None:
    field, dy2 = solobj.field, solobj.dy()**2
  fdm = 0
  center = np.array([0,2])
  stencil = np.array([[-1,16,-30,16,-1]])/12.0
  #i,j stencil index
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      #-ix, -iy np.roll index
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dy2
  return fdm

def laplace2_m11m11(solobj=None, field=None, dy2=None):
  if solobj is not None:
    field, dy2 = solobj.field, solobj.dy()**2
  fdm = 0
  center = np.array([1,1])
  stencil = np.array([[0,1,0],
                      [1,-4,1],
                      [0,1,0]])
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dy2
  return fdm

def dxdyy4_m22m22(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx()*solobj.dy()**2
  fdm = 0
  center = np.array([2,2])
  stencil = np.array([[-1,16,-30,16,-1],
                      [8,-128,240,-128,8],
                      [0,0,0,0,0],
                      [-8,128,-240,128,-8],
                      [1,-16,30,-16,1]])/144
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx3
  return fdm

#biharmonic fdm 4th order, inner stencil
def biharm4_m33m33(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.field, solobj.dx()**4
  fdm = 0
  center = np.array([3,3])
  stencil = np.array([[0,0,0,-1,0,0,0],
                      [0,0,-1,14,-1,0,0],
                      [0,-1,20,-77,20,-1,0],
                      [-1,14,-77,184,-77,14,-1],
                      [0,-1,20,-77,20,-1,0],
                      [0,0,-1,14,-1,0,0],
                      [0,0,0,-1,0,0,0]])/6
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx4
  return fdm

#biharmonic fdm 4th order, next to left boundary
def biharm4_m33m25(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.field, solobj.dx()**4
  fdm = 0
  center = np.array([3,2])
  stencil = np.array([[0,0,-1,0,0,0,0,0],
                      [0,-1,14,-1,0,0,0,0],
                      [-1,20,-77,20,-1,0,0,0],
                      [6,-49,128,-7,-42,27,-8,1],
                      [-1,20,-77,20,-1,0,0,0],
                      [0,-1,14,-1,0,0,0,0],
                      [0,0,-1,0,0,0,0,0]])/6
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx4
  return fdm

#biharmonic fdm 4th order, next to upper boundary
def biharm4_m25m33(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.field, solobj.dx()**4
  fdm = 0
  center = np.array([2,3])
  stencil = np.array([[0,0,-1,0,0,0,0,0],
                      [0,-1,14,-1,0,0,0,0],
                      [-1,20,-77,20,-1,0,0,0],
                      [6,-49,128,-7,-42,27,-8,1],
                      [-1,20,-77,20,-1,0,0,0],
                      [0,-1,14,-1,0,0,0,0],
                      [0,0,-1,0,0,0,0,0]])/6
  stencil = np.rot90(stencil,axes=(1,0))
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx4
  return fdm