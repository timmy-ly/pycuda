import numpy as np
import warnings
from pathlib import Path
# scipy in interpolation

# methods that apply to all cuda problems
# add .dat to filepath
def dat(filepath):
  return filepath.with_suffix('.dat')

# add .bin to filepath
def bin(filepath):
  return filepath.with_suffix('.bin')

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
  filepath = str(bin(filepath))
  # print(filepath)
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
  from scipy.interpolate import RectBivariateSpline as rbs
  x = np.linspace(0,Lx,Nx)
  y = np.linspace(0,Ly,Ny)
  newx = np.linspace(0,Lx,newNx)
  newy = np.linspace(0,Ly,newNy)
  interpolatedobject = rbs(x,y,array,bbox=[0,Lx,0,Ly])
  # evaluate interpolatedobject at newx, newy points
  return interpolatedobject.__call__(newx,newy)

# finite difference stencils

# 4th order
# dy fdm 4th order, forward, copied from cuda code
def dy4_04(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.field, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -25.0*f(0,0) + 48.0*f(0,1) - 36.0*f(0,2) + 16.0*f(0,3) - 3.0*f(0,4) )/12.0/dx
# dy fdm 4th order, center, copied from cuda code
def dy4_m22(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.field, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (-f(0,2) + 8.0*f(0,1) - 8.0*f(0,-1) + f(0,-2) )/12.0/dx
# dy fdm 4th order, 1 left node, copied from cuda code
def dy4_m13(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.field, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (-3.0*f(0,-1) - 10.0*f(0,0) + 18.0*f(0,1) - 6.0*f(0,2) + f(0,3) )/12.0/dx
# dy fdm 4th order, 1 right node, copied from cuda code
def dy4_m31(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.field, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(0,-3) + 6.0*f(0,-2) - 18.0*f(0,-1) + 10.0*f(0,0) + 3.0*f(0,1) )/12.0/dx
# dxy fdm 4th order, center, copied from cuda code
def dxy4_m22_m22(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.field, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( f(-2,-2) - f(-2,2) - f(2,-2) + f(2,2)
         + 8.0*(-f(-1,-2) - f(-2,-1) + f(-2,1) + f(-1,2) + f(1,-2) + f(2,-1) - f(2,1) - f(1,2))
         + 64.0*(f(-1,-1) - f(-1,1) - f(1,-1) + f(1,1)) )/144.0/dx2
# dxy fdm 4th order, 1 left node, copied from cuda code
def dxy4_m22_m13(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.field, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 2.0*(-f(-2,-1) + f(2,-1)) + 3.0*(-f(-2,0) + f(2,0) - f(-1,3) + f(1,3)) + 6.0*(f(-2,1) - f(2,1))
          - f(-2,2) + f(2,2) + 13.0*(f(-1,-1) - f(1,-1)) + 36.0*(f(-1,0) - f(1,0)) 
          + 66.0*(-f(-1,1) + f(1,1)) + 20.0*(f(-1,2) - f(1,2)) )/72.0/dx2
# dxy fdm 4th order, 1 right node, copied from cuda code
def dxy4_m22_m31(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.field, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( f(-2,-2) - f(2,-2) + 6.0*(-f(-2,-1) + f(2,-1)) + 3.0*(f(-2,0) + f(-1,-3) - f(1,-3) - f(2,0))
          + 2.0*(f(-2,1) - f(2,1)) + 20.0*(-f(-1,-2) + f(1,-2)) + 66.0*(f(-1,-1) - f(1,-1))
          + 36.0*(-f(-1,0) + f(1,0)) + 13.0*(-f(-1,1) + f(1,1)) )/72.0/dx2
# dyy fdm 4th order, 1 left node, copied from cuda code
def dyy4_m14(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.field, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 10.0*f(0,-1) - 15.0*f(0,0) - 4.0*f(0,1) + 14.0*f(0,2) - 6.0*f(0,3) + f(0,4) )/12.0/dx2
# dyy fdm 4th order, 1 right node, copied from cuda code
def dyy4_m41(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.field, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( f(0,-4) - 6.0*f(0,-3) + 14.0*f(0,-2) - 4.0*f(0,-1) - 15.0*f(0,0) + 10.0*f(0,1) )/12.0/dx2
# laplace fdm 4th order, center, copied from cuda code
def laplace4_m22_m22(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.field, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (-60.0*f(0,0) + 16.0*(f(1,0) + f(0,1) + f(-1,0) + f(0,-1))
         - (f(2,0) + f(0,2) + f(-2,0) + f(0,-2)) )/12.0/dx2
# dyyy fdm 4th order, center, copied from cuda code
def dyyy4_m33(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (-f(0,3) + 8.0*f(0,2) - 13.0*f(0,1) + 13.0*f(0,-1) - 8.0*f(0,-2) + f(0,-3))/8.0/dx3
# dyyy fdm 4th order, node -2 to 4, copied from cuda code
def dyyy4_m24(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (f(0,4) - 8.0*f(0,3) + 29.0*f(0,2) - 48.0*f(0,1)
          +35.0*f(0,0) - 8.0*f(0,-1) - f(0,-2) )/8.0/dx3
# dyyy fdm 4th order, node -4 to 2, copied from cuda code
def dyyy4_m42(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (f(0,2) + 8.0*f(0,1) - 35.0*f(0,0) + 48.0*f(0,-1)
          -29.0*f(0,-2) + 8.0*f(0,-3) - f(0,-4) )/8.0/dx3
# dxyy fdm 4th order, center, copied from cuda code
def dxyy4_m22_m22(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(-2,-2) - f(-2,2) + f(2,-2) + f(2,2)
          + 16.0*(f(-2,-1) + f(-2,1) - f(2,-1) - f(2,1))
          + 8.0*(f(-1,-2) + f(-1,2) - f(1,-2) - f(1,2))
          + 30.0*(-f(-2,0) + f(2,0))
          + 128.0*(-f(-1,-1) - f(-1,1) + f(1,-1) + f(1,1))
          + 240.0*(f(-1,0) - f(1,0)) )/144.0/dx3
# dyxx fdm 4th order, center, copied from cuda code
def dyxx4_m22_m22(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(-2,-2) + f(-2,2) - f(2,-2) + f(2,2)
          + 16.0*(f(-1,-2) - f(-1,2) + f(1,-2) - f(1,2))
          + 8.0*(f(-2,-1) - f(-2,1) + f(2,-1) - f(2,1))
          + 30.0*(-f(0,-2) + f(0,2))
          + 128.0*(-f(-1,-1) + f(-1,1) - f(1,-1) + f(1,1))
          + 240*(f(0,-1) - f(0,1)) )/144.0/dx3
# biharm fdm 4th order, center, copied from cuda code
def biharm4_m33_m33(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.field, solobj.dx4()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -(f(0,3) + f(0,-3) + f(3,0) + f(-3,0))
             + 14.0*(f(0,2) + f(0,-2) + f(2,0) + f(-2,0))
             -77.0*(f(0,1) + f(0,-1) + f(1,0) + f(-1,0))
             +184.0*f(0,0)
             + 20.0*(f(1,1) + f(1,-1) + f(-1,1) + f(-1,-1))
             -(f(1,2) + f(2,1) + f(1,-2) + f(2,-1) + f(-1,2) + f(-2,1)
             + f(-1,-2) + f(-2,-1)) )/6.0/dx4
# biharm fdm 4th order, ynode -2 to 5, copied from cuda code
def biharm4_m33_m25(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.field, solobj.dx4()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(-3,0) - f(-2,-1) - f(-2,1) - f(-1,-2) - f(-1,2) + f(0,5) - f(1,-2) - f(1,2) - f(2,-1) - f(2,1) - f(3,0)
          + 14.0*(f(-2,0) + f(2,0))
          + 20.0*(f(-1,-1) + f(-1,1) + f(1,-1) + f(1,1))
          + 77.0*(-f(-1,0) - f(1,0))
          + 6.0*f(0,-2) - 49.0*f(0,-1) + 128.0*f(0,0) - 7.0*f(0,1) - 42.0*f(0,2) + 27.0*f(0,3) - 8.0*f(0,4) )/6.0/dx4
# biharm fdm 4th order, ynode -5 to 2, copied from cuda code
def biharm4_m33_m52(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.field, solobj.dx4()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(-3,0) - f(-2,-1) - f(-2,1) - f(-1,-2) - f(-1,2) + f(0,-5) - f(1,-2) - f(1,2) - f(2,-1) - f(2,1) - f(3,0)
          + 14.0*(f(-2,0) + f(2,0))
          + 20.0*(f(-1,-1) + f(-1,1) + f(1,-1) + f(1,1))
          + 77.0*(-f(-1,0) - f(1,0))
          + 6.0*f(0,2) - 49.0*f(0,1) + 128.0*f(0,0) - 7.0*f(0,-1) - 42.0*f(0,-2) + 27.0*f(0,-3) - 8.0*f(0,-4) )/6.0/dx4


# 6th order
#dx fdm 6th order, center, not copied from cuda code (preprocessor function)
def dx6_m33(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.field, solobj.dx()
  fdm = 0
  center = np.array([3,0])
  stencil = np.array([[-1,9,-45,0,45,-9,1]])/60
  stencil = np.transpose(stencil)
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx
  return fdm
# dy fdm 6th order, 2 left nodes, copied from cuda code
def dy6_m24(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.field, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 2.0*f(0,-2) - 24.0*f(0,-1) - 35.0*f(0,0) + 80.0*f(0,1) - 30.0*f(0,2) + 8.0*f(0,3) - f(0,4) )/60.0/dx
# dy fdm 6th order, center, copied from cuda code
def dy6_m33(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.field, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(0,-3) + 9.0*f(0,-2) - 45.0*f(0,-1) + 45.0*f(0,1) - 9.0*f(0,2) + f(0,3) )/60.0/dx
# dxy fdm 6th order, center, copied from cuda code
def dxy6_m33_m33(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.field, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 6.0*(f(-3,-1) - f(-3,1) + f(-1,-3) - f(-1,3) - f(1,-3) + f(1,3) - f(3,-1) + f(3,1))
         + 5.0*(f(-2,-2) - f(-2,2) - f(2,-2) + f(2,2))
         + 64.0*(-f(-2,-1) + f(-2,1) - f(-1,-2) + f(-1,2) + f(1,-2) - f(1,2) + f(2,-1) - f(2,1))
         + 380.0*(f(-1,-1) - f(-1,1) - f(1,-1) + f(1,1)) )/720.0/dx2
# laplace fdm 6th order, center, copied from cuda code
def laplace6_m33_m33(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.field, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 2.0*(f(-3,0) + f(0,-3) + f(0,3) + f(3,0))
          + 27.0*(-f(-2,0) - f(0,-2) - f(0,2) - f(2,0))
          + 270.0*(f(-1,0) + f(0,-1) + f(0,1) + f(1,0))
          - 980.0*f(0,0) )/180.0/dx2
#dxxx fdm 6th order, center, not copied from cuda code (preprocessor function)
def dxxx6_m44(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  fdm = 0
  center = np.array([4,0])
  stencil = np.array([[-7, 72, -338, 488, 0, -488, 338, -72, 7]])/240
  stencil = np.transpose(stencil)
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx3
  return fdm
# dyyy fdm 6th order, center, copied from cuda code
def dyyy6_m44(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -7.0*f(0,-4) + 72.0*f(0,-3) - 338.0*f(0,-2) + 488.0*f(0,-1) - 488.0*f(0,1) + 338.0*f(0,2) - 72.0*f(0,3) + 7.0*f(0,4) )/240.0/dx3
# dxyy fdm 6th order, center, copied from cuda code
def dxyy6_m33_m33(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 12.0*(-f(-3,-1) - f(-3,1) + f(3,-1) + f(3,1))
          + 5.0*(-f(-2,-2) - f(-2,2) + f(2,-2) + f(2,2))
          + 4.0*(-f(-1,-3) - f(-1,3) + f(1,-3) + f(1,3))
          + 64.0*(f(-1,-2) + f(-1,2) - f(1,-2) - f(1,2))
          + 128.0*(f(-2,-1) + f(-2,1) - f(2,-1) - f(2,1))
          + 760.0*(-f(-1,-1) - f(-1,1) + f(1,-1) + f(1,1))
          + 24.0*(f(-3,0) - f(3,0)) + 246.0*(-f(-2,0) + f(2,0))
          + 1400.0*(f(-1,0) - f(1,0)) )/720.0/dx3
# dyxx fdm 6th order, center, copied from cuda code
def dyxx6_m33_m33(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.field, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 12.0*(-f(-1,-3) - f(1,-3) + f(-1,3) + f(1,3))
          + 5.0*(-f(-2,-2) - f(2,-2) + f(-2,2) + f(2,2))
          + 4.0*(-f(-3,-1) - f(3,-1) + f(-3,1) + f(3,1))
          + 64.0*(f(-2,-1) + f(2,-1) - f(-2,1) - f(2,1))
          + 128.0*(f(-1,-2) + f(1,-2) - f(-1,2) - f(1,2))
          + 760.0*(-f(-1,-1) - f(1,-1) + f(-1,1) + f(1,1))
          + 24.0*(f(0,-3) - f(0,3)) + 246.0*(-f(0,-2) + f(0,2))
          + 1400.0*(f(0,-1) - f(0,1)) )/720.0/dx3
# biharm fdm 6th order, center, copied from cuda code
def biharm6_m44_m44(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.field, solobj.dx4()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 21.0*(f(-4,0) + f(0,-4) + f(0,4) + f(4,0))
          + 16.0*(f(-3,-1) + f(-3,1) + f(-1,-3) + f(-1,3) + f(1,-3) + f(1,3) + f(3,-1) + f(3,1))
          + 10.0*(f(-2,-2) + f(-2,2) + f(2,-2) + f(2,2))
          + 320.0*(-f(-3,0) - f(0,-3) - f(0,3) - f(3,0))
          + 256.0*(-f(-2,-1) - f(-2,1) - f(-1,-2) - f(-1,2) - f(1,-2) - f(1,2) - f(2,-1) - f(2,1))
          + 2520.0*(f(-2,0) + f(0,-2) + f(0,2) + f(2,0))
          + 3040.0*(f(-1,-1) + f(-1,1) + f(1,-1) + f(1,1))
          + 11456.0*(-f(-1,0) - f(0,-1) - f(0,1) - f(1,0))
          + 26660.0*f(0,0) )/720.0/dx4






# manually typed stencils, not copied from cuda, should be correct but cannot check for errors done within the cuda stencils
# # forward difference from node 0 to node 4
# def dy4_04(solobj = None, field = None, dy = None):
#   if solobj is not None:
#     field, dy = solobj.field, solobj.dy()
#   return (-25*field + 4*12*np.roll(field,-1,1) - 3*12*np.roll(field,-2,1) + 4/3*12*np.roll(field,-3,1) - 12/4*np.roll(field,-4,1))/(12*dy)

# # node -1 to node 3
# def dy4_m13(solobj = None, field = None, dy = None):
#   if solobj is not None:
#     field, dy = solobj.field, solobj.dy()
#   return (-3*np.roll(field,1,1) - 10*field + 18*np.roll(field,-1,1) - 6*np.roll(field,-2,1) + np.roll(field,-3,1) )/(12*dy)

# # node -2 to node 2
# def dy4_m22(solobj = None, field = None, dy = None):
#   if solobj is not None:
#     field, dy = solobj.field, solobj.dy()
#   return (np.roll(field,2,1) - 8*np.roll(field,1,1) + 0 + 8*np.roll(field,-1,1) - np.roll(field,-2,1) )/(12*dy)

# # node 0 to 6
# def dyyy4_06(solobj=None, field=None, dy3=None):
#   if solobj is not None:
#     field, dy3 = solobj.field, solobj.dy()**3
#   return (-49*field + 29*8*np.roll(field,-1,1) - 461*np.roll(field,-2,1) + 62*8*np.roll(field,-3,1) - 307*np.roll(field,-4,1) + 13*8*np.roll(field,-5,1) - 15*np.roll(field,-6,1))/(8*dy3)

# # node -1 to node 5
# def dyyy4_m15(solobj=None, field=None, dy3=None):
#   if solobj is not None:
#     field, dy3 = solobj.field, solobj.dy()**3
#   return (-15*np.roll(field,1,1) + 56*field - 83*np.roll(field,-1,1) + 64*np.roll(field,-2,1) -29*np.roll(field,-3,1) + 8*np.roll(field,-4,1) - np.roll(field,-5,1) )/(8*dy3)

# # node -2 to node 4  
# def dyyy4_m24(solobj=None, field=None, dy3=None):
#   if solobj is not None:
#     field, dy3 = solobj.field, solobj.dy()**3
#   return (-np.roll(field,2,1) - 8*np.roll(field,1,1) + 35*field - 48*np.roll(field,-1,1) + 29*np.roll(field,-2,1) -8*np.roll(field,-3,1) + np.roll(field,-4,1) )/(8*dy3)

# def dyy4_m22(solobj=None, field=None, dy2=None):
#   if solobj is not None:
#     field, dy2 = solobj.field, solobj.dy()**2
#   fdm = 0
#   center = np.array([0,2])
#   stencil = np.array([[-1,16,-30,16,-1]])/12.0
#   #i,j stencil index
#   for i in np.arange(len(stencil)):
#     for j in np.arange(len(stencil[0])):
#       #-ix, -iy np.roll index
#       ix, iy = i - center[0], j - center[1]
#       fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dy2
#   return fdm

# def laplace2_m11m11(solobj=None, field=None, dy2=None):
#   if solobj is not None:
#     field, dy2 = solobj.field, solobj.dy()**2
#   fdm = 0
#   center = np.array([1,1])
#   stencil = np.array([[0,1,0],
#                       [1,-4,1],
#                       [0,1,0]])
#   for i in np.arange(len(stencil)):
#     for j in np.arange(len(stencil[0])):
#       ix, iy = i - center[0], j - center[1]
#       fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dy2
#   return fdm

# def dxdyy4_m22m22(solobj=None, field=None, dx3=None):
#   if solobj is not None:
#     field, dx3 = solobj.field, solobj.dx()*solobj.dy()**2
#   fdm = 0
#   center = np.array([2,2])
#   stencil = np.array([[-1,16,-30,16,-1],
#                       [8,-128,240,-128,8],
#                       [0,0,0,0,0],
#                       [-8,128,-240,128,-8],
#                       [1,-16,30,-16,1]])/144
#   for i in np.arange(len(stencil)):
#     for j in np.arange(len(stencil[0])):
#       ix, iy = i - center[0], j - center[1]
#       fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx3
#   return fdm

# #biharmonic fdm 4th order, inner stencil
# def biharm4_m33m33(solobj=None, field=None, dx4=None):
#   if solobj is not None:
#     field, dx4 = solobj.field, solobj.dx()**4
#   fdm = 0
#   center = np.array([3,3])
#   stencil = np.array([[0,0,0,-1,0,0,0],
#                       [0,0,-1,14,-1,0,0],
#                       [0,-1,20,-77,20,-1,0],
#                       [-1,14,-77,184,-77,14,-1],
#                       [0,-1,20,-77,20,-1,0],
#                       [0,0,-1,14,-1,0,0],
#                       [0,0,0,-1,0,0,0]])/6
#   for i in np.arange(len(stencil)):
#     for j in np.arange(len(stencil[0])):
#       ix, iy = i - center[0], j - center[1]
#       fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx4
#   return fdm

# #biharmonic fdm 4th order, next to left boundary
# def biharm4_m33m25(solobj=None, field=None, dx4=None):
#   if solobj is not None:
#     field, dx4 = solobj.field, solobj.dx()**4
#   fdm = 0
#   center = np.array([3,2])
#   stencil = np.array([[0,0,-1,0,0,0,0,0],
#                       [0,-1,14,-1,0,0,0,0],
#                       [-1,20,-77,20,-1,0,0,0],
#                       [6,-49,128,-7,-42,27,-8,1],
#                       [-1,20,-77,20,-1,0,0,0],
#                       [0,-1,14,-1,0,0,0,0],
#                       [0,0,-1,0,0,0,0,0]])/6
#   for i in np.arange(len(stencil)):
#     for j in np.arange(len(stencil[0])):
#       ix, iy = i - center[0], j - center[1]
#       fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx4
#   return fdm

# #biharmonic fdm 4th order, next to upper boundary
# def biharm4_m25m33(solobj=None, field=None, dx4=None):
#   if solobj is not None:
#     field, dx4 = solobj.field, solobj.dx()**4
#   fdm = 0
#   center = np.array([2,3])
#   stencil = np.array([[0,0,-1,0,0,0,0,0],
#                       [0,-1,14,-1,0,0,0,0],
#                       [-1,20,-77,20,-1,0,0,0],
#                       [6,-49,128,-7,-42,27,-8,1],
#                       [-1,20,-77,20,-1,0,0,0],
#                       [0,-1,14,-1,0,0,0,0],
#                       [0,0,-1,0,0,0,0,0]])/6
#   # possibly flips previous axis 0
#   stencil = np.rot90(stencil,axes=(1,0))
#   for i in np.arange(len(stencil)):
#     for j in np.arange(len(stencil[0])):
#       ix, iy = i - center[0], j - center[1]
#       fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx4
#   return fdm