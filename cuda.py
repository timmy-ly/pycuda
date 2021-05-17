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
def readbin(solobj=None, filepath=None, dtype=None, Nx=None, Ny=None):
  if (solobj is not None):
    filepath, dtype, Nx, Ny = solobj.path, solobj.dtype, solobj.Nx, solobj.Ny
  filepath = str(bin(filepath))
  # print(filepath)
  if dtype == 'float':
    array = np.fromfile(filepath,np.dtype('float32'))
    array = array.astype('float')
  else:
    array = np.fromfile(filepath,np.dtype('float64'))
    array = array.astype('double')
  array = array.reshape((-1,Nx,Ny))
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

def get_crosssection_y(solobj, IdxX=None):
  if(IdxX is None):
    IdxX = solobj.Nx//2
  # get crosssection of solobj.fields
  return solobj.fields[:,IdxX]
  
def get_crosssection_1field_y(field, IdxX=None):
  if(IdxX is None):
    IdxX = int(len(field)/2)
  # get crosssection of solobj.fields
  return field[IdxX]

def mass(solobj, fieldnr):
  return np.sum(solobj.fields[fieldnr])/(solobj.Nx*solobj.Ny)

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
# ? order
# curvature
def curvature(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.fields, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( (f(1,0) - f(0,0))/np.sqrt( (f(1,0) - f(0,0))**2 + ((f(1,1) + f(0,1) - f(1,-1) - f(0,-1))**2)/16.0 ) 
         - (f(0,0) - f(-1,0))/np.sqrt( (f(0,0) - f(-1,0))**2 + ((f(-1,1) + f(0,1) - f(-1,-1) - f(0,-1))**2)/16.0 )
         + (f(0,1) - f(0,0))/np.sqrt( (f(0,1) - f(0,0))**2 + ((f(1,1) + f(1,0) - f(-1,1) - f(-1,0))**2)/16.0 )
         - (f(0,0) - f(0,-1))/np.sqrt( (f(0,0) - f(0,-1))**2 + ((f(1,-1) + f(1,0) - f(-1,-1) - f(-1,0))**2)/16.0 ) )/dx

# 4th order
# dy fdm 4th order, forward, copied from cuda code
def dy4_04(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.fields, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -25.0*f(0,0) + 48.0*f(0,1) - 36.0*f(0,2) + 16.0*f(0,3) - 3.0*f(0,4) )/12.0/dx
# dy fdm 4th order, center, copied from cuda code
def dy4_m22(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.fields, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (-f(0,2) + 8.0*f(0,1) - 8.0*f(0,-1) + f(0,-2) )/12.0/dx
# dy fdm 4th order, 1 left node, copied from cuda code
def dy4_m13(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.fields, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (-3.0*f(0,-1) - 10.0*f(0,0) + 18.0*f(0,1) - 6.0*f(0,2) + f(0,3) )/12.0/dx
# dy fdm 4th order, 1 right node, copied from cuda code
def dy4_m31(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.fields, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(0,-3) + 6.0*f(0,-2) - 18.0*f(0,-1) + 10.0*f(0,0) + 3.0*f(0,1) )/12.0/dx
# dxy fdm 4th order, center, copied from cuda code
def dxy4_m22_m22(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( f(-2,-2) - f(-2,2) - f(2,-2) + f(2,2)
         + 8.0*(-f(-1,-2) - f(-2,-1) + f(-2,1) + f(-1,2) + f(1,-2) + f(2,-1) - f(2,1) - f(1,2))
         + 64.0*(f(-1,-1) - f(-1,1) - f(1,-1) + f(1,1)) )/144.0/dx2
# dxy fdm 4th order, 1 left node, copied from cuda code
def dxy4_m22_m13(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 2.0*(-f(-2,-1) + f(2,-1)) + 3.0*(-f(-2,0) + f(2,0) - f(-1,3) + f(1,3)) + 6.0*(f(-2,1) - f(2,1))
          - f(-2,2) + f(2,2) + 13.0*(f(-1,-1) - f(1,-1)) + 36.0*(f(-1,0) - f(1,0)) 
          + 66.0*(-f(-1,1) + f(1,1)) + 20.0*(f(-1,2) - f(1,2)) )/72.0/dx2
# dxy fdm 4th order, 1 right node, copied from cuda code
def dxy4_m22_m31(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( f(-2,-2) - f(2,-2) + 6.0*(-f(-2,-1) + f(2,-1)) + 3.0*(f(-2,0) + f(-1,-3) - f(1,-3) - f(2,0))
          + 2.0*(f(-2,1) - f(2,1)) + 20.0*(-f(-1,-2) + f(1,-2)) + 66.0*(f(-1,-1) - f(1,-1))
          + 36.0*(-f(-1,0) + f(1,0)) + 13.0*(-f(-1,1) + f(1,1)) )/72.0/dx2
# dyy fdm 4th order, 1 left node, copied from cuda code
def dyy4_m14(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 10.0*f(0,-1) - 15.0*f(0,0) - 4.0*f(0,1) + 14.0*f(0,2) - 6.0*f(0,3) + f(0,4) )/12.0/dx2
# dyy fdm 4th order, center
def dyy4_m22(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(0,-2) + 16.0*f(0,-1) - 30.0*f(0,0) + 16.0*f(0,1) -f(0,2) )/12.0/dx2
# dyy fdm 4th order, 1 right node, copied from cuda code
def dyy4_m41(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( f(0,-4) - 6.0*f(0,-3) + 14.0*f(0,-2) - 4.0*f(0,-1) - 15.0*f(0,0) + 10.0*f(0,1) )/12.0/dx2
# laplace fdm 4th order, center, copied from cuda code
def laplace4_m22_m22(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (-60.0*f(0,0) + 16.0*(f(1,0) + f(0,1) + f(-1,0) + f(0,-1))
         - (f(2,0) + f(0,2) + f(-2,0) + f(0,-2)) )/12.0/dx2
# dyyy fdm 4th order, center, copied from cuda code
def dyyy4_m33(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (-f(0,3) + 8.0*f(0,2) - 13.0*f(0,1) + 13.0*f(0,-1) - 8.0*f(0,-2) + f(0,-3))/8.0/dx3
# dyyy fdm 4th order, node -2 to 4, copied from cuda code
def dyyy4_m24(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (f(0,4) - 8.0*f(0,3) + 29.0*f(0,2) - 48.0*f(0,1)
          +35.0*f(0,0) - 8.0*f(0,-1) - f(0,-2) )/8.0/dx3
# dyyy fdm 4th order, node -4 to 2, copied from cuda code
def dyyy4_m42(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return (f(0,2) + 8.0*f(0,1) - 35.0*f(0,0) + 48.0*f(0,-1)
          -29.0*f(0,-2) + 8.0*f(0,-3) - f(0,-4) )/8.0/dx3
# dxyy fdm 4th order, center, copied from cuda code
def dxyy4_m22_m22(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
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
    field, dx3 = solobj.fields, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(-2,-2) + f(-2,2) - f(2,-2) + f(2,2)
          + 16.0*(f(-1,-2) - f(-1,2) + f(1,-2) - f(1,2))
          + 8.0*(f(-2,-1) - f(-2,1) + f(2,-1) - f(2,1))
          + 30.0*(-f(0,-2) + f(0,2))
          + 128.0*(-f(-1,-1) + f(-1,1) - f(1,-1) + f(1,1))
          + 240*(f(0,-1) - f(0,1)) )/144.0/dx3
# dyyyy fdm 4th order, 2 left nodes
def dyyyy4_m25(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.fields, solobj.dx4()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( f(0,5) - 8.0*f(0,4) + 27.0*f(0,3) - 44.0*f(0,2) + 31.0*f(0,1) - 11.0*f(0,-1) + 4.0*f(0,-2) )/6.0/dx4
# biharm fdm 4th order, center, copied from cuda code
def biharm4_m33_m33(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.fields, solobj.dx4()
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
    field, dx4 = solobj.fields, solobj.dx4()
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
    field, dx4 = solobj.fields, solobj.dx4()
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
    field, dx = solobj.fields, solobj.dx()
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
    field, dx = solobj.fields, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 2.0*f(0,-2) - 24.0*f(0,-1) - 35.0*f(0,0) + 80.0*f(0,1) - 30.0*f(0,2) + 8.0*f(0,3) - f(0,4) )/60.0/dx
# dy fdm 6th order, center, copied from cuda code
def dy6_m33(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.fields, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(0,-3) + 9.0*f(0,-2) - 45.0*f(0,-1) + 45.0*f(0,1) - 9.0*f(0,2) + f(0,3) )/60.0/dx
# dxy fdm 6th order, center, copied from cuda code
def dxy6_m33_m33(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 6.0*(f(-3,-1) - f(-3,1) + f(-1,-3) - f(-1,3) - f(1,-3) + f(1,3) - f(3,-1) + f(3,1))
         + 5.0*(f(-2,-2) - f(-2,2) - f(2,-2) + f(2,2))
         + 64.0*(-f(-2,-1) + f(-2,1) - f(-1,-2) + f(-1,2) + f(1,-2) - f(1,2) + f(2,-1) - f(2,1))
         + 380.0*(f(-1,-1) - f(-1,1) - f(1,-1) + f(1,1)) )/720.0/dx2
# dxy fdm 6th order, center, some uneven contributions in 6thorder=0
def dxy6_m33_m33_2(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -f(-3,-2) + f(-3,2) - f(-2,-3) + f(-2,3) + f(2,-3) - f(2,3) + f(3,-2) - f(3,2)
         + 8.0*(f(-3,-1) - f(-3,1) + f(-1,-3) - f(-1,3) - f(1,-3) + f(1,3) - f(3,-1) + f(3,1))
         + 13.0*(f(-2,-2) - f(-2,2) - f(2,-2) + f(2,2))
         + 77.0*(-f(-2,-1) + f(-2,1) - f(-1,-2) + f(-1,2) + f(1,-2) - f(1,2) + f(2,-1) - f(2,1))
         + 400.0*(f(-1,-1) - f(-1,1) - f(1,-1) + f(1,1)) )/720.0/dx2
# laplace fdm 6th order, center, copied from cuda code
def laplace6_m33_m33(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 2.0*(f(-3,0) + f(0,-3) + f(0,3) + f(3,0))
          + 27.0*(-f(-2,0) - f(0,-2) - f(0,2) - f(2,0))
          + 270.0*(f(-1,0) + f(0,-1) + f(0,1) + f(1,0))
          - 980.0*f(0,0) )/180.0/dx2
# dxx fdm 6th order, center
def dxx6_m33(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 2.0*f(-3,0) - 27.0*f(-2,0) + 270.0*f(-1,0) - 490.0*f(0,0) + 270.0*f(1,0) - 27.0*f(2,0) + 2.0*f(3,0) )/180.0/dx2
# dyy fdm 6th order, 2 left nodes, copied to cuda code
def dyy6_m25(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( - 11.0*f(0,-2) + 214.0*f(0,-1) - 378.0*f(0,0) + 130.0*f(0,1) + 85.0*f(0,2) - 54.0*f(0,3) + 16.0*f(0,4) - 2.0*f(0,5) )/180.0/dx2
# dyy fdm 6th order, center, copied from cuda code
def dyy6_m33(solobj=None, field=None, dx2=None):
  if solobj is not None:
    field, dx2 = solobj.fields, solobj.dx2()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 2.0*f(0,-3) - 27.0*f(0,-2) + 270.0*f(0,-1) - 490.0*f(0,0) + 270.0*f(0,1) - 27.0*f(0,2) + 2.0*f(0,3) )/180.0/dx2
#dxxx fdm 6th order, center, not copied from cuda code (preprocessor function)
def dxxx6_m44(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
  fdm = 0
  center = np.array([4,0])
  stencil = np.array([[-7, 72, -338, 488, 0, -488, 338, -72, 7]])/240
  stencil = np.transpose(stencil)
  for i in np.arange(len(stencil)):
    for j in np.arange(len(stencil[0])):
      ix, iy = i - center[0], j - center[1]
      fdm += stencil[i,j]*np.roll(field,(-ix,-iy),(0,1))/dx3
  return fdm
# dyyy fdm 6th order, 2 left nodes, copied to cuda code
def dyyy6_m26(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -5.0*f(0,-2) - 424.0*f(0,-1) + 1638.0*f(0,0) - 2504.0*f(0,1) + 2060.0*f(0,2) - 1080.0*f(0,3) + 394.0*f(0,4) - 88.0*f(0,5) + 9.0*f(0,6) )/240.0/dx3
# dyyy fdm 6th order, 3 left nodes, copied to cuda code
def dyyy6_m35(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 9.0*f(0,-3) - 86.0*f(0,-2) - 100.0*f(0,-1) + 882.0*f(0,0) - 1370.0*f(0,1) + 926.0*f(0,2) - 324.0*f(0,3) + 70.0*f(0,4) - 7.0*f(0,5) )/240.0/dx3
# dyyy fdm 6th order, center, copied from cuda code
def dyyy6_m44(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -7.0*f(0,-4) + 72.0*f(0,-3) - 338.0*f(0,-2) + 488.0*f(0,-1) - 488.0*f(0,1) + 338.0*f(0,2) - 72.0*f(0,3) + 7.0*f(0,4) )/240.0/dx3
# dxyy fdm 6th order, center, copied from cuda code
def dxyy6_m33_m33(solobj=None, field=None, dx3=None):
  if solobj is not None:
    field, dx3 = solobj.fields, solobj.dx3()
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
    field, dx3 = solobj.fields, solobj.dx3()
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
# dyyyy fdm 6th order, 2 left nodes, copied to cuda code
def dyyyy6_m27(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.fields, solobj.dx4()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 101.0*f(0,-2) + 58.0*f(0,-1) - 1860.0*f(0,0) + 5272.0*f(0,1) - 7346.0*f(0,2) + 6204.0*f(0,3) - 3428.0*f(0,4) + 1240.0*f(0,5) - 267.0*f(0,6) + 26.0*f(0,7) )/240.0/dx4
# dyyyy fdm 6th order, 3 left nodes, copied to cuda code
def dyyyy6_m36(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.fields, solobj.dx4()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( -26.0*f(0,-3) + 361.0*f(0,-2) - 1112.0*f(0,-1) + 1260.0*f(0,0) - 188.0*f(0,1) - 794.0*f(0,2) + 744.0*f(0,3) - 308.0*f(0,4) + 70.0*f(0,5) - 7.0*f(0,6) )/240.0/dx4
# biharm fdm 6th order, center, copied from cuda code
def biharm6_m44_m44(solobj=None, field=None, dx4=None):
  if solobj is not None:
    field, dx4 = solobj.fields, solobj.dx4()
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


# 8th order
# dy fdm 8th order, center
def dy8_m44(solobj=None, field=None, dx=None):
  if solobj is not None:
    field, dx = solobj.fields, solobj.dx()
  def f(ix,iy, field = field):
    return np.roll(field,(-ix,-iy),(0,1))
  return ( 3.0*f(0,-4) - 32.0*f(0,-3) + 168.0*f(0,-2) - 672.0*f(0,-1) + 672.0*f(0,1) - 168.0*f(0,2) + 32.0*f(0,3) - 3.0*f(0,4) )/840.0/dx
