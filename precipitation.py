import numpy as np
from pathlib import Path
import cuda
from simulation import Simulation
from solution import solution

def DefaultParams(object):
  print("Applying DefaultParams")
  object.nof = 4
  # model
  object.v = 0.2
  object.ups0 = 0.0
  object.chi = 1.5
  object.ups1 = 0.0
  object.ups2 = 1.0
  object.ups3 = -5.0
  object.g = 10**-3
  object.beta = 2.0
  object.lamb = 1.8
  object.LAMB = 1.4
  object.sigma = 1.8
  object.alpha = 0.0
  # BC
  object.bc = 1
  # BC/IC
  object.h0 = 20.0
  object.c0 = 0.3
  object.phi0 = 1.0
  # IC
  object.noise = 0.0
  object.h1 = 0.0

# methods and attributes that apply to all precipiti problems
class precipiti(solution):
  def __init__(self, path = None):
    super().__init__()
    if path is not None:
      self.path = Path(path)
      # try:
      self.readparams(self.path)
      self.set_coordinates()
      self.fields = cuda.readbin(self)
      self.nof = len(self.fields)
      if(self.nof >1):
        self.set_psi1()
        self.set_psi2()
        self.set_C()
      self.set_h()
      self.set_dfdh()
    else:
      DefaultParams(self)
      # except FileNotFoundError:
        # print("no corresponding .dat and/or .bin file")

  def readparams(self, filepath=None):
    if filepath is None:
      filepath = self.path
    filepath = str(cuda.dat(filepath))
    # print(filepath)
    with open(filepath,'r') as f:
      lines = f.readlines()		#list, not array
    for i in np.arange(len(lines)):
      if lines[i].split()[0] == 'Nx':
        self.Nx = int(lines[i].split()[1])
      elif lines[i].split()[0] == 'Ny':
        self.Ny = int(lines[i].split()[1])
      elif lines[i].split()[0] == 'Lx':
        self.Lx = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'Ly':
        self.Ly = float(lines[i].split()[1])
      elif lines[i].split()[0] == 't':
        self.t = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'dt':
        self.dt = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'imagenumber':
        self.imagenumber = int(lines[i].split()[1])
      elif lines[i].split()[0] == 'h0':
        self.h0 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'v':
        self.v = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'ups0':
        self.ups0 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'chi':
        self.chi = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'ups1':
        self.ups1 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'ups2':
        self.ups2 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'ups3':
        self.ups3 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'g':
        self.g = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'beta':
        self.beta = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'lamb':
        self.lamb = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'LAMB':
        self.LAMB = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'sigma':
        self.sigma = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'alpha':
        self.alpha = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'c0':
        self.C0 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'Ceq':
        self.Ceq = float(lines[i].split()[1])
  def set_psi1(self):
    self.psi1 = self.fields[0]
  def set_psi2(self):
    self.psi2 = self.fields[1]
  def set_h(self):
    if(self.nof >1):
      self.h = self.fields[0] + self.fields[1]
    else:
      self.h = self.fields[0]
  def set_C(self):
    self.C = self.fields[1]/(self.fields[0] + self.fields[1])
  def set_DeltaC0(self):
    self.DeltaC0 = self.fields[1]/(self.fields[0] + self.fields[1]) - self.C0
  def set_phi(self):
    self.phi = self.fields[2]
  def set_zeta(self):
    self.zeta = self.fields[3]
  def set_dfdh(self):
    self.dfdh = self.h**(-3) - self.h**(-6)
  def set_dyh(self):
    self.dyh = self.dy4_m22(self.h)
  def set_dyyh(self):
    self.dyyh = self.dyy4_m22(self.h)
  def set_dyyyh(self):
    self.dyyyh = self.dyyy4_m33(self.h)
  def set_pressure(self):
    if(hasattr(self, "dyyh")):
      self.pressure = -self.dyyh + self.dfdh
    else:
      self.pressure = -cuda.dyy4_m22(field = self.h, dx2 = self.dx2()) + self.dfdh
  def set_dydfdh(self):
    self.dydfdh = self.dy4_m22(self.dfdh)
  def set_Gy(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    self.Gy = self.g*(self.dyh + self.beta)
  def set_dyPressure(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.dyPressure = self.dyyyh - dydfdh - self.g*(self.dyh + self.beta)
  def set_conv(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv = (self.h**3)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta)) + self.v*self.h   
  def set_conv1(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv1 = (self.psi1*self.h**2)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta)) + self.v*self.psi1
  def set_conv1Comoving(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv1Comoving = (self.psi1*self.h**2)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta))
  def set_conv2(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv2 = (self.psi2*self.h**2)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta)) + self.v*self.psi2
  def set_conv2Comoving(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv2Comoving = (self.psi2*self.h**2)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta))
  def set_diff1(self):
    dyC1 = self.dy4_m22((1-self.C))
    self.diff1 = -self.ups0*self.h*(1-2*self.chi*(1-self.C)*self.C)*dyC1
  def set_diff2(self):
    dyC2 = self.dy4_m22(self.C)
    self.diff2 = -self.ups0*self.h*(1-2*self.chi*(1-self.C)*self.C)*dyC2
  def set_osmo(self):
    self.osmo = self.ups2*(np.log(1-self.C) - 1 + self.chi*self.C*self.C)
  def set_evap(self):
    if(self.nof>1):
      if(hasattr(self, "osmo") and hasattr(self, "dyyh")):
        self.evap = -self.ups1*(-self.dyyh + self.dfdh  + self.osmo - self.ups3)
      else:
        self.evap = -self.ups1*(-self.dyy4_m22(self.h) + self.dfdh  + self.ups2*(np.log(1-self.C) - 1 + self.chi*self.C*self.C) - self.ups3)
    else:
      if not hasattr(self, "dyyh"):
        self.set_dyyh()
      self.evap = -self.ups1*(-self.dyyh + self.dfdh  - self.ups3)
  def set_MaskedEvap(self):
    if(hasattr(self, "osmo") and hasattr(self, "dyyh")):
      self.MaskedEvap = self.ups1*(-self.dyyh + self.dfdh  + self.osmo - self.ups3)
    else:
      self.MaskedEvap = self.ups1*(-self.dyy4_m22(self.h) + self.dfdh  + self.ups2*(np.log(1-self.C) - 1 + self.chi*self.C*self.C) - self.ups3)
    self.MaskedEvap = 0.5*(np.tanh(35.0*(self.h-1.1))+1.0)*self.MaskedEvap
  # overwrite method from ParentClass
  # fields can only be 2D array or 3d array with axis 0 being fieldnr
  def ApplyBC(self, fields=None):
    if(fields is None):
      fields = self.fields
    # need to always pre/append same number of ghost points in order to keep shape of numpy array
    # since numpy arrays cannot have inhomogeneous lengths within an axis
    # # BC for fields=self.fields (the case when fields is not a single 2D field but has shape (nof, Nx, Ny))
    # if(NameOfBC == 'Dipcoating4th'):

    #   if(nof==4):
    if(self.BC == 'DirichletNeumann'):
      YM1, YM2 = self.LeftDirichletSlope(fields, self.h0, -self.beta, self.dy())
      YN1, YN2 = self.RightNeumann(fields)
      fields = np.insert(fields, 0, YM1, axis = 1)
      fields = np.insert(fields, 0, YM2, axis = 1)
      fields = np.append(fields, np.reshape(YN1, (len(YN1),1)), axis = 1)
      fields = np.append(fields, np.reshape(YN1, (len(YN1),1)), axis = 1)
      # Top, Bottom, Left, Right "number" of ghost points
      return fields, None, None, 2, -2
    else:
      return super().ApplyBC(fields)

  # field must be 2d array
  def LeftDirichletSlope(self, field, dirichlet, slope, dx):
    YM1 = dirichlet
    YM2 = (12.0*dx*slope - 10.0*dirichlet + 18.0*field[:,0] - 6.0*field[:,1] + field[:,2])/3.0
    return YM1, YM2
  def RightNeumann(self, field):
    YN1 = (173.0*field[:,-1] - 94.0*field[:,-2] + 34.0*field[:,-3] - 8.0*field[:,-4] + field[:,-5])/106.
    YN2 = (89.0*field[:,-1] + 152.0*field[:,-2] - 117.0*field[:,-3] + 40.0*field[:,-4] - 5.0*field[:,-5])/159.
    return YN1, YN2
  
  
class PrecipitiSimu(Simulation):
  def __init__(self, path, start = None, end = None):
    super().__init__(path, start = None, end = None)
    self.objectclass = precipiti
  # expected equilibrium precursor height, depending on Ups, Mu, Chi and initial concentration c
  # requires 
  def set_hp(self, c):
    if(self.sols is not None):
      sol = self.sols[0]
      self.hp = (2.0/(1.0 + np.sqrt(1. + 4.*(-sol.ups3 + sol.ups2*(-1 + c*c*sol.chi+np.log(1-c))))))**(1./3.)
    else:
      print("Solutions not set!")

class XuMeakin(solution):
  def __init__(self, path = None):
    super().__init__()
    if path is not None:
      self.path = Path(path)
      # try:
      self.readparams(self.path)
      self.set_coordinates()
      self.fields = cuda.readbin(self)
      self.nof = len(self.fields)
      if(self.nof >1):
        self.set_C()
        self.set_phi()
    else:
      DefaultParams(self)
      # except FileNotFoundError:
        # print("no corresponding .dat and/or .bin file")

  def readparams(self, filepath=None):
    if filepath is None:
      filepath = self.path
    filepath = str(cuda.dat(filepath))
    # print(filepath)
    with open(filepath,'r') as f:
      lines = f.readlines()		#list, not array
    for i in np.arange(len(lines)):
      if lines[i].split()[0] == 'Nx':
        self.Nx = int(lines[i].split()[1])
      elif lines[i].split()[0] == 'Ny':
        self.Ny = int(lines[i].split()[1])
      elif lines[i].split()[0] == 'Lx':
        self.Lx = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'Ly':
        self.Ly = float(lines[i].split()[1])
      elif lines[i].split()[0] == 't':
        self.t = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'dt':
        self.dt = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'imagenumber':
        self.imagenumber = int(lines[i].split()[1])
      elif lines[i].split()[0] == 'lamb':
        self.lamb = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'PeXM':
        self.PeXM = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'alpha':
        self.alpha = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'c0':
        self.C0 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'Ceq':
        self.Ceq = float(lines[i].split()[1])

  def set_C(self):
    self.C = self.fields[0]
  def set_phi(self):
    self.phi = self.fields[1]
  def set_dxphi(self):
    self.dxphi = cuda.dx4_m22(self.phi, self.dx())
  def set_dyphi(self):
    self.dyphi = cuda.dy4_m22(self.phi, self.dx())
  def set_dxxphi(self):
    self.dxxphi = cuda.dxx4_m22(self.phi, self.dx2())
  def set_dxyphi(self):
    self.dxyphi = cuda.dxy4_m22_m22(self.phi, self.dx2())
  def set_dyyphi(self):
    self.dyyphi = cuda.dyy4_m22(self.phi, self.dx2())
  def set_lapcurv(self):
    if not (hasattr(self, "dxphi") and hasattr(self, "dyphi") and hasattr(self, "dxxphi") and hasattr(self, "dxyphi") and hasattr(self, "dyyphi")):
      self.set_dxphi()
      self.set_dyphi()
      self.set_dxxphi()
      self.set_dxyphi()
      self.set_dyyphi()
    dxphi, dyphi, dxxphi, dxyphi, dyyphi = self.dxphi, self.dyphi, self.dxxphi, self.dxyphi, self.dyyphi
    expression1 = np.divide(dyyphi*dyphi**2,dyphi**2 + dxphi**2, out=np.zeros_like(self.phi), where=( dyphi!=0 ))
    expression2 = np.divide(dxxphi*dxphi**2,dxphi**2 + dyphi**2, out=np.zeros_like(self.phi), where=( dxphi!=0 ))
    expression3 = 2.0*np.divide(dxyphi*dxphi*dyphi,dxphi**2 + dyphi**2, out=np.zeros_like(self.phi), where=((dxphi!=0) | (dyphi!=0)))
    self.lapcurv = expression1 + expression2 + expression3
    # self.lapcurv = dyyphi/(1.0 + dxphi/dyphi*dxphi/dyphi) + dxxphi/(1.0 + dyphi/dxphi*dyphi/dxphi) + 2.0*dxyphi/(dxphi/dyphi + dyphi/dxphi)