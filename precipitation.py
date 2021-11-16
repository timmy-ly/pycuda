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
      self.set_SpatialGrid1Dy()
      self.fields = cuda.readbin(self)
      self.nof = len(self.fields)
      if(self.nof >1):
        self.set_psi1()
        self.set_psi2()
        self.set_C()
      if(self.nof >2):
        self.set_phi()
        self.set_zeta()
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
  ##### attributes with same shape as self.fields[i]
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
  def set_dyC1(self):
    self.dyC1 = cuda.dy4_m22((1-self.C),self.dx())
  def set_dyC2(self):
    self.dyC2 = cuda.dy4_m22(self.C,self.dx())
  def set_DeltaC0(self):
    self.DeltaC0 = self.fields[1]/(self.fields[0] + self.fields[1]) - self.C0
  def set_phi(self):
    self.phi = self.fields[2]
  def set_zeta(self):
    self.zeta = self.fields[3]
  def set_dfdh(self):
    self.dfdh = self.h**(-3) - self.h**(-6)
  def set_dypsi1(self):
    self.dypsi1 = self.dy4_m22(self.psi1)
  def set_dypsi2(self):
    self.dypsi2 = self.dy4_m22(self.psi2)
  def set_dyh(self):
    self.dyh = self.dy4_m22(self.h)
  def set_dyyh(self):
    self.dyyh = cuda.dyy4_m22(self.h,self.dx2())
  def set_dyyyh(self):
    self.dyyyh = self.dyyy4_m33(self.h)
  def set_dyyyyh(self):
    self.dyyyyh = cuda.dyyyy4_m33(self.h,self.dx4())
  def set_dyyphi(self):
    self.dyyphi = cuda.dyy4_m22(self.phi, self.dx2())
  def set_M(self):
    self.M = 1.0/3.0*self.h*self.h*self.h
  def set_M1(self):
    self.M1 = 1.0/3.0*self.psi1*self.h*self.h
  def set_M2(self):
    self.M2 = 1.0/3.0*self.psi2*self.h*self.h
  def set_dyM(self):
    if(not hasattr(self,"h")):
      self.set_dyh()
    self.dyM = 3.0*self.h*self.h*self.dyh
  def set_dyM1(self):
    if(not hasattr(self,"dypsi1")):
      self.set_dypsi1()
    self.dyM1 = 1.0/3.0*self.h*(self.h*self.dypsi1 + 2.0*self.psi1*self.dyh)
  def set_dyM2(self):
    if(not hasattr(self,"dypsi2")):
      self.set_dypsi2()
    self.dyM2 = 1.0/3.0*self.h*(self.h*self.dypsi2 + 2.0*self.psi2*self.dyh)
  def set_pressure(self):
    if(hasattr(self, "dyyh")):
      self.pressure = -self.dyyh + self.dfdh
    else:
      self.pressure = -cuda.dyy4_m22(field = self.h, dx2 = self.dx2()) + self.dfdh
  def set_dydfdh(self):
    self.dydfdh = self.dy4_m22(self.dfdh)
  def set_mask(self):
    self.mask = 0.5*(np.tanh(35.0*(self.h-1.1))+1.0)
  def set_Gy(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    self.Gy = self.g*(self.dyh + self.beta)
  def set_dyGy(self):
    if(not hasattr(self,"dyyh")):
      self.set_dyyh()
    self.dyGy = self.g*self.dyyh
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
  def set_convrate(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    dydfdh = cuda.dy4_m22(self.dfdh, self.dx())
    dyydfdh = cuda.dyy4_m22(self.dfdh, self.dx2())
    self.set_M()
    self.set_dyM()
    self.set_Gy()
    self.set_dyGy()
    self.convrate = self.dyM*(self.dyyyh - dydfdh) + self.M*(self.dyyyyh - dyydfdh) - self.dyM*self.Gy - self.M*self.dyGy + self.v*self.dyh
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
  def set_conv1rate(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    dydfdh = cuda.dy4_m22(self.dfdh, self.dx())
    dyydfdh = cuda.dyy4_m22(self.dfdh, self.dx2())
    self.set_M1()
    self.set_dyM1()
    self.set_Gy()
    self.set_dyGy()
    self.conv1rate = -( self.dyM1*(self.dyyyh - dydfdh) + self.M1*(self.dyyyyh - dyydfdh) - self.dyM1*self.Gy - self.M1*self.dyGy + self.v*self.dypsi1 )
  def set_conv1rateComoving(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    dydfdh = cuda.dy4_m22(self.dfdh, self.dx())
    dyydfdh = cuda.dyy4_m22(self.dfdh, self.dx2())
    self.set_M1()
    self.set_dyM1()
    self.set_Gy()
    self.set_dyGy()
    self.conv1rateComoving = -( self.dyM1*(self.dyyyh - dydfdh) + self.M1*(self.dyyyyh - dyydfdh) - self.dyM1*self.Gy - self.M1*self.dyGy )
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
  def set_conv2rate(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    dydfdh = cuda.dy4_m22(self.dfdh, self.dx())
    dyydfdh = cuda.dyy4_m22(self.dfdh, self.dx2())
    self.set_M2()
    self.set_dyM2()
    self.set_Gy()
    self.set_dyGy()
    self.conv2rate = -( self.dyM2*(self.dyyyh - dydfdh) + self.M2*(self.dyyyyh - dyydfdh) - self.dyM2*self.Gy - self.M2*self.dyGy + self.v*self.dypsi2 )
  def set_conv2rateComoving(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    dydfdh = cuda.dy4_m22(self.dfdh, self.dx())
    dyydfdh = cuda.dyy4_m22(self.dfdh, self.dx2())
    self.set_M2()
    self.set_dyM2()
    self.set_Gy()
    self.set_dyGy()
    self.conv2rateComoving = -( self.dyM2*(self.dyyyh - dydfdh) + self.M2*(self.dyyyyh - dyydfdh) - self.dyM2*self.Gy - self.M2*self.dyGy )
  def set_diff1(self):
    dyC1 = cuda.dy4_m22((1-self.C),self.dx())
    self.diff1 = -self.ups0*self.h*(1-2*self.chi*(1-self.C)*self.C)*dyC1
  def set_diff1rate(self):
    mdiff = -self.ups0*self.h*(1-2*self.chi*(1-self.C*self.C))
    dymdiff = cuda.dy4_m22(mdiff, self.dx())
    dyC1 = cuda.dy4_m22((1-self.C),self.dx())
    dyyC1 = cuda.dyy4_m22((1-self.C),self.dx2())
    self.diff1rate = -( dymdiff*dyC1 + mdiff*dyyC1 )
  def set_diff1rateMasked(self):
    self.CheckSetAttr("diff1rate", "mask")
    self.diff1rateMasked = self.mask*self.diff1rate
  def set_diff2(self):
    dyC2 = cuda.dy4_m22(self.C,self.dx())
    self.diff2 = -self.ups0*self.h*(1-2*self.chi*(1-self.C)*self.C)*dyC2
  def set_diff2rate(self):
    mdiff = -self.ups0*self.h*(1-2*self.chi*(1-self.C*self.C))
    dymdiff = cuda.dy4_m22(mdiff, self.dx())
    dyC2 = cuda.dy4_m22(self.C,self.dx())
    dyyC2 = cuda.dyy4_m22(self.C,self.dx2())
    self.diff2rate = -( dymdiff*dyC2 + mdiff*dyyC2 )
  def set_diff2rateMasked(self):
    self.set_diff2rate()
    self.set_mask()
    self.diff2rateMasked = self.mask*self.diff2rate
  def set_osmo(self):
    if(self.ups2==0):
      self.osmo = 0
    else:
      self.osmo = self.ups2*(np.log(1-self.C) - 1 + self.chi*self.C*self.C)
  def set_evap(self):
    if(self.nof>1):
      if not hasattr(self, "osmo"):
        self.set_osmo()
      if not hasattr(self, "dyyh"):
        self.set_dyyh()
      self.evap = -self.ups1*(-self.dyyh + self.dfdh  + self.osmo - self.ups3)
    else:
      if not hasattr(self, "dyyh"):
        self.set_dyyh()
      self.evap = -self.ups1*(-self.dyyh + self.dfdh  - self.ups3)
  def set_MaskedEvap(self):
    self.CheckSetAttr("evap", "mask")
    self.MaskedEvap = self.mask*self.evap
  def set_dfXMdphi(self):
    self.dfXMdphi = -(1.0 - self.phi*self.phi)*(self.phi - self.lamb*(self.C-self.Ceq))
  # this needs to be added for 2D!
  def set_MeanCurv(self):
    self.MeanCurv = 0.0
  # set time derivative of phi but without advection
  def set_DtPhiNoAdvec(self):
    self.CheckSetAttr("dyyphi", "dfXMdphi", "MeanCurv")
    self.DtPhiNoAdvec = self.sigma*(self.dyyphi/(self.LAMB*self.LAMB) - self.dfXMdphi - self.MeanCurv/(self.LAMB*self.LAMB))
  def set_dtpsi1(self):
    self.CheckSetAttr("conv1rateComoving", "diff1rateMasked", "MaskedEvap")
    self.dtpsi1 = self.conv1rateComoving + self.diff1rateMasked + self.MaskedEvap
  def set_dtpsi2(self):
    self.CheckSetAttr("conv2rateComoving", "diff2rateMasked", "DtPhiNoAdvec")
    self.dtpsi2 = self.conv2rateComoving + self.diff2rateMasked + self.alpha*self.h*self.DtPhiNoAdvec 

  ##### 0 dimensional attributes
  def mean(self, field):
    shape = np.shape(field)
    if(len(shape)==1):
      return np.sum(field)/shape[0]
    elif(len(shape)==2):
      return np.sum(field)/shape[0]/shape[1]
  def set_mean_h(self):
    self.mean_h = self.mean(self.h)
  def set_mean_C(self):
    self.mean_C = self.mean(self.C)
  # left or right half of the domain
  def WhichHalf(self, direction='right'):
    startidx = self.Ny//2
    endidx = None
    if(not hasattr(self, "y")):
      self.set_SpatialGrid1Dy()
    if(direction == 'left'):
      startidx = 0
      endidx = self.Ny//2
    return startidx, endidx
  # inflection point in 1D
  def set_yInflection(self, direction='right'):
    startidx, endidx = self.WhichHalf(direction)
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    dyh1D = self.transform_crosssection('dyh')
    yidx = np.argmax(np.abs(dyh1D[startidx:endidx]))
    self.yInflection = self.y[startidx:endidx][yidx]
  # todo: method for contact line position
  
  # position of local concentration maximum
  def set_yCMax(self, direction='right'):
    startidx, endidx = self.WhichHalf(direction)
    c1D = self.transform_crosssection('C')
    yidx = np.argmax(c1D[startidx:endidx])
    self.yCMax = self.y[startidx:endidx][yidx]
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
  def __init__(self, path, start = None, end = None, file = 'frame_0000.dat'):
    super().__init__(path, start = start, end = end, file = file)
    self.objectclass = precipiti
  # expected equilibrium precursor height, depending on Ups, Mu, Chi and initial concentration c
  # requires 
  def set_hp(self, c):
    if(self.sols is not None):
      sol = self.sols[0]
      self.hp = (2.0/(1.0 + np.sqrt(1. + 4.*(-sol.ups3 + sol.ups2*(-1 + c*c*sol.chi+np.log(1-c))))))**(1./3.)
    else:
      print("Solutions not set!")
  def get_OnsetOfPrecipitation(self, Threshold = 0.95):
    # self.set_CalculatedFields('set_DtPhiNoAdvec')
    for i,sol in enumerate(self.sols):
      if(sol.CheckIfBelowThreshold(sol.phi, Threshold)):
        break
    self.OnsetOfPrecipitationIndex = i

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

class XuMeakinSimu(Simulation):
  def __init__(self, path, start = None, end = None):
    super().__init__(path, start = None, end = None)
    self.objectclass = XuMeakin