import numpy as np
from pathlib import Path
import cuda
from simulation import Simulation, SimulMeasures, TransientError, IndexWindowError, SimulatedTooShortError
from solution import NoExtremaError, SolutionMeasures, solution, FieldProps
from scipy.signal import find_peaks

attribute = 'imagenumber'
class OnlyOneMinimumError(Exception):
  pass

class NoDepositError(Exception):
  pass
class OnsetTooLateError(Exception):
  pass
class PrecipitiMeasures(SolutionMeasures):
  def __init__(self):
    super().__init__()
    self.PeakIndex = None
    self.PeakPosition = None
    self.Height = None
    self.Prominence = None
    self.Base = None

class PrecipitiSimulMeasures(SimulMeasures):
  def __init__(self):
    super().__init__()
    self.t = []
    self.dt = None
    self.MaximumThickness = []
    self.BaseThickness = []
    self.Prominence = []
    self.MeanMaximumThickness = None
    self.MeanBaseThickness = None
    self.MeanProminence = None
  # save measures
  def SaveMeasures(self, PeakIndices, properties, t, ZetaOfT):
    self.PeakIndices = PeakIndices
    self.t = t
    self.ZetaOfT = ZetaOfT
    self.tPeaks = t[PeakIndices]
    self.MaximumThickness = np.array(properties['peak_heights'])[1:]
    self.Prominence = np.array(properties['prominences'])[1:]
    self.set_BaseThickness()
    self.TransformToPeriods()
  def TransformToPeriods(self):
    self.Periods = self.tPeaks - np.roll(self.tPeaks, 1)
    self.Periods = self.Periods[1:]
  # For the other quantities we drop the first peak in order to match the shape to the Periods
  # here, since we measure peaks, we have one less minimum which perfectly matches the shape
  # of periods. The +1 is there since in a[:i], a[i] is not contained. 
  def set_BaseThickness(self):
    self.BaseThickness = [np.min(self.ZetaOfT[self.PeakIndices[i]:self.PeakIndices[i+1]+1]) 
                          for i in range(len(self.PeakIndices)-1)]
  # calculate the periods and means
  def set_ZeroDimMeasures(self):
    TimeWindow = self.tPeaks[-1] - self.tPeaks[0]
    dt = self.t - np.roll(self.t, 1)
    self.MeanPeriod = np.mean(self.Periods)
    self.MeanMaximumThickness = np.sum(self.Periods*self.MaximumThickness)/TimeWindow
    self.MeanBaseThickness = np.sum(self.Periods*self.BaseThickness)/TimeWindow
    self.MeanProminence = np.sum(self.Periods*self.Prominence)/TimeWindow
    self.MeanDeposit = np.sum(dt[1:]*self.ZetaOfT[1:])/TimeWindow
  def CutOffTransient(self):
    if(not hasattr(self, 'EndOfTransient')):
      raise TransientError('EndOfTransient attribute does not exist. Try calling FindEndOfTransient') 
    n = self.EndOfTransient
    self.Periods = self.Periods[n:]
    self.MaximumThickness = self.MaximumThickness[n:]
    self.BaseThickness = self.BaseThickness[n:]
    self.Prominence = self.Prominence[n:]
    # len(tpeaks)=m, len(periods) = m-1 after [1:]
    # here, we filter periods[n:]. Since tpeaks is longer by 1 at the front, we should
    # use n+1 there
    mask = self.t>=self.tPeaks[n+1]
    self.t = self.t[mask]
    self.ZetaOfT = self.ZetaOfT[mask]

  # calculate the periods and means
  # legacy
  def set_PeriodicMeasures(self):
    self.ListsToArrays()
    # calculate Periods
    self.Periods = self.tPeaks - np.roll(self.tPeaks, 1)
    TimeWindow = self.tPeaks[-1] - self.tPeaks[0]
    # since numpy.roll uses periodic BC, we need to drop index 0 on the periods
    # to maintain shape we have to do this for the other quantities too
    self.Periods = self.Periods[1:]
    self.MaximumThickness = self.MaximumThickness[1:]
    self.BaseThickness = self.BaseThickness[1:]
    self.Prominence = self.Prominence[1:]
    self.MeanMaximumThickness = np.sum(self.Periods*self.MaximumThickness)/TimeWindow
    self.MeanBaseThickness = np.sum(self.Periods*self.BaseThickness)/TimeWindow
    self.MeanProminence = np.sum(self.Periods*self.Prominence)/TimeWindow
    self.MeanPeriod = np.mean(self.Periods)
  # save measures
  # legacy
  def SaveMeasuresToLists(self, sol):
    self.t.append(sol.t)
    self.MaximumThickness.append(sol.Measures.Max)
    self.BaseThickness.append(sol.Measures.MinHeight)
    self.Prominence.append(sol.Measures.Prominence)
  # legacy
  def ListsToArrays(self):
    self.t = np.array(self.t)
    self.MaximumThickness = np.array(self.MaximumThickness)
    self.BaseThickness = np.array(self.BaseThickness)
    self.Prominence = np.array(self.Prominence)


# def DefaultParams(object):
#   if(not object.silent):
#     print("Applying DefaultParams")
#   object.nof = 4
#   # model
#   object.v = 0.2
#   object.ups0 = 0.0
#   object.chi = 1.5
#   object.ups1 = 0.0
#   object.ups2 = 1.0
#   object.ups3 = -5.0
#   object.g = 10**-3
#   object.beta = 2.0
#   object.lamb = 1.8
#   object.LAMB = 1.4
#   object.sigma = 1.8
#   object.alpha = 0.0
#   # BC
#   object.bc = 1
#   # BC/IC
#   object.h0 = 20.0
#   object.c0 = 0.3
#   object.phi0 = 1.0
#   # IC
#   object.noise = 0.0
#   object.h1 = 0.0


# methods and attributes that apply to all precipiti problems
class precipiti(solution):
  def __init__(self, path = None, silent = False):
    super().__init__()
    if path is not None:
      self.path = Path(path)
      # try:
      self.readparams(self.path)
      self.set_SpatialDimensions()
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
        self.set_phi1D()
        self.set_zeta()
        self.set_zeta1D()
      self.set_h()
      # self.set_dfdh()
      # self.set_disjp()
      self.zeta1DProps = FieldProps()
      self.Measures = PrecipitiMeasures()
    else:
      if(not silent):
        print('no path provided, creating default solution object')
      # except FileNotFoundError:
        # print("no corresponding .dat and/or .bin file")

  def set_FieldProps(self, data, FieldName):
    FieldPropsName = FieldName + 'Props'
    setattr(self, FieldPropsName, FieldProps(data, FieldName))
  ##### attributes with same shape as self.fields[i]
  def set_psi1(self):
    self.psi1 = self.fields[0]
  def set_psi2(self):
    self.psi2 = self.fields[1]
  def set_psi2_1D(self, **kwargs):
    self.psi2_1D = self.get_crosssection_y(self.psi2, **kwargs)
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
  def set_phi1D(self, **kwargs):
    self.phi1D = self.get_crosssection_y(self.phi, **kwargs)
  def set_zeta(self):
    self.zeta = self.fields[3]
  def set_zeta1D(self, **kwargs):
    self.zeta1D = self.get_crosssection_y(self.zeta, **kwargs)
  def set_dfdh(self):
    self.dfdh = self.h**(-3) - self.h**(-6)
  def set_disjp(self):
    self.disjp = -self.h**(-3) + self.h**(-6)
  def set_dypsi1(self):
    self.dypsi1 = self.dy4_m22(self.psi1)
  def set_adv1(self):
    self.adv1 = cuda.dy4_m31(self.psi1, self.dx())
  def set_dypsi2(self):
    self.dypsi2 = self.dy4_m22(self.psi2)
  def set_adv2(self):
    self.adv2 = cuda.dy4_m31(self.psi2, self.dx())
  def set_advphi(self):
    self.advphi = cuda.dy4_m31(self.phi, self.dx())
  def set_advzeta(self):
    self.advzeta = cuda.dy4_m31(self.zeta, self.dx())
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
    if(not hasattr(self,"dyyh")):
      self.set_dyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    self.pressure = -self.dyyh + self.dfdh
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
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.dyPressure = self.dyyyh - dydfdh - self.g*(self.dyh + self.beta)
  def set_conv(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv = (self.h**3)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta)) + self.v*self.h   
  def set_convrate(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
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
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv1 = (self.psi1*self.h**2)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta)) + self.v*self.psi1
  def set_conv1Comoving(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv1Comoving = (self.psi1*self.h**2)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta))
  def set_conv1rate(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    dydfdh = cuda.dy4_m22(self.dfdh, self.dx())
    dyydfdh = cuda.dyy4_m22(self.dfdh, self.dx2())
    self.set_M1()
    self.set_dyM1()
    self.set_Gy()
    self.set_dyGy()
    self.conv1rate = -( self.dyM1*(self.dyyyh - dydfdh) + self.M1*(self.dyyyyh - dyydfdh) - self.dyM1*self.Gy - self.M1*self.dyGy + self.v*self.adv1 )
  def set_conv1rateComoving(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
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
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv2 = (self.psi2*self.h**2)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta)) + self.v*self.psi2
  def set_conv2Comoving(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    dydfdh = self.dy4_m22(self.dfdh)
    self.conv2Comoving = (self.psi2*self.h**2)/3.0*(self.dyyyh - dydfdh - self.g*(self.dyh + self.beta))
  def set_conv2rate(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    dydfdh = cuda.dy4_m22(self.dfdh, self.dx())
    dyydfdh = cuda.dyy4_m22(self.dfdh, self.dx2())
    self.set_M2()
    self.set_dyM2()
    self.set_Gy()
    self.set_dyGy()
    self.conv2rate = -( self.dyM2*(self.dyyyh - dydfdh) + self.M2*(self.dyyyyh - dyydfdh) - self.dyM2*self.Gy - self.M2*self.dyGy + self.v*self.adv2 )
  def set_conv2rateComoving(self):
    if(not hasattr(self,"dyh")):
      self.set_dyh()
    if(not hasattr(self,"dyyyh")):
      self.set_dyyyh()
    if(not hasattr(self,"dyyyyh")):
      self.set_dyyyyh()
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
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
      self.osmo = np.zeros((self.Nx, self.Ny))
    else:
      self.osmo = self.ups2*(np.log(1-self.C) - 1 + self.chi*self.C*self.C)
  def set_evap(self):
    if(not hasattr(self,"dfdh")):
      self.set_dfdh()
    if not hasattr(self, "dyyh"):
      self.set_dyyh()
    if(self.nof>1):
      if not hasattr(self, "osmo"):
        self.set_osmo()
      self.evap = -self.ups1*(-self.dyyh + self.dfdh  + self.osmo - self.ups3)
    else:
      self.evap = -self.ups1*(-self.dyyh + self.dfdh  - self.ups3)
  def set_MaskedEvap(self):
    self.CheckSetAttr("evap", "mask")
    self.MaskedEvap = self.mask*self.evap
  def set_dfXMdphi(self):
    self.dfXMdphi = -(1.0 - self.phi*self.phi)*(self.phi - self.lamb*(self.C-self.Ceq))
  # this needs to be added for 2D!
  def set_MeanCurv(self):
    self.MeanCurv = 0.0
  def set_dtpsi1Comov(self):
    self.CheckSetAttr("conv1rateComoving", "diff1rateMasked", "MaskedEvap")
    self.dtpsi1Comov = self.conv1rateComoving + self.diff1rateMasked + self.MaskedEvap
  def set_dtpsi2Comov(self):
    self.CheckSetAttr("conv2rateComoving", "diff2rateMasked", "dtphiComov")
    self.dtpsi2Comov = self.conv2rateComoving + self.diff2rateMasked + self.alpha*self.h*self.dtphiComov 
  def set_dtpsi1(self):
    self.CheckSetAttr("conv1rate", "diff1rateMasked", "MaskedEvap")
    self.dtpsi1 = self.conv1rate + self.diff1rateMasked + self.MaskedEvap
  def set_dtpsi2(self):
    self.CheckSetAttr("conv2rate", "diff2rateMasked", "dtphi")
    self.dtpsi2 = self.conv2rate + self.diff2rateMasked + self.alpha*self.h*self.dtphi 
  # set time derivative of phi but without advection
  def set_dtphiComov(self):
    self.CheckSetAttr("dyyphi", "dfXMdphi", "MeanCurv")
    self.dtphiComov = self.sigma*(self.dyyphi/(self.LAMB*self.LAMB) - self.dfXMdphi - self.MeanCurv/(self.LAMB*self.LAMB))
  # set time derivative (or rhs) of phi
  def set_dtphi(self):
    self.CheckSetAttr("dtphiComov")
    self.dtphi = self.dtphiComov - self.v*self.advphi
  def set_dtzeta(self):
    self.CheckSetAttr("dtphiComov")
    self.dtzeta = -self.alpha*self.h*self.dtphiComov - self.v*self.advzeta
  def set_tanhIC_pert(self, Ampl = 0, n = 1, hp = 1, ys = 0):
    x, y = self.x2D, self.y2D
    ls = (self.h0-hp)/self.beta
    k = 2*np.pi/(self.Lx/n)
    Perturbation = Ampl*np.cos(k*x)
    h = self.h0 - (self.h0 - hp)*np.tanh(((y*(1.0+Perturbation)+self.Ly/self.Ny) - ys)/ls)
    # solvent
    c = self.c0*np.ones(self.Ny)
    c = np.tile(c, (self.Nx, 1))
    psi1 = (1.0-c)*h
    # solute
    psi2 = c*h
    # phase-field
    # initialize only solution
    phi = 0.99*np.ones((self.Nx, self.Ny))
    # deposit
    zeta = np.zeros((self.Nx, self.Ny))
    self.fields = np.concatenate((psi1,psi2,phi,zeta), axis = 0)
    self.fields = self.fields.reshape((-1, self.Nx, self.Ny))
  def set_tanhIC(self, hp = 1, ys = 0):
    y = self.y2D
    ls = (self.h0-hp)/self.beta
    h = self.h0 - (self.h0 - hp)*np.tanh(((y+self.Ly/self.Ny) - ys)/ls)
    # solvent
    c = self.c0*np.ones(self.Ny)
    c = np.tile(c, (self.Nx, 1))
    psi1 = (1.0-c)*h
    # solute
    psi2 = c*h
    # phase-field
    # initialize only solution
    phi = 0.99*np.ones((self.Nx, self.Ny))
    # deposit
    zeta = np.zeros((self.Nx, self.Ny))
    self.fields = np.concatenate((psi1,psi2,phi,zeta), axis = 0)
  
# complex attributes
  # Find largest peaks of zeta on the right
  # wrapper for FindHighestPeaksRight1D
  def FindHighestZetaPeaksRight1D(self, *args, **kwargs):
    # make sure data is 1D
    self.set_zeta1D()
    # allocate data to Field object
    self.set_FieldProps(self.zeta1D, 'zeta1D')
    # find highest peaks
    PeakIndices, properties = self.FindHighestPeaksRight1D(self.zeta1D, *args, **kwargs)
    # save results
    self.zeta1DProps.PeakIndices = PeakIndices
    self.zeta1DProps.properties = properties
    # shift index to match shape of full domain
    # self.ShiftIndices(self.zeta1DProps, 'PeakIndices')

  def FindSmallestMinimaZetaPeaksRight1D(self, *args, **kwargs):
    # make sure data is 1D
    self.set_zeta1D()
    # allocate data to Field object
    self.set_FieldProps(self.zeta1D, 'zeta1D')
    # find minima
    MinimaIndices, properties = self.FindSmallestMinimaRight1D(self.zeta1D, *args, **kwargs)
    # save results
    self.zeta1DProps.MinimaIndices = MinimaIndices
    self.zeta1DProps.properties = properties
    # shift index to match shape of full domain
    # self.ShiftIndices(self.zeta1DProps, 'MinimaIndices')

  # PeakIndices and MinimaIndices have different names, therefore need to use get/setattr
  def ShiftIndices(self, FieldProps, IndicesStr):
    Indices = getattr(FieldProps, IndicesStr) + self.Ny - len(self.Maskedy)
    setattr(FieldProps, IndicesStr, Indices)


  # find the peak that is closest to the measurepoint
  # depending on how the peak is measured, it may be the position of the maximum or the minimum
  def PositionOfPeakClosestToMP(self, MeasurePoint):
    positions = self.zeta1DProps.properties['positions']
    distances = np.abs(positions - MeasurePoint)
    # Index relative to the peaks
    ClosestPeakIndex = np.argmin(distances)
    ClosestPeakPosition = positions[ClosestPeakIndex]
    return ClosestPeakPosition, ClosestPeakIndex

  # get the coordinates and values of the peak left of the minimum that just passed the measurepoint
  # MinimaIndices must be of at least length 2. Otherwise it is unknown where to stop the peak starts or ends
  def SetPeakLeftOfMinimum(self, FieldProperties, Index):
    # if(len(FieldProperties.MinimaIndices)==0):
    #   raise NoExtremaError('This should not happen as there should not exist an index. ')
    # elif(len(FieldProperties.MinimaIndices)==1):
    #   print(self.imagenumber, self.t)
    #   raise OnlyOneMinimumError(
    #     'Check if solution has advected by one domain, is periodic and has not \
    #      too large period. Else, try increasing domain size')
    # apply to MinimaIndices
    PeakMinIdx = FieldProperties.MinimaIndices[Index - 1]
    PeakMaxIdx = FieldProperties.MinimaIndices[Index]
    # element number PeakMaxIdx itself is left out
    PeakIndices = range(PeakMinIdx, PeakMaxIdx)
    # apply to actual data
    yPeak = self.y[PeakIndices]
    Peak = getattr(self, FieldProperties.FieldName)[PeakIndices]
    FieldProperties.PeakIndices = PeakIndices
    FieldProperties.yPeak = yPeak
    FieldProperties.Peak = Peak
    # use this to verify that positions and yPeak match
    # you can also see how element number PeakMaxIdx itself is left out
    # print(FieldProperties.properties['positions'], yPeak, Peak)
    # return yPeak, Peak, PeakIndices

  # need to provide data from FindHighestPeaksMasked1D
  # take data of rightmost peak as Measures
  def SetPeriodicSolutionMeasures(self):
    self.Measures.Max = self.zeta1DProps.GetMax()
    self.Measures.MaxPos = self.zeta1DProps.GetMaxPos()
    self.Measures.MinHeight = self.zeta1DProps.GetMin()
    self.Measures.MinPos = self.zeta1DProps.GetMinPos()
    self.Measures.Prominence = self.Measures.Max - self.Measures.MinHeight

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
  def get_PrecipitationOnset(self):
    # mask = self.phi1D<0
    mask = self.zeta1D>0
    if (not any(mask)):
      raise NoDepositError
    else:
      return np.argmax(mask)
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
  def __init__(self, path, start = None, end = None, file = 'frame_0000.dat', 
              objectclass = precipiti, attribute = attribute):
              # objectclass = precipiti, attribute = attribute, nCPU = 1):
    super().__init__(path, start = start, end = end, file = file, 
                    objectclass = objectclass, attribute = attribute)
                    # objectclass = objectclass, attribute = attribute, nCPU = nCPU)
    self.Measures = PrecipitiSimulMeasures()
    self.set_SpatialGrid1Dy()
    self.tmin = 0
    self.tminIndex = 0
    self.Stationary = False
    self.TooShort = False
    self.Periodic = False
    self.Ridge = False
    self.Deposit = False
    self.Transient = False

  # newer better method to calculate periods, amplitudes and other properties
  def set_PeriodicDeposit(self, Factor = 1.1, FractionOfMaximumProminence = 0, 
                  PeakSamples = 10, FindPeaksKwargs={'height':0, 'prominence':0}, 
                  TransientKwargs={'Threshold':1e-3, 'Windowlength':20}):
    self.set_PrecipitationOnset()
    self.set_MPIdx()
    self.set_tminIndex(Factor)
    self.ZetaOfT = np.array([sol.zeta1D[self.MPIdx] for sol in self.sols[self.tminIndex:]])
    try:
      PeakIndices, properties = cuda.FindHighestPeaks1D(
                                self.ZetaOfT, FractionOfMaximumProminence, 
                                PeakSamples = PeakSamples, **FindPeaksKwargs)
    except ValueError:
      raise SimulatedTooShortError('no Peaks found in zeta. Solution could be stationary \
        and not calculated long enough...\n {:}'.format(self.path))
    t = self.t[self.tminIndex:]
    self.Measures.SaveMeasures(PeakIndices, properties, t, self.ZetaOfT)
    try:
      self.Measures.FindEndOfTransient('MaximumThickness', **TransientKwargs)
      # simulation has passed transient stage
      self.Transient = False
      self.Periodic = True
      self.Measures.CutOffTransient()
    except TransientError as e:
      # simulation is still in transient stage
      self.Transient = True
      self.Periodic = False
      print(e)
    self.Measures.set_ZeroDimMeasures()
    return PeakIndices, properties
  # periodic2
  def set_PeriodicSolute(self, Factor = 1.0, RelativeMP = 0.9, FractionOfMaximumProminence = 0, 
                  PeakSamples = 10, FindPeaksKwargs={'height':0, 'prominence':0}, 
                  TransientKwargs={'Threshold':1e-3, 'Windowlength':20}):
    self.ApplyToAll('set_psi2_1D')
    MPIdx = int(RelativeMP*self.params['Ny'])
    self.set_tminIndex(Factor)
    self.Psi2OfT = np.array([sol.psi2_1D[MPIdx] for sol in self.sols[self.tminIndex:]])
    try:
      PeakIndices, properties = cuda.FindHighestPeaks1D(
                                self.Psi2OfT, FractionOfMaximumProminence, 
                                PeakSamples = PeakSamples, **FindPeaksKwargs)
    except ValueError:
      raise SimulatedTooShortError('no Peaks found in zeta. Solution could be stationary \
        and not calculated long enough...\n {:}'.format(self.path))
    self.Psi2Measures = PrecipitiSimulMeasures()
    t = self.t[self.tminIndex:]
    self.Psi2Measures.SaveMeasures(PeakIndices, properties, t, self.Psi2OfT)
    try:
      self.Psi2Measures.FindEndOfTransient('MaximumThickness', **TransientKwargs)
      # simulation has passed transient stage
      self.TransientPsi2 = False
      self.Periodic = True
      self.Psi2Measures.CutOffTransient()
    except TransientError as e:
      # simulation is still in transient stage
      self.TransientPsi2 = True
      self.Periodic = False
      print(e)
      self.Psi2Measures.set_ZeroDimMeasures()
    return PeakIndices, properties
  def set_TransientByPsi2(self, Samples = 200, **kwargs):
    self.TransientPsi2 = False
    PeakIndicesRight = []
    # loop through solutions
    for sol in self.sols[-Samples:]:
      sol.set_psi2_1D()
      PeakIndices, properties = sol.FindPeaks1D(sol.psi2_1D, **kwargs)
      try:
        PeakIndicesRight.append(PeakIndices[-1])
      except IndexError:
        raise SimulatedTooShortError('no Peaks found in psi2. Solution could be stationary \
        and not calculated long enough...or it is a periodic foot solution')
    PeakIndicesRight = np.array(PeakIndicesRight)
    # check if the right most peak is further on the right or equal than previously for each step
    # if yes: solution is still in transient
    # if not: probably periodic
    mask = PeakIndicesRight[1:]>=PeakIndicesRight[:-1]
    if(np.all(mask)):
      self.TransientPsi2 = True
    else:
      self.Periodic = True
    # return PeakIndicesRight



  # Calculate the simulation measures, mainly for periodic solutions
  # havent tested yet if it successfully ignores non-periodic solutions
  # RelativeMeasurePt: at which relative point(left of it) in the domain the peaks are measured
  # FractionOfMaximumProminence: only smallest minima are considered. However
  # since the minima tend to be larger on the right, there needs to be a bit of tolerance for
  # the minima. 0.9 means that all minima are considered that are at least 0.9 as prominent/deep
  # as the smallest minimum
  # Factor: Since the system is advected, we can approximately calculate how long it takes
  # for drawn out material to reach the end of the domain
  # in general, periodic solutions have relaxed after one domain has passed, therefore, Factor
  # should be >1
  def set_Periodic_old(self, RelativeMeasurePt = 0.9, FractionOfMaximumProminence = 0.9, 
                  height = (None, 0), Factor = 1.2, Threshold = 1e-3, Windowlength = 20):
    # Spatial coordinate a peak has to pass to be measured
    MeasurePt = RelativeMeasurePt*self.params["Ly"]
    # initialize ClosestPeakPositionOld
    ClosestPeakPositionOld = 0
    self.set_tminIndex(Factor)
    # main algorithm. loop through each timestep to find the peaks and measure their properties
    for sol in self.sols[self.tminIndex:]:
      # print(sol.imagenumber)
      # look for peaks (by looking for minima)
      try:
        sol.FindSmallestMinimaZetaPeaksRight1D(FractionOfMaximumProminence, height = height)
      except NoExtremaError:
        continue
      # continue if minimaindices has length less than 2, since then the boundaries of the peak are unknown
      if(len(sol.zeta1DProps.MinimaIndices)<2):
        continue
      # take the peak closest to the measurept
      ClosestPeakPosition, ClosestPeakIndex = sol.PositionOfPeakClosestToMP(MeasurePt)
      # if the minimum closest to the measurepoint is the first minimum, there is no other minimum on the left
      # this messes up the peak indices if not handled (min=index-1, max=index. Index=0 is problematic)
      if(ClosestPeakIndex == 0):
        continue
      # if the peak is on the right. Check if it previously was on the left. Only 
      # save Measures if a peak has just passed Measurept
      if(ClosestPeakPosition>MeasurePt) and (ClosestPeakPositionOld <= MeasurePt):
        # create attrbibute to save Measures in
        FieldProperties = sol.zeta1DProps
        # take data from closest peak
        sol.SetPeakLeftOfMinimum(FieldProperties, ClosestPeakIndex)
        sol.SetPeriodicSolutionMeasures()
        # since there wont be many periods, we simply append
        # append is faster for lists than for numpy arrays
        self.Measures.SaveMeasuresToLists(sol)
      ClosestPeakPositionOld = ClosestPeakPosition
    # calculate Simulation measures after individual peaks have been analyzed
    self.Measures.set_PeriodicMeasures()
    try:
      self.Measures.FindEndOfTransient('MaximumThickness', Threshold = Threshold, Windowlength = Windowlength)
      # simulation is not in transient stage
      self.Transient = False
      self.Measures.CutOffTransient()
      self.Periodic = True
    except TransientError:
      # simulation is still in transient stage
      self.Transient = True
      self.Periodic = False

  def set_MPIdx(self, ybuffer = 100):
    Ly = self.params['Ly']
    mp1 = self.PrecipitationOnset + ybuffer
    mp2 = self.PrecipitationOnset + (Ly-self.PrecipitationOnset)/2
    self.MPIdx = cuda.find_nearest(self.y, np.min([mp1, mp2]))
    # print(self.MeasurePoint)
    # if(self.MeasurePoint>=0.95*Ly):
    #   raise OnsetTooLateError

  def set_PrecipitationOnset(self, n = 200):
    if(not hasattr(self, 'Deposit')):
      self.set_Deposit(n)
      if(not self.Deposit):
        raise NoDepositError
    IndicesOnset = []
    for sol in self.sols[-n:]:
      try:
        Index = sol.get_PrecipitationOnset()
      except NoDepositError:
        continue
      IndicesOnset.append(Index)
    IndexOnset = np.max(IndicesOnset)
    self.PrecipitationOnset = sol.y[IndexOnset]
  def set_h_explodes(self, factor = 1.0, n = 200):
    for sol in self.sols[-n:]:
      self.hExplodes = False
      mask = sol.h > sol.h0
      if(mask.any()):
        self.hExplodes = True
        break

      
  
  def set_OutputDT(self):
    tend = self.t[-1]
    tstart = self.t[0]
    Deltat = tend - tstart
    self.OutputDT = Deltat/(len(self.t) - 1)
  def set_tminIndex(self, Factor):
    if(not hasattr(self, 'OutputDT')):
      self.set_OutputDT()
    if(self.tmin == 0):
      self.set_MinimumDuration(Factor)
    self.tminIndex = int(self.tmin/self.OutputDT)

  # Check if the simulation has even run long enough
  def set_MinimumDurationPassed(self, Factor = 1.1):
    self.set_MinimumDuration(Factor)
    if(self.t[-1]>self.tmin):
      self.TooShort = False
    else:
      self.TooShort = True
      print(self.path)
      raise SimulatedTooShortError('Minimum Duration not passed: ' + str(self.path))
  def set_MinimumDuration(self, Factor = 1.1):
    # self.tmin = self.params['Ly']/self.params['v']*Factor
    self.tmin = self.params['Ly']/(self.params['v'])**Factor

  def CharacterizeSolution(self, Factor = 1.1, nSamples = 200, eps = 1e-10, Ridgekwargs={}, 
                            Periodickwargs={}, Psi2kwargs={}):
    try: 
      # check if enough advected
      self.set_MinimumDurationPassed(Factor)
      # check if enough frames
      self.CheckSampleSize(nSamples)
      # check if stationary
      self.set_Stationary(nSamples, eps)
      # check if h explodes
      self.set_h_explodes(nSamples)
      if(self.hExplodes):
        return
      # check if h has ridge
      self.set_Ridge(nSamples, **Ridgekwargs)
      # check if zeta>0
      self.set_Deposit(nSamples)
      # check periodicity of zeta or psi2(if no deposit)
      if(not self.Stationary):
        if(self.Deposit):
          self.set_PeriodicDeposit(**Periodickwargs)
          if(self.Transient):
            self.TooShort = True
        else:
          self.set_PeriodicSolute(**Periodickwargs)
          if(self.TransientPsi2):
            self.TooShort = True
    except SimulatedTooShortError as e:
      print(e)
      self.TooShort = True
      return



  # Check if the last n solutions have at least one ridge, aka at least one local maximum
  def set_Ridge(self, nSamples = 200, **FindPeaksKwargs):
    self.CheckSampleSize(nSamples)
    self.Ridge = False
    # loop through solutions
    for sol in self.sols[-nSamples:]:
      data1D = sol.get_crosssection_y(sol.h)
      # allocate data to Field object
      if(not hasattr(self, 'h1DProps')):
        sol.set_FieldProps(data1D, 'h1D')
      PeakIndices, properties = sol.FindPeaks1D(data1D, **FindPeaksKwargs)
      sol.h1DProps.MaximaIndices = PeakIndices
      sol.h1DProps.properties = properties
      if(len(PeakIndices)>0):
        self.Ridge = True
        break

  def set_Deposit(self, nSamples = 200):
    self.CheckSampleSize(nSamples)
    for sol in self.sols[-nSamples:]:
      SolidPhase = sol.zeta1D > 0
  #     phi1D = sol.get_crosssection_y(sol.phi)
  #     SolidPhase = phi1D < 0
      if(any(SolidPhase)):
        self.Deposit = True
        break
      else:
        self.Deposit = False



  # expected equilibrium precursor height, depending on Ups, Mu, Chi and initial concentration c
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
  def __init__(self, path = None, silent = False):
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
      if(not silent):
        print('no path provided, creating default solution object')

  # def readparams(self, filepath=None):
  #   if filepath is None:
  #     filepath = self.path
  #   filepath = str(cuda.dat(filepath))
  #   # print(filepath)
  #   with open(filepath,'r') as f:
  #     lines = f.readlines()		#list, not array
  #   for line in lines:
  #     if line.split()[0] == 'Nx':
  #       self.Nx = int(line.split()[1])
  #     elif line.split()[0] == 'Ny':
  #       self.Ny = int(line.split()[1])
  #     elif line.split()[0] == 'Lx':
  #       self.Lx = float(line.split()[1])
  #     elif line.split()[0] == 'Ly':
  #       self.Ly = float(line.split()[1])
  #     elif line.split()[0] == 't':
  #       self.t = float(line.split()[1])
  #     elif line.split()[0] == 'dt':
  #       self.dt = float(line.split()[1])
  #     elif line.split()[0] == 'imagenumber':
  #       self.imagenumber = int(line.split()[1])
  #     elif line.split()[0] == 'lamb':
  #       self.lamb = float(line.split()[1])
  #     elif line.split()[0] == 'PeXM':
  #       self.PeXM = float(line.split()[1])
  #     elif line.split()[0] == 'alpha':
  #       self.alpha = float(line.split()[1])
  #     elif line.split()[0] == 'c0':
  #       self.C0 = float(line.split()[1])
  #     elif line.split()[0] == 'Ceq':
  #       self.Ceq = float(line.split()[1])

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
    super().__init__(path, start = None, end = None, objectclass = XuMeakin)

