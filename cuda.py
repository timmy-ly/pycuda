import numpy as np
import warnings
from pathlib import Path, PurePath

import os
# scipy in interpolation

class Error(Exception):
  """Base class for exceptions in this module"""
  pass
class ArgumentsError(Error):
  """Exception raised for errors in the input.
  Attributes:
      message -- explanation of the error"""
  def __init__(self, message):
    self.message = message
class GridError(Error):
  """Exception raised for mismatched grid parameters."""
  def __init__(self, message):
    self.message = message

# class GridSpacing:
#   def __init__(self, dx):
#     self.dx = dx
#     self.dx2 = dx*dx
#     self.dx3 = dx**3
#     self.dx4 = dx**4

def create_object_and_handle(SomeClass, *args, handle=lambda e : None, **kwargs):  
  try:  
    return SomeClass(*args, **kwargs)  
  except FileNotFoundError as err: 
    print("err: ", err)
    return handle(err)
def handle(err):
  return None


# methods that apply to all cuda problems
# add .dat to filepath
def dat(filepath):
  filepath = Path(PurePath(filepath))
  return filepath.with_suffix('.dat')

# add .bin to filepath
def bin(filepath):
  filepath = Path(PurePath(filepath))
  return filepath.with_suffix('.bin')

def convert(val):
  if(val == 'True'):
    return True
  elif(val == 'False'):
    return False
  constructors = [int, float, str]
  for c in constructors:
    try:
      return c(val)
    except ValueError:
      pass


# turn line in file into dictionary
def LineToDict(keys, line):
  vals = [convert(val) for val in line.split()]
  return dict(zip(keys, vals))

# todo: maybe sort the dictionary?
# method to manually trigger update

# class to collect phase data from multiple simulations
# the determination of the phase is done elsewhere
# PhaseDataName: storage file for the phase data
# FilePattern: pattern of the parameter files of the solution frames
# ParameterStrs: strings of the parameters that span the parameter space
# AdditionalColumns: Other relevant quantities per parameter set (such as 
# phasetype)
class PhaseData:
  def __init__(self, ParameterStrs, AdditionalColumns, SaveName = 'PhaseDiagram'):
    self.PhaseDataName = Path(PurePath(SaveName + '.dat'))
    self.FilePattern = 'frame_[0-9]*dat'
    self.ParameterStrs = ParameterStrs
    self.AdditionalColumns = AdditionalColumns
    self.ReadCreatePhaseDiagramDict()
    
  def ReadCreatePhaseDiagramDict(self):
      # DataPoints is a dictionary of dictionaries
      # the outerkeys are parametervalues corresponding to ParameterStrs
      # the inner dictionaries contain ParameterStr:ParameterValue entries as well
      # as more entries characterizing each Datapoint
      self.DataPoints = {}
      if(self.PhaseDataName.exists()):
        with open(self.PhaseDataName, 'r') as f:
          firstline = f.readline()
          # parser is on next line
          # print(firstline)
          innerkeys = firstline.split()
          for line in f:
            # next 2 lines are for trailing lines
            line = line.rstrip()
            if(line):
              DataPoint = LineToDict(innerkeys, line)
              outerkeys = self.GetOuterkeys(DataPoint)
              self.DataPoints[outerkeys] = DataPoint

  def WriteToFile(self):
    with open(self.PhaseDataName, 'w') as f:
      f.write(" ".join(self.ParameterStrs + self.AdditionalColumns) + "\n")
      keys = self.Sort()
      for key in keys:
        DataPoint = self.DataPoints[key]
      # for DataPoint in self.DataPoints.values():
        InnerValues = [str(InnerValue) for InnerValue in DataPoint.values()]
        f.write(" ".join(InnerValues) + "\n")
  # cannot sort dictionary, must copy keys and then sort
  def Sort(self):
    # I think this is the only part that I have not made independent of dimension yet
    if(len(self.ParameterStrs)==2):
      return sorted(self.DataPoints.keys(), key=lambda x: (x[0], x[1]))
    elif(len(self.ParameterStrs)==1):
      return sorted(self.DataPoints.keys(), key=lambda x: (x[0]))

  # check if DataPaths contains simulations that are not present in PhaseDataName.dat
  # also check if existing entries have less frames
  def FilterDataToUpdate(self, DataPaths):
    self.DataPathsToUpdate = []
    # loop through the provided simulation paths
    for DataPath in DataPaths:
      # get parameter sets from one of the frames
      try:
        Parameters = self.GetParametersFromSol(DataPath)
      except FileNotFoundError:
        print('Probably not even started yet: ')
        print(DataPath)
        continue
      # get the relevant parameter values as outerkeys
      # cannot use list, as list is not hashable
      outerkey = tuple(Parameters[ParameterStr] for ParameterStr in self.ParameterStrs)
      if(outerkey not in self.DataPoints):
        self.DataPathsToUpdate.append(DataPath)
      elif(self.DataPoints[outerkey]['n']<Parameters['n']):
        self.DataPathsToUpdate.append(DataPath)
  def DeleteOriginalFile(self):
    if os.path.exists(self.PhaseDataName):
      os.remove(self.PhaseDataName)
  # 
  def GetOuterkeys(self, DataPoint):
    return tuple(DataPoint[ParameterStr] for ParameterStr in self.ParameterStrs)

  # update DataPoints
  def AddUpdateDataToDict(self, DataPoint):
    outerkeys = self.GetOuterkeys(DataPoint)
    self.DataPoints[outerkeys] = DataPoint

  # get parameters from cuda parameter file as a dictionary with an additional entry
  # that represents the number of .dat files according to self.FilePattern
  def GetParametersFromSol(self, Path, FileName = 'frame_0001.dat'):
    Parameters = {}
    with open(Path / FileName) as f:
      for line in f:
        entries = line.split()
        if len(entries)>1:
          Parameters[entries[0]] = convert(entries[1])
    n = len(list(Path.glob(self.FilePattern)))
    Parameters['n'] = n
    return Parameters

def get_last_file(Top, pattern = "frame_[0-9]*dat", SortAttribute = "imagenumber"):
  from pathlib import Path, PurePath
  Top = Path(PurePath(Top))
  Filepaths = list(Top.glob(pattern))
  Filepaths = sorted(Filepaths, key = lambda path:GetOnePropertyFromParamFile(path, SortAttribute))
  return Filepaths[-1]

def GetOnePropertyFromParamFile(path, property):
  with open(path, 'r') as f:
    for line in f:
      entries = line.split()
      if(entries[0]==property):
        return convert(entries[1])
def GetTwoPropertiesFromParamFile(path, property1, property2):
  prop1_read = False
  prop2_read = False
  with open(path, 'r') as f:
    for line in f:
      entries = line.split()
      if(entries[0]==property1):
        val1 = convert(entries[1])
        prop1_read = True
      elif(entries[0]==property2):
        val2 = convert(entries[1])
        prop2_read = True
      if(prop1_read and prop2_read):
        return (val1, val2)

def FindHighestPeaks1D(data1D, FractionOfMaximumProminence = 0, PeakSamples = 10, 
                      **findpeaksKwargs):
    from scipy.signal import find_peaks
    PeakIndices, properties = find_peaks(data1D, **findpeaksKwargs)
    # get the reference prominence as stated above
    n = len(properties['prominences'])
    if(PeakSamples>n):
      PeakSamples = n
    MaxProminence = np.max(properties['prominences'][-PeakSamples:])
    # exclude the peaks that have too small prominence
    ProminenceMask = MaskFromValue(properties, 'prominences', MaxProminence, 
                                      FractionOfMaximumProminence)
    ExcludePeaksFromProperties(properties, PeakIndices, ProminenceMask)
    # Update PeakIndices
    PeakIndices = PeakIndices[ProminenceMask]
    return PeakIndices, properties
# create mask that only takes the values of key that are above maximum*fraction
def MaskFromValue(properties, key, value, fraction):
  return properties[key]>(value*fraction)

# take each property and mask each array
# for my purpose it would have been better if scipy.find_peaks returned an array
# of peak objects so that each property value is tied to each peak
# the values are only indirectly tied to each peak by their indices in the property
# arrays right now. It is probabaly done like this with lots of peaks in mind
def ExcludePeaksFromProperties(properties, ReferenceArray, mask):
  # only modify those values that have same shape as ReferenceArray
  n = len(ReferenceArray)
  for key, value in properties.items():
    if(len(value) == n):
      # call dictionary directly in order to change its values
      properties[key] = value[mask]
    else:
      continue


def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

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

def get_field_dims(field):
  Nx, Ny = 1, 1
  if(len(np.shape(field))==2):
    Nx, Ny = np.shape(field)
  elif(len(np.shape(field))==1):
    Ny = len(field)
  return Nx, Ny

#
def cut_field_y(field, cut = 0.25):
  Nx, Ny = get_field_dims(field)
  Ny_up = int(Ny*(1-cut))
  Ny_down = int(Ny*cut)
  return field[:,Ny_down:]
def shift_field(field):
  return field - np.mean(field)

# fourier stuff
# create 2d arrays for wavenumbers, use fftshift to have intuitive order (aka from negative to positive values)
def set_k(field, dx, dy):
  Nx, Ny = get_field_dims(field)
  ky, kx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(Ny,dy/2.0/np.pi)), np.fft.fftshift(np.fft.fftfreq(Nx,dx/2.0/np.pi)))
  return kx, ky
def fft_field(field):
  # find the complex fft -- shift to put 0 wavenumber at the center
  return np.fft.fftshift(np.fft.fft2(field))
def normalize_fft(field):
  FFTMag = np.abs(field)
  return FFTMag/np.amax(FFTMag)
def argmax2d(field, n = 2):
  # partition the field, meaning shift the n largest values to the end of the flattened array and all the others to the left. the partitions themselves are not sorted! then return the new indices, same shape as field, here: flattened
  # None is important, so that it deals with the flattened array
  # you could also flatten the array by hand which would be more robust
  index_flattened = np.argpartition(field, -n, axis=None)[-n:]
  # sort indices
  index_flattened = index_flattened[np.argsort(field.flatten()[index_flattened], axis = None)]
  # undo the flattening so to say
  # returns two arrays: xindices, yindices
  idx_x, idx_y = np.unravel_index(index_flattened, np.array(field).shape)
  index_pairs = np.array([idx_x, idx_y]).T
  return index_pairs

def get_fft_local_max_mask_2d(field, Threshold, **kwargs):
  # one could probably use np.where instead of masks
  import scipy.ndimage as ndimage
  import scipy.ndimage.filters as filters
  # find the local 2d maxima of the fourier spectrum, find the fundamental k in particular
  # legacy, needed for mode = "constant" which we usually do not use
  const_value = 0
  # find all local maxima?
  field_max = filters.maximum_filter(field, cval = const_value, **kwargs)
  mask_max = (field == field_max)
  # find all local minima?
  field_min = filters.minimum_filter(field, cval = const_value, **kwargs)
  # only take those extrema that are relatively large, meaning larger than threshold
  diff = ((field_max - field_min) > Threshold)
  # print(np.where(diff))
  # print(np.all(diff==0))
  # print(~diff)
  mask_max[~diff] = False
  # print(np.any(mask_max))
  return mask_max
# only consider frequencies positive in y
def filter_pos_yfreq(field, mask_max, kx, ky):
  FilteredMask = filter_yfreq_from_mask(field, mask_max)
  filtered_kx, filtered_ky = filter_yfreq_from_k(kx, ky)
  # FilteredField = filter_yfreq_from_field(field)
  return filtered_kx, filtered_ky, FilteredMask
def filter_yfreq_from_field(field):
  n, m = np.shape(field)
  return field[:,m//2:]
def filter_yfreq_from_mask(field, mask_max):
  n, m = np.shape(field)
  return mask_max[:,m//2:]
def filter_yfreq_from_k(kx,ky):
  n, m = np.shape(ky)
  ky_half = ky[:,m//2:]
  kx_half = kx[:,m//2:]
  return kx_half, ky_half
# select n smallest frequencies
def filter_smallest_k(mask_max, kx, ky,n = 2):
  k = get_k(kx,ky)
  # k at local power maxima
  kxMaxPower = kx[mask_max]
  kyMaxPower = ky[mask_max]
  kMaxPower = k[mask_max]
  # if the k with maximum power have vanishing y-components: vertical stripes or homogeneous deposit: only select positive kx
  PositivekxMask = np.ones_like(kMaxPower, dtype = bool)
  if(vanishing_elements(kyMaxPower)):
    PositivekxMask = (kxMaxPower>0)
    # only filter the positive kx if the kx dont vanish as well (kx and ky vanish means no pattern, homogeneous)
    if(not vanishing_elements(kxMaxPower)):
      kMaxPower = kMaxPower[PositivekxMask]
  # smallest k of these maxima
  n = np.min([len(kMaxPower),n])
  kmin = np.sort(kMaxPower)[:n]
  # if(len(kMaxPower)>1):
  #   kmin = np.partition(kMaxPower, 1)[:n]
  # else:
  #   kmin = kMaxPower
  # create new mask_max by comparing these kmin with k
  maskkMin = np.isin(k, kmin)
  # this is true for all k that have the same magnitude as these kmin which means that there may be "new" local maxima
  # which is why one needs to combine this mask with the old one
  # print(np.any((maskkMin & mask_max)))
  return (maskkMin & mask_max)

def vanishing_elements(array, eps = 1e-5):
  return np.all(np.abs(array)<eps)

def approx_equal_elements_indices(array, ReferenceValue, eps=1e-4):
  # nonzero returns tuple of shape (1,array.shape)
  return np.nonzero(np.abs(array - ReferenceValue)<eps)[0]

def get_k(kx,ky):
  return np.sqrt(kx*kx + ky*ky)

def get_ky_of_smallest_k(kx,ky):
  k = get_k(kx,ky)
  # get kx and ky of smallest k
  # get index of smallest k
  i = np.argmin(k)
  ix,iy = flat_index_to_tuple(i,np.shape(k))
  return ky[ix,iy]

def flat_index_to_tuple(index, shape):
  return np.unravel_index(index, shape)


# calculate difference between two solutions
def DifferenceSols(sol1, sol2):
  if(sol1.Ny != sol1.Ny):
    raise GridError('y-dimensions do not match')
  # make use of numpy's broadcasting rules. All dimensions match except for possibly the x-dimension which is then 1 (if you compare 2d with 1d)
  # Then the axis of length 1 will be broadcast to match the other solution's axis' length
  Difference = sol1.fields - sol2.fields
  return Difference
# and take norm
# only works on the standard sol.fields
# use different method for other fields
def DifferenceNorm(sol1, sol2):
  Difference = DifferenceSols(sol1, sol2)
  # calculate norm for each field (sum over second and third axis)
  # Norm = np.sum(np.abs(Difference), (1,2))
  # take maximum error per field
  Norm = np.amax(np.abs(Difference), (1,2))
  return Norm

# Simul1 and Simul2 must have same indices and timesteps
# cant always use condition that the solution time t is equal since the timestep MIGHT be different 
# depending on the purpose of the comparison
def DifferenceNormEvolution(Simul1, Simul2):
  # get number of fields
  nof = len(Simul1.sols[0].fields)
  # take into account that the simulations may be of different lengths
  # only compare up until end of shorter one
  if(len(Simul1.sols)>len(Simul2.sols)):
    n = len(Simul2.sols)
  else:
    n = len(Simul1.sols)
  # initialize the error array, use ones to emphasize the dummy nature
  # of this initialization
  DifferenceNormArray = np.ones((nof, n))
  # calculate the error for each timestep
  for i in range(n):
    DifferenceNormArray[:,i] = DifferenceNorm(Simul1.sols[i],Simul2.sols[i])
  # print(np.shape(Simul1.t[range(n)]), np.shape(DifferenceNormArray))
  # return the time steps and errors
  return Simul1.t[range(n)], DifferenceNormArray


def l2norm(array):
  shape = np.shape(array)
  nx = shape[0]
  if(len(shape)== 2):
    ny = np.shape(array)[1]
  else:
    ny = 1
  return np.sqrt(np.sum(array*array)/(nx*ny))

def read_l2(filename, **kwargs):
  array = np.loadtxt(filename, dtype = float, skiprows = 1, unpack = True, **kwargs)
  return array
def mean_l2(t,l2, ReltStart = 0.5):
  mask = t>(t[-1]*ReltStart)
  t = t[mask]
  l2 = np.sqrt(l2[mask])
  dt = t - np.roll(t,1)
  T = t[-1]-t[0]
  mean = np.sum(l2[1:]*dt[1:])/T
  return mean

# interpolate array with Nx, Ny, Lx, Ly to newNx, Newy
def interpolate(array, Nx, Ny, Lx, Ly, newNx, newNy, xmin = 0, ymin = 0):
  from scipy.interpolate import RectBivariateSpline as rbs
  x = np.linspace(xmin,Lx,Nx)
  y = np.linspace(ymin,Ly,Ny)
  newx = np.linspace(xmin,Lx,newNx)
  newy = np.linspace(ymin,Ly,newNy)
  interpolatedobject = rbs(x,y,array)
  # evaluate interpolatedobject at newx, newy points
  return interpolatedobject.__call__(newx,newy)
def interp_1D(array, Nx, Lx, newNx):
  from scipy.interpolate import interp1d
  dx = Lx/Nx
  x = dx*np.arange(Nx)
  newdx = Lx/newNx
  newx = newdx*np.arange(newNx)
  interpolatedobject = interp1d(x,array)
  # evaluate interpolatedobject at newx, newy points
  return interpolatedobject.__call__(newx)

# finite difference stencils
# template for stencil components
def fTemplate(fields):
  def Rolled(ix,iy,fields=fields):
    return np.roll(fields,(-ix,-iy),(-2,-1))
  return Rolled
# def CheckSolobjFields(solobj, fields):
#   if (solobj is None) and (fields is None):
#     raise ArgumentsError("Provided neither solobj nor fields. No field(s) to calculate. ")
#   elif (solobj is not None) and (fields is None):
#     fields = solobj.fields
#   return fields
# ? order
# curvature
def curvature(fields, dx):
  f = fTemplate(fields)
  return ( (f(1,0) - f(0,0))/np.sqrt( (f(1,0) - f(0,0))**2 + ((f(1,1) + f(0,1) - f(1,-1) - f(0,-1))**2)/16.0 ) 
         - (f(0,0) - f(-1,0))/np.sqrt( (f(0,0) - f(-1,0))**2 + ((f(-1,1) + f(0,1) - f(-1,-1) - f(0,-1))**2)/16.0 )
         + (f(0,1) - f(0,0))/np.sqrt( (f(0,1) - f(0,0))**2 + ((f(1,1) + f(1,0) - f(-1,1) - f(-1,0))**2)/16.0 )
         - (f(0,0) - f(0,-1))/np.sqrt( (f(0,0) - f(0,-1))**2 + ((f(1,-1) + f(1,0) - f(-1,-1) - f(-1,0))**2)/16.0 ) )/dx

# 2nd order
def dx2_m11(fields, dx):
  f = fTemplate(fields)
  return ( f(1,0) - f(-1,0) )/2.0/dx
# 4th order
# dy fdm 4th order, forward, copied from cuda code
def dy4_04(fields, dx):
  f = fTemplate(fields)
  return ( -25.0*f(0,0) + 48.0*f(0,1) - 36.0*f(0,2) + 16.0*f(0,3) - 3.0*f(0,4) )/12.0/dx
# dx fdm 4th order, center
def dx4_m22(fields, dx):
  f = fTemplate(fields)
  return (-f(2,0) + 8.0*f(1,0) - 8.0*f(-1,0) + f(-2,0) )/12.0/dx
# dy fdm 4th order, center, copied from cuda code
def dy4_m22(fields, dx):
  f = fTemplate(fields)
  return (-f(0,2) + 8.0*f(0,1) - 8.0*f(0,-1) + f(0,-2) )/12.0/dx
# dy fdm 4th order, 1 left node, copied from cuda code
def dy4_m13(fields, dx):
  f = fTemplate(fields)
  return (-3.0*f(0,-1) - 10.0*f(0,0) + 18.0*f(0,1) - 6.0*f(0,2) + f(0,3) )/12.0/dx
# dy fdm 4th order, 1 right node, copied from cuda code
def dy4_m31(fields, dx):
  f = fTemplate(fields)
  return ( -f(0,-3) + 6.0*f(0,-2) - 18.0*f(0,-1) + 10.0*f(0,0) + 3.0*f(0,1) )/12.0/dx
# dxx fdm 4th order, center
def dxx4_m22(fields, dx2):
  f = fTemplate(fields)
  return ( -f(-2,0) + 16.0*f(-1,0) - 30.0*f(0,0) + 16.0*f(1,0) -f(2,0) )/12.0/dx2
# dxy fdm 4th order, center, copied from cuda code
def dxy4_m22_m22(fields, dx2):
  f = fTemplate(fields)
  return ( f(-2,-2) - f(-2,2) - f(2,-2) + f(2,2)
         + 8.0*(-f(-1,-2) - f(-2,-1) + f(-2,1) + f(-1,2) + f(1,-2) + f(2,-1) - f(2,1) - f(1,2))
         + 64.0*(f(-1,-1) - f(-1,1) - f(1,-1) + f(1,1)) )/144.0/dx2
# dxy fdm 4th order, 1 left node, copied from cuda code
def dxy4_m22_m13(fields, dx2):
  f = fTemplate(fields)
  return ( 2.0*(-f(-2,-1) + f(2,-1)) + 3.0*(-f(-2,0) + f(2,0) - f(-1,3) + f(1,3)) + 6.0*(f(-2,1) - f(2,1))
          - f(-2,2) + f(2,2) + 13.0*(f(-1,-1) - f(1,-1)) + 36.0*(f(-1,0) - f(1,0)) 
          + 66.0*(-f(-1,1) + f(1,1)) + 20.0*(f(-1,2) - f(1,2)) )/72.0/dx2
# dxy fdm 4th order, 1 right node, copied from cuda code
def dxy4_m22_m31(fields, dx2):
  f = fTemplate(fields)
  return ( f(-2,-2) - f(2,-2) + 6.0*(-f(-2,-1) + f(2,-1)) + 3.0*(f(-2,0) + f(-1,-3) - f(1,-3) - f(2,0))
          + 2.0*(f(-2,1) - f(2,1)) + 20.0*(-f(-1,-2) + f(1,-2)) + 66.0*(f(-1,-1) - f(1,-1))
          + 36.0*(-f(-1,0) + f(1,0)) + 13.0*(-f(-1,1) + f(1,1)) )/72.0/dx2
# dyy fdm 4th order, 1 left node, copied from cuda code
def dyy4_m14(fields, dx2):
  f = fTemplate(fields)
  return ( 10.0*f(0,-1) - 15.0*f(0,0) - 4.0*f(0,1) + 14.0*f(0,2) - 6.0*f(0,3) + f(0,4) )/12.0/dx2
# dyy fdm 4th order, center
def dyy4_m22(fields, dx2):
  f = fTemplate(fields)
  return ( -f(0,-2) + 16.0*f(0,-1) - 30.0*f(0,0) + 16.0*f(0,1) -f(0,2) )/12.0/dx2
# dyy fdm 4th order, 1 right node, copied from cuda code
def dyy4_m41(fields, dx2):
  f = fTemplate(fields)
  return ( f(0,-4) - 6.0*f(0,-3) + 14.0*f(0,-2) - 4.0*f(0,-1) - 15.0*f(0,0) + 10.0*f(0,1) )/12.0/dx2
# laplace fdm 4th order, center, copied from cuda code
def laplace4_m22_m22(fields, dx2):
  f = fTemplate(fields)
  return (-60.0*f(0,0) + 16.0*(f(1,0) + f(0,1) + f(-1,0) + f(0,-1))
         - (f(2,0) + f(0,2) + f(-2,0) + f(0,-2)) )/12.0/dx2
# dyyy fdm 4th order, center, copied from cuda code
def dyyy4_m33(fields, dx3):
  f = fTemplate(fields)
  return (-f(0,3) + 8.0*f(0,2) - 13.0*f(0,1) + 13.0*f(0,-1) - 8.0*f(0,-2) + f(0,-3))/8.0/dx3
# dyyy fdm 4th order, node -2 to 4, copied from cuda code
def dyyy4_m24(fields, dx3):
  f = fTemplate(fields)
  return (f(0,4) - 8.0*f(0,3) + 29.0*f(0,2) - 48.0*f(0,1)
          +35.0*f(0,0) - 8.0*f(0,-1) - f(0,-2) )/8.0/dx3
# dyyy fdm 4th order, node -4 to 2, copied from cuda code
def dyyy4_m42(fields, dx3):
  f = fTemplate(fields)
  return (f(0,2) + 8.0*f(0,1) - 35.0*f(0,0) + 48.0*f(0,-1)
          -29.0*f(0,-2) + 8.0*f(0,-3) - f(0,-4) )/8.0/dx3
# dxyy fdm 4th order, center, copied from cuda code
def dxyy4_m22_m22(fields, dx3):
  f = fTemplate(fields)
  return ( -f(-2,-2) - f(-2,2) + f(2,-2) + f(2,2)
          + 16.0*(f(-2,-1) + f(-2,1) - f(2,-1) - f(2,1))
          + 8.0*(f(-1,-2) + f(-1,2) - f(1,-2) - f(1,2))
          + 30.0*(-f(-2,0) + f(2,0))
          + 128.0*(-f(-1,-1) - f(-1,1) + f(1,-1) + f(1,1))
          + 240.0*(f(-1,0) - f(1,0)) )/144.0/dx3
# dyxx fdm 4th order, center, copied from cuda code
def dyxx4_m22_m22(fields, dx3):
  f = fTemplate(fields)
  return ( -f(-2,-2) + f(-2,2) - f(2,-2) + f(2,2)
          + 16.0*(f(-1,-2) - f(-1,2) + f(1,-2) - f(1,2))
          + 8.0*(f(-2,-1) - f(-2,1) + f(2,-1) - f(2,1))
          + 30.0*(-f(0,-2) + f(0,2))
          + 128.0*(-f(-1,-1) + f(-1,1) - f(1,-1) + f(1,1))
          + 240*(f(0,-1) - f(0,1)) )/144.0/dx3
# dyyyy fdm 4th order, 2 left nodes
def dyyyy4_m25(fields, dx4):
  f = fTemplate(fields)
  return ( f(0,5) - 8.0*f(0,4) + 27.0*f(0,3) - 44.0*f(0,2) + 31.0*f(0,1) - 11.0*f(0,-1) + 4.0*f(0,-2) )/6.0/dx4
# dyyyy fdm 4th order, center
def dyyyy4_m33(fields, dx4):
  f = fTemplate(fields)
  return ( -f(0,3) + 12.0*f(0,2) - 39.0*f(0,1) + 56.0*f(0,0) - 39.0*f(0,-1) + 12.0*f(0,-2) - f(0,-3) )/6.0/dx4
# biharm fdm 4th order, center, copied from cuda code
def biharm4_m33_m33(fields, dx4):
  f = fTemplate(fields)
  return ( -(f(0,3) + f(0,-3) + f(3,0) + f(-3,0))
             + 14.0*(f(0,2) + f(0,-2) + f(2,0) + f(-2,0))
             -77.0*(f(0,1) + f(0,-1) + f(1,0) + f(-1,0))
             +184.0*f(0,0)
             + 20.0*(f(1,1) + f(1,-1) + f(-1,1) + f(-1,-1))
             -(f(1,2) + f(2,1) + f(1,-2) + f(2,-1) + f(-1,2) + f(-2,1)
             + f(-1,-2) + f(-2,-1)) )/6.0/dx4
# biharm fdm 4th order, ynode -2 to 5, copied from cuda code
def biharm4_m33_m25(fields, dx4):
  f = fTemplate(fields)
  return ( -f(-3,0) - f(-2,-1) - f(-2,1) - f(-1,-2) - f(-1,2) + f(0,5) - f(1,-2) - f(1,2) - f(2,-1) - f(2,1) - f(3,0)
          + 14.0*(f(-2,0) + f(2,0))
          + 20.0*(f(-1,-1) + f(-1,1) + f(1,-1) + f(1,1))
          + 77.0*(-f(-1,0) - f(1,0))
          + 6.0*f(0,-2) - 49.0*f(0,-1) + 128.0*f(0,0) - 7.0*f(0,1) - 42.0*f(0,2) + 27.0*f(0,3) - 8.0*f(0,4) )/6.0/dx4
# biharm fdm 4th order, ynode -5 to 2, copied from cuda code
def biharm4_m33_m52(fields, dx4):
  f = fTemplate(fields)
  return ( -f(-3,0) - f(-2,-1) - f(-2,1) - f(-1,-2) - f(-1,2) + f(0,-5) - f(1,-2) - f(1,2) - f(2,-1) - f(2,1) - f(3,0)
          + 14.0*(f(-2,0) + f(2,0))
          + 20.0*(f(-1,-1) + f(-1,1) + f(1,-1) + f(1,1))
          + 77.0*(-f(-1,0) - f(1,0))
          + 6.0*f(0,2) - 49.0*f(0,1) + 128.0*f(0,0) - 7.0*f(0,-1) - 42.0*f(0,-2) + 27.0*f(0,-3) - 8.0*f(0,-4) )/6.0/dx4


# 6th order
#dx fdm 6th order, center, not copied from cuda code (preprocessor function)
# legacy
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
def dy6_m24(fields, dx):
  f = fTemplate(fields)
  return ( 2.0*f(0,-2) - 24.0*f(0,-1) - 35.0*f(0,0) + 80.0*f(0,1) - 30.0*f(0,2) + 8.0*f(0,3) - f(0,4) )/60.0/dx
# dy fdm 6th order, center, copied from cuda code
def dy6_m33(fields, dx):
  f = fTemplate(fields)
  return ( -f(0,-3) + 9.0*f(0,-2) - 45.0*f(0,-1) + 45.0*f(0,1) - 9.0*f(0,2) + f(0,3) )/60.0/dx
# dxy fdm 6th order, center, copied from cuda code
def dxy6_m33_m33(fields, dx2):
  f = fTemplate(fields)
  return ( 6.0*(f(-3,-1) - f(-3,1) + f(-1,-3) - f(-1,3) - f(1,-3) + f(1,3) - f(3,-1) + f(3,1))
         + 5.0*(f(-2,-2) - f(-2,2) - f(2,-2) + f(2,2))
         + 64.0*(-f(-2,-1) + f(-2,1) - f(-1,-2) + f(-1,2) + f(1,-2) - f(1,2) + f(2,-1) - f(2,1))
         + 380.0*(f(-1,-1) - f(-1,1) - f(1,-1) + f(1,1)) )/720.0/dx2
# dxy fdm 6th order, center, some uneven contributions in 6thorder=0
def dxy6_m33_m33_2(fields, dx2):
  f = fTemplate(fields)
  return ( -f(-3,-2) + f(-3,2) - f(-2,-3) + f(-2,3) + f(2,-3) - f(2,3) + f(3,-2) - f(3,2)
         + 8.0*(f(-3,-1) - f(-3,1) + f(-1,-3) - f(-1,3) - f(1,-3) + f(1,3) - f(3,-1) + f(3,1))
         + 13.0*(f(-2,-2) - f(-2,2) - f(2,-2) + f(2,2))
         + 77.0*(-f(-2,-1) + f(-2,1) - f(-1,-2) + f(-1,2) + f(1,-2) - f(1,2) + f(2,-1) - f(2,1))
         + 400.0*(f(-1,-1) - f(-1,1) - f(1,-1) + f(1,1)) )/720.0/dx2
# laplace fdm 6th order, center, copied from cuda code
def laplace6_m33_m33(fields, dx2):
  f = fTemplate(fields)
  return ( 2.0*(f(-3,0) + f(0,-3) + f(0,3) + f(3,0))
          + 27.0*(-f(-2,0) - f(0,-2) - f(0,2) - f(2,0))
          + 270.0*(f(-1,0) + f(0,-1) + f(0,1) + f(1,0))
          - 980.0*f(0,0) )/180.0/dx2
# dxx fdm 6th order, center
def dxx6_m33(fields, dx2):
  f = fTemplate(fields)
  return ( 2.0*f(-3,0) - 27.0*f(-2,0) + 270.0*f(-1,0) - 490.0*f(0,0) + 270.0*f(1,0) - 27.0*f(2,0) + 2.0*f(3,0) )/180.0/dx2
# dyy fdm 6th order, 2 left nodes, copied to cuda code
def dyy6_m25(fields, dx2):
  f = fTemplate(fields)
  return ( - 11.0*f(0,-2) + 214.0*f(0,-1) - 378.0*f(0,0) + 130.0*f(0,1) + 85.0*f(0,2) - 54.0*f(0,3) + 16.0*f(0,4) - 2.0*f(0,5) )/180.0/dx2
# dyy fdm 6th order, center, copied from cuda code
def dyy6_m33(fields, dx2):
  f = fTemplate(fields)
  return ( 2.0*f(0,-3) - 27.0*f(0,-2) + 270.0*f(0,-1) - 490.0*f(0,0) + 270.0*f(0,1) - 27.0*f(0,2) + 2.0*f(0,3) )/180.0/dx2
#dxxx fdm 6th order, center, not copied from cuda code (preprocessor function)
# legacy
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
def dyyy6_m26(fields, dx3):
  f = fTemplate(fields)
  return ( -5.0*f(0,-2) - 424.0*f(0,-1) + 1638.0*f(0,0) - 2504.0*f(0,1) + 2060.0*f(0,2) - 1080.0*f(0,3) + 394.0*f(0,4) - 88.0*f(0,5) + 9.0*f(0,6) )/240.0/dx3
# dyyy fdm 6th order, 3 left nodes, copied to cuda code
def dyyy6_m35(fields, dx3):
  f = fTemplate(fields)
  return ( 9.0*f(0,-3) - 86.0*f(0,-2) - 100.0*f(0,-1) + 882.0*f(0,0) - 1370.0*f(0,1) + 926.0*f(0,2) - 324.0*f(0,3) + 70.0*f(0,4) - 7.0*f(0,5) )/240.0/dx3
# dyyy fdm 6th order, center, copied from cuda code
def dyyy6_m44(fields, dx3):
  f = fTemplate(fields)
  return ( -7.0*f(0,-4) + 72.0*f(0,-3) - 338.0*f(0,-2) + 488.0*f(0,-1) - 488.0*f(0,1) + 338.0*f(0,2) - 72.0*f(0,3) + 7.0*f(0,4) )/240.0/dx3
# dxyy fdm 6th order, center, copied from cuda code
def dxyy6_m33_m33(fields, dx3):
  f = fTemplate(fields)
  return ( 12.0*(-f(-3,-1) - f(-3,1) + f(3,-1) + f(3,1))
          + 5.0*(-f(-2,-2) - f(-2,2) + f(2,-2) + f(2,2))
          + 4.0*(-f(-1,-3) - f(-1,3) + f(1,-3) + f(1,3))
          + 64.0*(f(-1,-2) + f(-1,2) - f(1,-2) - f(1,2))
          + 128.0*(f(-2,-1) + f(-2,1) - f(2,-1) - f(2,1))
          + 760.0*(-f(-1,-1) - f(-1,1) + f(1,-1) + f(1,1))
          + 24.0*(f(-3,0) - f(3,0)) + 246.0*(-f(-2,0) + f(2,0))
          + 1400.0*(f(-1,0) - f(1,0)) )/720.0/dx3
# dyxx fdm 6th order, center, copied from cuda code
def dyxx6_m33_m33(fields, dx3):
  f = fTemplate(fields)
  return ( 12.0*(-f(-1,-3) - f(1,-3) + f(-1,3) + f(1,3))
          + 5.0*(-f(-2,-2) - f(2,-2) + f(-2,2) + f(2,2))
          + 4.0*(-f(-3,-1) - f(3,-1) + f(-3,1) + f(3,1))
          + 64.0*(f(-2,-1) + f(2,-1) - f(-2,1) - f(2,1))
          + 128.0*(f(-1,-2) + f(1,-2) - f(-1,2) - f(1,2))
          + 760.0*(-f(-1,-1) - f(1,-1) + f(-1,1) + f(1,1))
          + 24.0*(f(0,-3) - f(0,3)) + 246.0*(-f(0,-2) + f(0,2))
          + 1400.0*(f(0,-1) - f(0,1)) )/720.0/dx3
# dyyyy fdm 6th order, 2 left nodes, copied to cuda code
def dyyyy6_m27(fields, dx4):
  f = fTemplate(fields)
  return ( 101.0*f(0,-2) + 58.0*f(0,-1) - 1860.0*f(0,0) + 5272.0*f(0,1) - 7346.0*f(0,2) + 6204.0*f(0,3) - 3428.0*f(0,4) + 1240.0*f(0,5) - 267.0*f(0,6) + 26.0*f(0,7) )/240.0/dx4
# dyyyy fdm 6th order, 3 left nodes, copied to cuda code
def dyyyy6_m36(fields, dx4):
  f = fTemplate(fields)
  return ( -26.0*f(0,-3) + 361.0*f(0,-2) - 1112.0*f(0,-1) + 1260.0*f(0,0) - 188.0*f(0,1) - 794.0*f(0,2) + 744.0*f(0,3) - 308.0*f(0,4) + 70.0*f(0,5) - 7.0*f(0,6) )/240.0/dx4
# biharm fdm 6th order, center, copied from cuda code
def biharm6_m44_m44(fields, dx4):
  f = fTemplate(fields)
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
def dy8_m44(fields, dx):
  f = fTemplate(fields)
  return ( 3.0*f(0,-4) - 32.0*f(0,-3) + 168.0*f(0,-2) - 672.0*f(0,-1) + 672.0*f(0,1) - 168.0*f(0,2) + 32.0*f(0,3) - 3.0*f(0,4) )/840.0/dx
