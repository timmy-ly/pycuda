#!/usr/bin/env python3

import numpy as np
import warnings
import attributes
# import AttributesCH

# methods that apply to all cuda problems
class problem:
  def __init__(self):
    pass

  #read cuda bin file
  def readbin(self, filepath, dtype, nof, Nx, Ny):
    if dtype == 'float':
      array = np.fromfile(filepath,np.dtype('float32'))
      array = array.astype('float')
    else:
      array = np.fromfile(filepath,np.dtype('float64'))
      array = array.astype('double')
    array = array.reshape((nof*Nx,Ny))
    return array
  # read balancedata
  def readbalance(self, filepath, n=0):
    with open(filepath,'r') as f:
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = np.loadtxt(f, skiprows=1, ndmin=2)
    # create copy of data that keeps column 0 and removes columns
    # 1 to n, np.s_ is a slice, last argument is axis
    return np.delete(data,np.s_[1:-n],1)
  # finite difference stencils
  def dy4_04(self, field, dy):
    # forward difference from node 0 to node 4
    return (-25*field + 4*12*np.roll(field,-1,1) - 3*12*np.roll(field,-2,1) + 4/3*12*np.roll(field,-3,1) - 12/4*np.roll(field,-4,1))/(12*dy)
  def dy4_m13(self, field, dy):
    # node -1 to node 3
    return (-3*np.roll(field,1,1) - 10*field + 18*np.roll(field,-1,1) - 6*np.roll(field,-2,1) + np.roll(field,-3,1) )/(12*dy)
  def dy4_m22(self, field, dy):
    # node -2 to node 2
    return (np.roll(field,2,1) - 8*np.roll(field,1,1) + 0 + 8*np.roll(field,-1,1) - np.roll(field,-2,1) )/(12*dy)
  def dyyy4_06(self, field, dy3):
    # node 0 to 6
    return (-49*field + 29*8*np.roll(field,-1,1) - 461*np.roll(field,-2,1) + 62*8*np.roll(field,-3,1) - 307*np.roll(field,-4,1) + 13*8*np.roll(field,-5,1) - 15*np.roll(field,-6,1))/(8*dy3)
  def dyyy4_m15(self, field, dy3):
    # node -1 to node 5
    return (-15*np.roll(field,1,1) + 56*field - 83*np.roll(field,-1,1) + 64*np.roll(field,-2,1) -29*np.roll(field,-3,1) + 8*np.roll(field,-4,1) - np.roll(field,-5,1) )/(8*dy3)
  def dyyy4_m24(self, field, dy3):
    # node -1 to node 5
    return (-np.roll(field,2,1) - 8*np.roll(field,1,1) + 35*field - 48*np.roll(field,-1,1) + 29*np.roll(field,-2,1) -8*np.roll(field,-3,1) + np.roll(field,-4,1) )/(8*dy3)


# methods and attributes that apply to all precipiti problems
class precipiti(problem):
  def __init__(self, nof = 4):
    super().__init__()
    # model, BC, IC
    self.params = attributes.AttributesPRECIPITI(nof = nof)

  def readparams(self, filepath):
    with open(filepath,'r') as f:
      lines = f.readlines()		#list, not array
    for i in np.arange(len(lines)):
      if lines[i].split()[0] == 'Nx':
        self.params.Nx = int(lines[i].split()[1])
      elif lines[i].split()[0] == 'Ny':
        self.params.Ny = int(lines[i].split()[1])
      elif lines[i].split()[0] == 'Lx':
        self.params.Lx = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'Ly':
        self.params.Ly = float(lines[i].split()[1])
      elif lines[i].split()[0] == 't':
        self.params.t = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'h0':
        self.params.h0 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'beta':
        self.params.beta = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'g':
        self.params.g = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'v':
        self.params.v = float(lines[i].split()[1])
  
  # problem: all values are strings
  # def readparams_dict(self, filepath):
  #   d = {}
  #   with open(filepath,'r') as f:
  #     for line in f:
  #       try:
  #         (key, val) = line.split()
  #       except ValueError:
  #         continue
  #       d[key] = val
  #   return d

  def readparam(self, filepath, param):
    with open(filepath,'r') as f:
      lines = f.readlines()
    for i in np.arange(len(lines)):
      if lines[i].split()[0] == param:
        param_value_str = lines[i].split()[1]
        break
    return param_value_str