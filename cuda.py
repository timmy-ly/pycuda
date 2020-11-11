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
  def readbalance(self, filepath):
    with open(filepath,'r') as f:
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data = np.loadtxt(f, skiprows=1)
    return data

# methods and attributes that apply to all precipiti problems
class precipiti(problem):
  def __init__(self):
    super().__init__()
    # model, BC, IC
    self.params = attributes.AttributesPRECIPITI()

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


# obj = precipiti()
# obj.params.nof = 5
# print(obj.params.nof)