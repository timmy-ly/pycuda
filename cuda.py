#!/usr/bin/env python3

import numpy as np
import attributes
# import AttributesCH

class problem:
  def __init__(self):
    pass

  #read cuda bin file
  def readbin(self, filepath):
    if self.dtype == 'float':
      array = np.fromfile(filepath_bin,np.dtype('float32'))
      array = array.astype('float')
    else:
      array = np.fromfile(filepath_bin,np.dtype('float64'))
      array = array.astype('double')
    array = array.reshape((self.nof*self.Nx,self.Ny))
    return array

class precipiti(problem):
  def __init__(self):
    super().__init__()
    # model, BC, IC
    self.params = attributes.AttributesPRECIPITI()

  def readparams(filepath):
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


obj = precipiti()
print(obj.params.nof)