import numpy as np
from pathlib import Path
from cuda import readbin, dat
from solution import solution


# methods and attributes that apply to all precipiti problems
class precipiti(solution):
  def __init__(self, path = None, nof = 4):
    super().__init__()
    self.nof = nof
    # model
    self.v = 0.2
    self.ups0 = 0.0
    self.chi = 1.5
    self.ups1 = 0.0
    self.ups2 = 1.0
    self.ups3 = -5.0
    self.g = 10**-3
    self.beta = 2.0
    self.lamb = 1.8
    self.LAMB = 1.4
    self.sigma = 1.8
    # model/BC
    self.alpha = 0.0
    # BC
    self.bc = 1
    # BC/IC
    self.h0 = 20.0
    self.c0 = 0.3
    self.phi0 = 1.0
    # IC
    self.noise = 0.0
    self.h1 = 0.0
    if path is not None:
      self.path = Path(path)
      # try:
      self.readparams(self.path)
      self.fields = readbin(self)
      # except FileNotFoundError:
        # print("no corresponding .dat and/or .bin file")




  @property
  def v(self):
    return self._v

  @v.setter
  def v(self, value):
    self._v = value  

  def readparams(self, filepath=None):
    if filepath is None:
      filepath = self.path
    filepath = dat(filepath)
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
      elif lines[i].split()[0] == 'h0':
        self.h0 = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'beta':
        self.beta = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'g':
        self.g = float(lines[i].split()[1])
      elif lines[i].split()[0] == 'v':
        self.v = float(lines[i].split()[1])
  
