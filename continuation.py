import numpy as np


FixedTimestep = True
dep = False
cont = False
bool_input = False
bool_adopt_params = False
pre = 5000,
outputDT = 100

class DictError(TypeError):
  pass

class Continuation():
  def __init__(self, ParamsCMDArgs):
    self.ParamsCMDArgs = {}
    self.key = None
    self.values = None
    self.executable = "none"

  def create_subdir(self, args):
    return 

  def construct_IC(self):
    pass

  def init_script(self, name):
    with open(name + ".sh", "w") as text_file:
      print("""#!/bin/bash
      """, file = text_file)

  def write_script(self, name, ParamsCMDArgs):
    ArgsOfEachStep = self.get_ArgsOfEachStep(ParamsCMDArgs)
    with open(name + ".sh", "a") as text_file:
      for args in ArgsOfEachStep:
        Subdir = self.create_subdir(args)
        SubdirCMD = "cd {Subdir:} \n".format(Subdir)
        ExeCMD = self.get_ExeCMD(args)
        print(SubdirCMD, file = text_file)
        print(ExeCMD, file = text_file)
        print("cd ..\n", file = text_file)

  def get_ArgsOfEachStep(self, ParamsCMDArgs):
    ArgsOfEachStep = []
    for value in self.values:
      ParamsCMDArgs[self.key] = value
    ArgsOfEachStep.append(ParamsCMDArgs)
    return ArgsOfEachStep
  def get_ExeCMD(self, args):
    # args is a dictionary
    if(type(args) is not dict):
      raise DictError("args is not a dictionary")
    ArgsStr = "../" + self.executable + " "
    for key in args.keys():
      ArgsStr += "-{key:} {value:} ".format(key, args[key])
    ArgsStr += "-in " + self.IC + " "
    return ArgsStr + "\n"