import numpy as np

class DictError(TypeError):
  pass

class Continuation():
  def __init__(self, ParamsCMDArgs, ParamsOther):
    self.SubdirPrefix = ParamsOther["SubdirPrefix"]
    self.ParamsCMDArgs = ParamsCMDArgs
    self.ParamsOther = ParamsOther
    self.key = ParamsOther["key"]
    self.values = ParamsOther["values"]
    self.executable = ParamsOther["executable"]
    self.IC = ParamsOther["IC"]
    self.FixedTimestep = ParamsOther["FixedTimestep"]
    self.FirstRun = True
    self.LastFramePreviousRun = None
    self.name = ParamsOther["name"]
    self.cont = True
    if("cont" in ParamsOther):
      self.cont = ParamsOther["cont"]

  def set_Subdir(self):
    self.Subdir = self.SubdirPrefix

  def init_script(self):
    with open(self.name + ".sh", "w") as text_file:
      print("""#!/bin/bash
      """, file = text_file)

  def write_script(self):
    ArgsOfEachStep = self.get_ArgsOfEachStep()
    with open(self.name + ".sh", "a") as text_file:
      for args in ArgsOfEachStep:
        Subdir = self.get_Subdir(args)
        MkdirCMD = "mkdir -p {Subdir:} ".format(Subdir = Subdir)
        SubdirCMD = "cd {Subdir:} ".format(Subdir = Subdir)

        ExeCMD = self.get_ExeCMD(args)
        print(MkdirCMD, file = text_file)
        print(SubdirCMD, file = text_file)
        print(ExeCMD, file = text_file)
        print("cd ..\n", file = text_file)
        self.LastFramePreviousRun = Subdir

  def get_ArgsOfEachStep(self):
    ArgsOfEachStep = []
    for value in self.ParamsOther["values"]:
      ArgsCopy = self.ParamsCMDArgs.copy()
      ArgsCopy[self.key] = value
      ArgsOfEachStep.append(ArgsCopy)
    return ArgsOfEachStep
  def get_ExeCMD(self, args):
    # args is a dictionary
    if(type(args) is not dict):
      raise DictError("args is not a dictionary")
    ArgsStr = "../" + self.executable + " "
    for key in args.keys():
      ArgsStr += "-{key:} {value:} ".format(key = key, value = args[key])
    ArgsStr += "-in " + self.IC + " "
    if(self.FixedTimestep):
      ArgsStr += "-outputDT {outputDT:d} ".format(outputDT = self.ParamsOther["outputDT"])
    else:
      ArgsStr += "-pre " + self.ParamsOther["pre"] + " "
    ArgsStr+= "-t 0 "
    if(self.FirstRun or (not self.cont)):
      ArgsStr += "-in " + self.IC + " "
      self.FirstRun = False
    else:
      ArgsStr += "-in ../" + self.LastFramePreviousRun + "/frame "
    return ArgsStr