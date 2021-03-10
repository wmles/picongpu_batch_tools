#!/usr/bin/env python3

""" Functions to initialize the simulation sets with parameter variation """

import importlib
import os, datetime
from shutil import copyfile

"""
TODO: documentation
TODO: extract all common stuff into decorator?
"""

def init_params(filename):
    """ initializes the default_params-Object of a simulation campaign
    takes the filename of an init_default_params.py file
    """
    spec = importlib.util.spec_from_file_location("my_parammodule", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.default_params

# TODO: eigentlich sollte die Funktion mit in die Params-Klasse rein
# und eigentlich wär's besser, wenn die default_params nicht in einem python-skript, sondern in einer reinen Datendatei stehen (die lambdas als strings), damit sie auch dumpbar sind

class Params(object):
    def __init__(self, default_params, paramdict={}):
        if isinstance(default_params, str): 
            self.default_params = init_params(default_params)
        else:
            self.default_params = default_params

        self._paramdict = paramdict
        self.cfgstring = ""
    
    def print_cmakeflagsstring(self):
        begin = "-DPARAM_OVERWRITES:LIST='"
        part_template = "-D{flag}={val}"
        parts = []
        paramdict = self._paramdict.copy()
        for catname, category in self.default_params.items:
            for attname, attribute in category.items:
                key = (f"{catname}.{attname}")
                if key in paramdict:
                    value = paramdict.pop(key)
                else:
                    value = attribute.value
                flag = attribute.cmake_flag 
                if flag is not None:
                    parts.append(part_template.format(flag=flag, val=attribute.func_val_to_flag(value)))
        if paramdict: # if not empty because of the pop()-s
            raise ValueError(
                "There were some keys specified in the paramdict that are not known in default_params" + 
                f"\n{paramdict}\n" +
                "The possible values to be specified are:\n" +
                str([f"{cname}.{aname}" for (cname, cat) in self.default_params.items for (aname, att) in cat.items])
            )
            
        return begin + ';'.join(parts) + "'" 

    def edit_cmakefile(self):
        infilename = f"{self.pic_paramSet}/cmakeFlags"
        outfilename = f"{self.compiledir}/cmakeFlags"
        delimiter = "###########################################################################"

        with open(infilename, "r") as fh:
            text = fh.read()
        
        parts = text.split(delimiter)
        if len(parts) != 3:
            raise ValueError("I expect to find a cmakeFlags-file with exactly two lines with a lot of #")

        text = '\n\n'.join([parts[0], delimiter, self.batch.print_cmakeflagsstring(index=self.number), delimiter, parts[2]])
        
        with open(outfilename, "w") as fh:
            fh.write(text)
        
    def edit_cfg(self):
        infilename = f"{self.pic_paramSet}/etc/picongpu/base.cfg"
        outfilename = f"{self.compiledir}/etc/picongpu/base.cfg"
        delimiter = "# my_params"
        
        with open(infilename, "r") as fh:
            text = fh.read()
        
        parts = text.split(delimiter)
        if len(parts) != 3:
            raise ValueError(f"I expect to find a cmakeFlags-file with exactly two lines with {delimiter}")

        text = '\n\n'.join([parts[0], delimiter, self.cfgstring, delimiter, parts[2]])
        
        with open(outfilename, "w") as fh:
            fh.write(text)

    def init_compiledir(self, basedir, pic_paramSet, dirnametemplate="{:03d}", dont_write=False):
        nr = self.number
        name = f"{basedir}/builds/{dirnametemplate.format(nr)}"
        self.compiledir = name
        self.rundir = f"{basedir}/runs/{dirnametemplate.format(nr)}"
        self.pic_paramSet = pic_paramSet
        if not dont_write:
            os.makedirs(name)
            os.system(f"cp -r {pic_paramSet}/* {name}")

    @property
    def compilestring(self):
        return f"cd {self.compiledir}; pic-build > compile.log 2>&1 & "
    @property
    def submitstring(self):
        return f"cd {self.compiledir}; tbg -s sbatch -c etc/picongpu/base.cfg -t etc/picongpu/hemera-hzdr/{self.queue}.tpl {self.rundir}"


    def check_runstatus(self, suffix=''):
        try:
            with open(f"{self.rundir}{suffix}/simOutput/output", 'r') as f:
                zeilen = f.readlines()
        except FileNotFoundError:
            self.started = False
            return

        self.started = True
        if 'full simulation time' in zeilen[-1]:
            self.completed = True
            self.runtime = zeilen[-1].split(' = ')[-1].replace(' sec', '')
        else:
            self.completed = False

    def backup_outputs(self, postfix='a', names=None, force=False):
        """ for the workflow of repeating sims into the same folder with submit.start
        backups the textfiles by renaming to use the postfix """
        outdir = f'{self.rundir}/simOutput/'
        allfiles = os.listdir(outdir)
        if not names:
            names = [name for name in allfiles if name[-4:]=='.dat'] + ['output']
        duplicates = [] # for error handling
        for name in names:
            if f"{name}_{postfix}" in allfiles:
                duplicates.append(name)
        if not force:
            self.check_runstatus()
            if not self.completed:
                raise ValueError("Simulation in {self.rundir} seems not finished")
            if len(duplicates) == len(names):
                return True
            elif 0 < len(duplicates) < len(names):
                print(f"Only some files already exist with postfix {postfix}: {duplicates}")
                return False

        for name in names:
            copyfile(f"{outdir}{name}", f"{outdir}{name}_{postfix}")
        with open(f"{outdir}/datetime_{postfix}", 'w') as f:
            f.write(str(os.path.getmtime(f"{outdir}output")))


    def __str__(self):
        return f"Params-instance"
    def __repr__(self):
        return str(self)

    
from collections.abc import MutableSequence
class Batch(MutableSequence):
    def __init__(self, default_params, pic_paramSet, paramdictlist=[{}]):
        self.default_params = default_params
        self.pic_paramSet = pic_paramSet
        self.paramslist = []
        for i, pdict in enumerate(paramdictlist):
            paramset = Params(default_params, pdict)
            paramset.number = i
            paramset.batch = self
            self.paramslist.append(paramset)
    
    # forward the methods for the MutableSequence to the inner paramslist
    for name in "__getitem__, __setitem__, __delitem__, __len__, insert".split(', '):
        exec(f"def {name}(self, *args):\n return self.paramslist.{name}(*args)")
    # override the setting methods to check for the value type
    def __setitem__(self, key, value):
        """ only allow Params objects as values """
        if not isinstance(value, Params):
            raise ValueError("Can only add instances of Params to Batch")
        else:
            value.number = key
            self.paramslist[key] = value
    def insert(self, index, value):
        """ only allow Params objects as values """
        if not isinstance(value, Params):
            raise ValueError("Can only add instances of Params to Batch")
        else:
            value.number = index
            self.paramslist.insert(index, value)
    
    def print_cmakeflagsstring(self, index=None):
        begin = 'flags[{nr}]="'
        lines = []
        if index is None:
            paramslist = self
        elif isinstance(index, int):
            paramslist = [self[index]]
        else:
            raise NotImplementedError("understand only integer indices, but more possible to implement")
            
        for nr, p in enumerate(paramslist):
            lines.append(begin.format(nr=nr) + p.print_cmakeflagsstring() + '"')
        
        return '\n'.join(lines)
            
    def __str__(self):
        return f"Batch-instance with {len(self)} Params"
    def __repr__(self):
        return f"Batch-instance with {len(self)} Params"

    def create_compiledirs(self, basedir, indices=None, dont_write=False):
        if indices is None:
            indices = slice(None)
        if isinstance(indices, int):
            indices = [indices]
        
        paramslist = self[indices]
        os.chdir(basedir)
        for ps in paramslist:
            # create new directory for paramSet and edit cmakeFlags and cfg
            ps.init_compiledir(basedir=basedir, pic_paramSet=self.pic_paramSet, dont_write=dont_write)
            ps.edit_cmakefile()
            ps.edit_cfg()
       
        cstrings, sstrings = [], []
        
        for ps in paramslist:
            if not hasattr(ps, "queue"):
                ps.queue = "k80"

        cstrings = [ps.compilestring for ps in paramslist]
        sstrings = [ps.submitstring for ps in paramslist]

        return cstrings, sstrings
