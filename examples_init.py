from create_manage_analyze.init_sims import *

b = PicongpuBatch()
b.path_to_code = "/home/goethe93/simulations/pulseshape1d_coll/picongpu"
b.path_to_paramset = "/home/goethe93/simulations/pulseshape1d_coll/paramSet"
b.code_commit = "bd6f4c84d9bf4d98cdf6e43dff82ef7fe826fb48"
b.path_to_initparamsfile = f"{b.path_to_paramset}/_init_default_params.py"
dic = b.dumpd()
a = PicongpuBatch.loadd(dic)

print(a.dumpd())

params_metadata = dict( # other form of representation of the data that can be set
    cmakeflags = { # direct correspondence to a cmakeFlag
        # name: description, default, param_name, function
        # TODO: this is not DRY, defaults here are unconnected to those in the paramfile
        "laser.profile": ["name of laser profile", "DivPulses", "PARAM_LASERPROFILE", None],
        "laser.a0": ["dimensionless amplitude", 50, "PARAM_A0", None],
        "laser.w0": ["laser width in micron", 50, "PARAM_W0", None],
        "laser.tau": ["temporal fwhm in fs", 30, "PARAM_PULSELENGTH", None],
        "laser.pulse_init": ["pulse initialisation time in pulse sigmas", 9.0, "PARAM_PULSEINIT", None],
        "laser.pol": ["laser polarization", "LINEAR_X", "PARAM_LASERPOL", None],
        "laser.time_start": [f"time of start of init in fs", -200, f"PARAM_TIME_START", None],
        "laser.time_end": [f"time of end of init in fs", 100, f"PARAM_TIME_END", None],

        "target.dens": ["density in critical densities", 20, "PARAM_TARGETDENS", None],
        "target.size": ["target thickness in micron", 555, "PARAM_TARGETSIZE", None],
        "target.pos": ["target y-position in micron", 4, "PARAM_TARGETPOS", None],
        "target.sfront": ["exponential scalelength on the front side in micron", 0.0, "PARAM_TARGETFS1", None],
        "target.sf1_len": ["exponential scalelength on the front side in micron", 0.0, "PARAM_TARGETFL1", None],
        "target.srear": ["exponential scalelength on the rear side in micron", 0.0, "PARAM_TARGETRS1", None],
        "target.sr1_len": ["exponential scalelength on the rear side in micron", 0.0, "PARAM_TARGETRL1", None],
        "target.temp": ["target initial temperature, in keV per particle", 0, "PARAM_TARGETTEMP", None],

        "grid.res": ["resolution: cells per wavelength", 32, "PARAM_RES", 'lambda x: float(x)'],
        "grid.ppc": ["nr of particles per cell", 40, "PARAM_PPC", 'lambda x: f"{int(x)}u"'],
        "grid.pshape": ["particle shape order 1-4", 3, "PARAM_PSHAPE", 'lambda x: {1:"CIC", 2:"TSC", 3:"PCS", 4:"P4S"}[x]'],
    }
)

for nr in '0123':
    params_metadata["cmakeflags"][f"laser.int_point_{nr}"] = [f"relative intensity of ramppoint {nr}", 1e-12, f"PARAM_INT_{nr}", None]
    params_metadata["cmakeflags"][f"time_point_{nr}"] = [f"time of ramppoint {nr} in fs", 0, f"PARAM_TIME_{nr}", None]
for i in range(1, 9):
    nr = f"{i:02d}"
    params_metadata["cmakeflags"][f"laser.p{nr}i"] = [f"relative intensity of gaussian pulse {nr}", 0, f"PARAM_P{nr}i", None]
    params_metadata["cmakeflags"][f"laser.p{nr}t"] = [f"time of gaussian pulse {nr} in fs", 0, f"PARAM_P{nr}t", None]
    params_metadata["cmakeflags"][f"laser.p{nr}l"] = [f"length of gaussian pulse {nr} (in mainpulse pulselengths)", 1, f"PARAM_P{nr}l", None]
    params_metadata["cmakeflags"][f"laser.s{nr}i"] = [f"relative peak intensity of sin-pulse {nr}", 0, f"PARAM_S{nr}i", None]
    params_metadata["cmakeflags"][f"laser.s{nr}s"] = [f"start time of sin-pulse {nr} in fs", -120, f"PARAM_S{nr}s", None]
    params_metadata["cmakeflags"][f"laser.s{nr}e"] = [f"end time of sin-pulse {nr} in fs", -80, f"PARAM_S{nr}e", None]
    

flags = params_metadata['cmakeflags']
default_values = {}
descriptions = {}
actions = {'cmakeflags': {}}

for name, vals in flags.items():
    default_values[name] = vals[1]
    descriptions[name] = vals[0]
    actions["cmakeflags"][name] = vals[-2:]

class ParamsMetadata(MyObject):
    """ class representing all parameters of a simulation batch 
    """
    necessary_attrs = ['default_values', 'descriptions', 'actions']
    additional_attrs = []

    # TODO: sort out everything special to picongpu into mixin

    def _create_cmakeflags(self, values):
        """ create cmakeflags file according to parameter values """
        metadata = self.actions['cmakeflags']
        flags = {}
        for name, value in values.items():
            try:
                flagname, functext = metadata[name]
            except KeyError:
                raise KeyError(f"Sorry, the passed paramter {name} (value {value}) is not known in the metadata!\n\nKnown parameters: {list(metadata.keys())}")
            if not functext is None:
                func = eval(functext)
                value = func(value)
            flags[flagname] = value


        print(flags)

class Params(MyObject):
    """ parameter values of one simulation
    """
    necessary_attrs = ['values', 'metadata']
    

pm = ParamsMetadata()
pm.default_values = default_values
pm.descriptions = descriptions
pm.actions = actions

pm._create_cmakeflags({"laser.a0": 40, "grid.pshape": 2, 'grid.ppc': 44})

