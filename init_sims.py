import importlib
import yaml

def init_params(filename):
    """ initializes the default_params-Object of a simulation campaign
    takes the filename of an init_default_params.py file
    """
    spec = importlib.util.spec_from_file_location("my_parammodule", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.default_params

class RegisteringMetaclass(type):
    """ adds class to list upon subclassing of MyObject """
    my_classes = {}
    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 1:
            base_class = cls.mro()[-2]
            my_classes = base_class.my_classes
            if hasattr(base_class, "my_classes"):
                my_classes = base_class.my_classes
            else:
                my_classes = {}
                base_class.my_classes = my_classes

            my_classes[name] = cls

        super().__init__(name, bases, clsdict)


class MyObject(object, metaclass=RegisteringMetaclass):
    """ Base object provides funtionality to dump and load """
    necessary_attrs = []
    additional_attrs = []
    def __init__(self):
        pass

    def dumpd(self):
        """ dump all data in a dict """
        result = {}
        for name in self.necessary_attrs:
            if not hasattr(self, name):
                raise ValueError(f"{self.__class__.__name__} object has no attribute {name}")
            else:
                result[name] = getattr(self, name)

        for name in self.additional_attrs:
            if not hasattr(self, name):
                pass
            else:
                result[name] = getattr(self, name)

        result['class'] = type(self)

        return result

    @classmethod
    def loadd(cls, datadict):
        """ create instance from a dict """

        if not issubclass(datadict['class'], cls):
            raise ValueError(f"class {cls} was called to load itself from a datadict corresponding to class {datadict['class']}. Giving up.")
        self = cls()

        for name in cls.necessary_attrs:
            if not name in datadict:
                raise ValueError(f"{cls.__name__} needs attribute {name}. Cannot load from given dict.")
            else:
                setattr(self, name, datadict[name])

        for name in cls.additional_attrs:
            if not name in datadict:
                pass
            else:
                setattr(self, name, datadict[name])

        return self


    def dumpd_prepare_dumps(self):
        """ includes actions to make object dumpeable to string.
        Can be appended/overwritten by subclasses """
        data = self.dumpd()
        # the attribute "class" is treated separately, only class name is dumped. The check on name compatibility is done upon loading.
        data['class'] = data['class'].__name__
        my_classes = type(self).my_classes
        for key, value in data.items():
            if type(value) in my_classes.values():
                data[f"{key}_string"] = value.dumps()
                del(data[key])

        return data


    def dumps(self):
        """ dump all data in a string """
        data = self.dumpd_prepare_dumps()
        text = yaml.dump(data)

        # check, that all other attributes are loadable correctly
        if not yaml.safe_load(text) == data:
            raise ValueError("Error in dump to string via yaml: one of the attributes would not be restored correctly!")
        else:
            return text

    @classmethod
    def loads(cls, text):
        """ create instance from a string representation """
        data = yaml.safe_load(text)
        # Don't use the loading cls, but the appropriate class, as specified in the string
        cls = cls.my_classes[data['class']]
        data['class'] = cls

        for key, value in data.items():
            if key[-7:]=="_string":
                value = cls.loads(value)
                data[key[:-7]] = value
                del(data[key])

        return cls.loadd(data)


class Batch(MyObject):
    necessary_attrs = ['path_to_code', 'path_to_paramset', 'code_name']

class PicongpuBatch(Batch):
    additional_attrs = ['code_commit']
    def __init__(self):
        self.code_name = 'PIConGPU'
        self.necessary_attrs += ['path_to_initparamsfile']


class Sim(MyObject):
    necessary_attrs = ['batch', 'params', 'status', 'rundir']
    status_attrs = ['created', 'submitted', 'running', 'ready']
    # TODO: methods to determine each of the status attrs; here or in PicongpuSim?


class PicongpuSim(Sim):
    def __init__(self, batch=None):
        self.status_attrs += ['compiled']
        if not batch is None:
            self.batch = batch


    # TODO: was von hier lieber hoch in Sim schieben

    def _init__metadata(self):
        self.template_cmakeflagsfile = f"{self.batch.path_to_paramset}/cmakeFlags"

    def create_cmakeflagsfile(self):
        # TODO: Doku

        # create the line with the content
        self.default_params = init_params(self.batch.path_to_initparamsfile) # TODO: das in abstraktere fkt schieben
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

        strings = begin + ';'.join(parts) + "'"
        line = 'flags[0]="' + strings + '"'

        # write this line into the file
        with open(self.template_cmakeflagsfile, 'r') as fh:
            text = fh.read()

        delimiter = "###########################################################################"

        parts = text.split(delimiter)
        if len(parts) != 3:
            raise ValueError("I expect to find a cmakeFlags-file with exactly two lines with a lot of #")

        start, _, end = parts
        text = "\n".join((start, delimiter, '\n'+line+'\n', delimiter, end))

        return text # TODO: dump to file or return string?
        import importlib


