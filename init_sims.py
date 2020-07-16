#!/usr/bin/env python3

""" Functions to initialize the simulation sets with parameter variation """

import importlib

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

