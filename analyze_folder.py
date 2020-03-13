#!/usr/bin/env python3

""" Functions to analyze PIConGPU simulation folders """

import adios as bp
import h5py
import numpy as np
import matplotlib.pyplot as plt

"""
TODO: documentation
TODO: extract all common stuff into decorator
"""

"""
Functions for parsing files
"""
def energy_cutoff_from_hist(filename, treshhold=0):
    """ return the timeline and max-value of cutoff energy """
    with open(filename, 'r') as f:
        zeilen = f.readlines()

    energies = [0] + list(map(float, zeilen[0].split()[2:-2]))
    lastline = len(zeilen)-2
    timeline = []
    # fill in the data, one dict per step
    for nrzeile in range(lastline // 4, lastline+1):
        bins = np.array(list(map(float, zeilen[nrzeile].split()[1:-2])))
        thisdata = {}
        timeline.append(thisdata)
        step = int(zeilen[nrzeile].split()[0])
        thisdata['step'] = step
        thisdata['linenr'] = nrzeile
        thisdata['emax'] = energies[np.where(bins > threshold)[0][-1]] / 1000 # in MeV

    # now compute overall maximum energies
    tuples = [(r['emax'], r['step']) for r in timeline]
    value, timestep = max(tuples)
    return tuples, value, timestep

