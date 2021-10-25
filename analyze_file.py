#!/usr/bin/env python3

""" Functions to analyze output files produced by PIConGPU """

import adios as bp
import h5py
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

"""
TODO: documentation
TODO: extract all common stuff into decorator/functions
"""


class AnalyzeAngularDistributions(object):
    """ collects methods for analyzing the angular distributions and spectra """
    @staticmethod
    def rectangle_from_particles(filename, ts=None):
        """ Analyzes the particles-output and returns a histogram
        with the bins (E, thetax, thetaz)
        Thetax and thetaz are not angles in any usual sperical angle
        sense, but comes from the p_x,z / p_y ratio, so it would equal
        theta if pz or px were zero. 
        """
        if ts is None: # then try to guess
            ts = filename.replace('.bp', '').split('_')[-1]

        # create the arrays for the energy- and angle-coordinates
        # definitions:
        Emax = 240
        txmax, tzmax = 20, 20
        Nx, Nz, NE = 24, 24, 50

        bins_E = np.linspace(0, Emax, NE+1)       # Emax in MeV
        bins_thetax = np.linspace(-txmax, txmax, Nx+1) # in deg
        bins_thetaz = np.linspace(-tzmax, tzmax, Nz+1) # in deg
        bins_py = [0., 1e9]                       # one bin - filter out backward

        # unit conversions:
        m_p = 1.6726219e-27 # in SI
        c = 3e8 # in SI
        J2MeV = 6.242e+12

        # new bins
        E_0 = m_p * c**2                                    # in SI
        bins_E_SI = bins_E / J2MeV                          # in SI
        bins_p2_SI = ((bins_E_SI + E_0)**2 - E_0**2) / c**2 # in SI
        bins_tanx = np.tan(bins_thetax / 180 * np.pi)       # the tan-values
        bins_tanz = np.tan(bins_thetaz / 180 * np.pi)       # the tan-values

        diffs_E = np.diff(bins_E)
        values_E = (bins_E[1:] + bins_E[:-1]) / 2
        diffs_thetax = np.diff(bins_thetax) / 180 * np.pi
        diffs_thetaz = np.diff(bins_thetaz) / 180 * np.pi
        values_thetax = (bins_thetax[:-1] + bins_thetax[1:]) / 360 * np.pi
        values_thetaz = (bins_thetaz[:-1] + bins_thetaz[1:]) / 360 * np.pi
        # !!!!!!!!!! Achtung, das ist eine Näherung die nur gilt wenn thetax oder thetaz klein ist:
        diffs_thetas = diffs_thetax.reshape((Nx, 1)) * diffs_thetaz.reshape((1, Nz))

        with bp.File(filename) as fh:
            px = fh[f'/data/{ts}/particles/H_all/momentum/x'][:]
            py = fh[f'/data/{ts}/particles/H_all/momentum/y'][:]
            pz = fh[f'/data/{ts}/particles/H_all/momentum/z'][:]
            un = fh[f'/data/{ts}/particles/H_all/momentum/y'].unitSI.value
            we = fh[f'/data/{ts}/particles/H_all/weighting'][:]

            p_y = py / we
            p_x = px / we
            p_z = pz / we
            tx = px / py
            tz = pz / py

            bins_p2 = bins_p2_SI / un**2
            hist, bin_edges = np.histogramdd(
                (p_x**2 + p_y**2 + p_z**2, tx, tz, p_y),
                bins=(bins_p2, bins_tanx, bins_tanz, bins_py),
                weights=we
            )

        # norm and remove the last dimension of length 1
        hist = hist[:, :, :, 0] / (              # norm to particles per MeV and sterad
            diffs_E.reshape((NE, 1, 1))          # per MeV
            * diffs_thetas.reshape((-1, Nx, Nz)) # per sterad
        )
        bin_edges = bin_edges[:-1]

        return hist, bins_E, bins_thetax, bins_thetaz


    @staticmethod
    def polar_from_particles(filename, histargs={}, ts=None):
        """ Analyzes the particles-output and returns a histogram
        with the bins (E, theta, phi), i.e. the usual sperical angles
        with respect to the laser propagation direction.
        returns normalized (particles per MeV and sterad) data.
        """
        if ts is None: # then try to guess
            ts = filename.replace('.bp', '').split('_')[-1]
        
        # add default parameters if not in histargs and copy to better readable object
        if histargs is None: # None should be valid default, too
            histargs = {}

        p = SimpleNamespace(**dict(
            thetamin=0.8, 
            thetamax=20,
            Emax=240,
            Nt=24,
            Np=120,
            NE=60,
        ))
        for k, v in histargs.items():
            setattr(p, k, v)
        Np4 = p.Np // 4 # adjust/correct the number of phi-bins
        p.Np = Np4 * 4  # it's required to be a multiple of 4
                    
        # construct the arrays for binning
        bins_E       = np.linspace(0, p.Emax, p.NE+1)              # Emax in MeV
        bins_theta   = np.linspace(p.thetamin, p.thetamax, p.Nt+1) # in deg
        bins_phi     = np.linspace(0, 2*np.pi, p.Np+1)             # in rad
        bins_sign    = [-1e99, 0, 1e99]                            # find sign

        # unit conversions:
        m_p = 1.6726219e-27 # in SI
        c = 3e8 # in SI
        J2MeV = 6.242e+12
        E_0 = m_p * c**2                                      # in SI

        # new bin borders that make it computationally cheaper
        bins_E_SI   = bins_E / J2MeV                          # in SI
        bins_p2_SI  = ((bins_E_SI + E_0)**2 - E_0**2) / c**2  # in SI
        bins_tan2t  = np.tan(bins_theta / 180 * np.pi) ** 2
        bins_phi_90 = np.linspace(0, 90, Np4, endpoint=False) # quarter of the circle
        bins_tan2p       = 1e99 * np.ones(Np4+1)
        bins_tan2p[:Np4] = np.tan(bins_phi_90 / 180 * np.pi) ** 2
        diffs_E = np.diff(bins_E)                             # for normalize per MeV

        # get the data of the particles
        with bp.File(filename) as fh:
            px = fh[f'/data/{ts}/particles/H_all/momentum/x'][:]
            py = fh[f'/data/{ts}/particles/H_all/momentum/y'][:]
            pz = fh[f'/data/{ts}/particles/H_all/momentum/z'][:]
            un = fh[f'/data/{ts}/particles/H_all/momentum/y'].unitSI.value
            we = fh[f'/data/{ts}/particles/H_all/weighting'][:]

        # compute derived quantities of the particles
        p_x = px / we
        p_y = py / we        
        p_z = pz / we
        p2y = p_y ** 2
        p2x = p_x ** 2
        p2z = p_z ** 2
        tan2t = (p2x + p2z) / p2y
        tan2p = p2z / p2x
        bins_p2 = bins_p2_SI / un**2

        # do the multidimensional binning
        hist_raw, bin_edges = np.histogramdd(
            (p2x+p2y+p2z, tan2t, tan2p, p_x, p_z, p_y),
            bins=(bins_p2, bins_tan2t, bins_tan2p, bins_sign, bins_sign, bins_sign),
            weights=we
        )

        # project the four quarters back on the whole circle
        # and remove last dimension of length 1
        # and normalize to sterad and MeV 
        costheta = np.cos(bins_theta / 180 * np.pi)
        diffsterad = ( - np.diff(costheta) * 2*np.pi / p.Np).reshape((1, p.Nt, 1))
        normfactor = diffs_E.reshape((p.NE, 1, 1)) * diffsterad
        hist = np.empty((p.NE, p.Nt, p.Np))
        hist[:, :, 0*Np4:1*Np4] = hist_raw[:, :, :, 1, 1, 1] / normfactor
        hist[:, :, 1*Np4:2*Np4] = hist_raw[:, :, ::-1, 0, 1, 1] / normfactor
        hist[:, :, 2*Np4:3*Np4] = hist_raw[:, :, :, 0, 0, 1] / normfactor
        hist[:, :, 3*Np4:4*Np4] = hist_raw[:, :, ::-1, 1, 0, 1] / normfactor
            
        return hist, bins_E, bins_theta, bins_phi


class AnalyzeDensities(object):
    @staticmethod
    def lineout_along_center(filename, ts=None, field="H_density", cellsz=10, cellsx=10, normfactor=1.742e27):
        if ts is None: # then try to guess
            ts = filename.replace('.bp', '').split('_')[-1]

        with bp.File(filename) as fh:
            data = fh[f"/data/{ts}/fields/{field}"]
            if data.ndim == 2:
                nx, ny = data.shape
                raise NotImplementedError
            elif data.ndim == 3:
                nz, ny, nx = data.shape
                if cellsz > nz or cellsx > nx:
                    raise ValueError("Requested to cut out a slice bigger than the whole array")
                    
                startx = nx // 2 - cellsx // 2
                startz = nz // 2 - cellsz // 2
                dataslice = data[startz:startz+cellsz, :, startx:startx+cellsx]
                # norm to critical density and project on axis
                lineout = np.mean(dataslice, axis=(0,2)) * data.attrs['unitSI'].value / normfactor
                return lineout

    def slice_from_particles(filename, ts=None, particle="H_all", axis=2, thickness=0.8):
        """ takes particle output to compute density slice along center """
        with bp.File(filename) as fh:
            x = fh[f'/data/{ts}/particles/{particle}/positionOffset/x'][:]
            y = fh[f'/data/{ts}/particles/{particle}/positionOffset/y'][:]
            z = fh[f'/data/{ts}/particles/{particle}/positionOffset/z'][:]

        hist_raw, bin_edges = np.histogramdd(
            (p2x+p2y+p2z, tan2t, tan2p, p_x, p_z, p_y),
            bins=(bins_p2, bins_tan2t, bins_tan2p, bins_sign, bins_sign, bins_sign),
            weights=we
        )

    @staticmethod
    def arbitrary_slice_from_particles(
                filename, ts=None,
                vec1=[1, 0, 0],
                vec2=[0, 1, 0],
                center_point=[432, 450, 216],
                poslims=[[-12, 12],[-15, 21]],
                mu_in_cells = 30,
                filter_point=None,
                filter_thickness=45,
                bins_E=np.linspace(0, 240, 241),
                species='H_all',
                verbose=True,
            ):
        """ returns a vec1-vec2-position-histogram, with energy as additional axis
        if filter_thickness is a number, filter +-filter_thickness around filter_point in the 
        direction of vec1 x vec2
        if it's an iterable, take it as the bins. The histogram has dimension one more in that case
        """
        if ts is None: # then try to guess
            ts = filename.replace('.bp', '').split('_')[-1]

        if filter_point is None:
            filter_point = center_point
        vec3 = np.cross(vec1, vec2)

        # some unit conversions:
        m_p = 1.6726219e-27 # in SI
        c = 3e8             # in SI
        J2MeV = 6.242e+12
        E_0 = m_p * c**2    # in SI

        ###
        ### compute all bins - we need bins in pic-units for effective binning, and bins for plotting
        ###

        bins_E_SI = bins_E / J2MeV 
        bins_p2_SI = ((bins_E_SI + E_0)**2 - E_0**2) / c**2  # in SI

        (pos1_1, pos1_2), (pos2_1, pos2_2) = poslims
        bins_vec1_mu = np.linspace(pos1_1, pos1_2, round((pos1_2-pos1_1)*mu_in_cells)+1)            # in mu, for plotting
        bins_vec2_mu = np.linspace(pos2_1, pos2_2, round((pos2_2-pos2_1)*mu_in_cells)+1)            # in mu, for plotting
        bins_vec1_cells = np.round(bins_vec1_mu * mu_in_cells)           # for pos-binning, in cells
        bins_vec2_cells = np.round(bins_vec2_mu * mu_in_cells)           # round to avoid strange bin overlaps
        try:
            bins_filter = [-np.inf] + list(filter_thickness) + [np.inf]     # for filter, in cells
            slice_filterdim = slice(1, -1, None) # to pick the filtered hist from its last dimension
            filter_nrcells = np.diff(bins_filter[slice_filterdim]).reshape((1, 1, 1, -1))
        except TypeError: # if it's not iterable
            bins_filter = [-1e9, -filter_thickness, filter_thickness, 1e9]  # for filter, in cells
            slice_filterdim = 1 # hist will have 3 entries in last dimension, pick the middle one with this
            filter_nrcells = 2*filter_thickness

        with bp.File(filename) as fh:
            ds = fh[f"/data/{ts}/particles/{species}/"]
            Np = ds["weighting"].shape[0]
            unitmom = ds["momentum/y"].unitSI.value
            bins_p2_pic = bins_p2_SI / unitmom**2

            # divide in chunks of less than 1e8 particles:
            nrchunks = (Np + 1) // 50000000 + 1
            if verbose:
                print(f"will read the {Np} macroparticles in {nrchunks} chunks")
            limits = list(np.linspace(0, Np, nrchunks, endpoint=False, dtype=int))[1:]
            indices = [slice(i, j, None) for (i, j) in zip([None] + limits, limits + [None])]
            hists = 0
        
            for i, index in enumerate(indices):
                # read the data
                weights = ds["weighting"][index]
                N = len(weights)
                pos = np.empty((N, 3))      # to store the data for taking scalar product later
                pos[:, 0] = ds["positionOffset/x"][index]
                pos[:, 1] = ds["positionOffset/y"][index]
                pos[:, 2] = ds["positionOffset/z"][index]
                p2 = (ds["momentum/x"][index]**2 + ds["momentum/y"][index]**2 + ds["momentum/z"][index]**2) / weights**2

                # do binning
                hist_raw, bin_edges = np.histogramdd(
                    (np.dot(pos-center_point, vec1), np.dot(pos-center_point, vec2), p2, np.dot(pos-center_point, vec3)),
                    bins=(bins_vec1_cells, bins_vec2_cells, bins_p2_pic, bins_filter),
                    weights=weights
                )
                hists += hist_raw[:, :, :, slice_filterdim]
                
                if verbose:
                    nrall = np.round(np.sum(hist_raw), 1)
                    nrfilt = np.round(np.sum(hist_raw[:, :, :, slice_filterdim]), )
                    print(f"In {i+1}. chunk: {nrfilt} particles, i.e. {np.round(nrfilt/nrall*100, 1)}% of the chunk, are inside the filter region")
        
        lastdims = (1, ) if slice_filterdim == 1 else (1, 1)
        hist = (hists
                / np.diff(bins_vec1_mu).reshape((-1, 1, *lastdims))
                / np.diff(bins_vec2_mu).reshape((1, -1, *lastdims))
                / (filter_nrcells / mu_in_cells)
                * 1e18     # from mu to m - particles per m^3
                / 1.742e27 # in n_c 
        )
        return hist, bins_vec1_mu, bins_vec2_mu, bins_E

    @staticmethod
    def lineout_n_over_gamma(
                filename, ts=None,
                vec1=[0, 0, 1],
                vec2=[1, 0, 0],
                center_point=[432, 450, 216],
                filter_thickness=15,
                poslim=[-15, 21],
                mu_in_cells = 30,
                species='e_all',
                verbose=True,
                dividebygamma=True,
            ):
        """ filter out a square given by vec1-vec2, return a function/histogram of 
        the density from position along axis vec1 x vec2
        """
        if ts is None: # then try to guess
            ts = filename.replace('.bp', '').split('_')[-1]

        vec3 = np.cross(vec1, vec2)
        
        pos1, pos2 = poslim
        bins_vec3_mu = np.linspace(pos1, pos2, round((pos2-pos1)*mu_in_cells)+1)            # in mu, for plotting
        bins_vec3_cells = np.round(bins_vec3_mu * mu_in_cells)           # round to avoid strange bin overlaps
        bins_filter = np.array([-1e9, -filter_thickness, filter_thickness, 1e9])
        slice_filterdim = 1 # hist will have 3 entries in last dimension, pick the middle one with this
        filter_nrcells = 2*filter_thickness

        m_e = 9.1094e-31    # in SI
        c = 3e8             # in SI
        
        with bp.File(filename) as fh:
            ds = fh[f"/data/{ts}/particles/{species}/"]
            Np = ds["weighting"].shape[0]
            unitmom = ds["momentum/y"].unitSI.value
            m2c2 = (m_e * c / unitmom)**2 # in PIC units

            # divide in chunks of less than 1e8 particles:
            nrchunks = (Np + 1) // 50000000 + 1
            if verbose:
                print(f"will read the {Np} macroparticles in {nrchunks} chunks")
            limits = list(np.linspace(0, Np, nrchunks, endpoint=False, dtype=int))[1:]
            indices = [slice(i, j, None) for (i, j) in zip([None] + limits, limits + [None])]
            hists = 0
        
            for i, index in enumerate(indices):
                # read the data
                weights = ds["weighting"][index]
                N = len(weights)
                pos = np.empty((N, 3))      # to store the data for taking scalar product later
                pos[:, 0] = ds["positionOffset/x"][index]
                pos[:, 1] = ds["positionOffset/y"][index]
                pos[:, 2] = ds["positionOffset/z"][index]
                p2 = (ds["momentum/x"][index]**2 + ds["momentum/y"][index]**2 + ds["momentum/z"][index]**2) / weights**2
                gamma = 1 if not dividebygamma else (1 + p2 / m2c2) ** 0.5

                # do binning
                hist_raw, bin_edges = np.histogramdd(
                    (np.dot(pos-center_point, vec3), np.dot(pos-center_point, vec1), np.dot(pos-center_point, vec2)),
                    bins=(bins_vec3_cells, bins_filter, bins_filter),
                    weights=weights / gamma
                )
                hists += hist_raw[:, slice_filterdim, slice_filterdim]
                
                if verbose:
                    nrall = np.round(np.sum(hist_raw), 1)
                    nrfilt = np.round(np.sum(hist_raw[:, slice_filterdim, slice_filterdim]), )
                    print(f"In {i+1}. chunk: {nrfilt} particles, i.e. {np.round(nrfilt/nrall*100, 1)}% of the chunk, are inside the filter region")
        
        print(hists.shape)
        lastdims = (1, ) if slice_filterdim == 1 else (1, 1)
        hist = (hists
                / np.diff(bins_vec3_mu)
                / (filter_nrcells / mu_in_cells)**2
                * 1e18     # from mu to m - particles per m^3
                / 1.742e27 # in n_c 
        )
        return hist, bins_vec3_mu



class AnalyzePhaseSpace(object):
    @staticmethod
    def posmom_from_particles(
                filename, ts=None,
                vec_ps=[0, 1, 0], 
                center_point=[432, 450, 216], 
                filter_point=None,
                filter_thickness=24,
                bins_pos_mu=None,  
                species="H_all",
                verbose=True,
                mom_or_energy='mom',
            ):
        """ returns phase space position-momentum along arbitrary axis 
        position is projected on vec_ps with respect to center_point
        momentum is projected on vec_ps
        the momentum bins are evenly spaced w.r.t. momentum or the corresponding energy
        returned is the histogram and the arrays of position-bins, momentum-bins and energy-bins (the latter two are synonymic)
        """
        if ts is None: # then try to guess
            ts = filename.replace('.bp', '').split('_')[-1]

        # set the directions for the ps and filtering
        vec_filter1 = np.array([1, 0, 0]) # those two vectors form the boundary of the lineout
        vec_filter2 = np.array([0, 0, 1]) # it is perpendicular to both
        if filter_point is None:
            filter_point = center_point # and goes through that point

        # some unit conversions:
        m_p = 1.6726219e-27 # in SI
        c = 3e8             # in SI
        J2MeV = 6.242e+12
        E_0 = m_p * c**2    # in SI

        ###
        ### compute all bins - we need bins in pic-units for effective binning, and bins for plotting
        ###

        # we can define bins w.r.t. momentum or the energy corresponding to the momentum:
        if mom_or_energy == 'energy':
            E_max_neg_MeV = 80
            E_max_pos_MeV = 150
            bins_Epos = np.linspace(0, E_max_pos_MeV, 401)                 # in MeV
            bins_Eneg = np.linspace(E_max_neg_MeV, 0, 101)                 # the positive/abs values
            bins_Epos_SI = bins_Epos / J2MeV 
            bins_Eneg_SI = bins_Eneg / J2MeV
            bins_ppos_SI =  (((bins_Epos_SI + E_0)**2 - E_0**2))**0.5 / c  # in SI
            bins_pneg_SI = -(((bins_Eneg_SI + E_0)**2 - E_0**2))**0.5 / c  # in SI
            bins_mom_SI = np.array(list(bins_pneg_SI)[:-1] + list(bins_ppos_SI)) # for binning, nearly, need to divide by unit, we get it only later from dataset
            bins_E = list(-bins_Eneg)[:-1] + list(bins_Epos)               # can be used for plotting
            bins_betagamma = bins_mom_SI / m_p / c                         # alternatively those for plotting
        elif mom_or_energy == 'mom':
            bins_betagamma_neg = np.linspace(-0.4, 0., 201)
            bins_betagamma_pos = np.linspace(0., 0.8, 401)
            bins_pneg_SI = bins_betagamma_neg * m_p * c
            bins_ppos_SI = bins_betagamma_pos * m_p * c
            bins_mom_SI = np.array(list(bins_pneg_SI)[:-1] + list(bins_ppos_SI)) # for binning, nearly, need to divide by unit, we get it only later from dataset
            bins_Epos_SI = (bins_ppos_SI**2 * c**2 + E_0**2) ** 0.5 - E_0
            bins_Eneg_SI = -((bins_pneg_SI**2 * c**2 + E_0**2) ** 0.5 - E_0)
            bins_Epos = bins_Epos_SI * J2MeV
            bins_Eneg = bins_Eneg_SI * J2MeV
            bins_E = list(bins_Eneg)[:-1] + list(bins_Epos)               # can be used for plotting
            bins_betagamma = list(bins_betagamma_neg)[:-1] + list(bins_betagamma_pos) # for plotting

        if bins_pos_mu is None:
            bins_pos_mu = np.linspace(-15, 20, 201)                    # in mu, for plotting
        bins_pos_cells = bins_pos_mu * 30                              # for pos-binning, in cells
        bins_filter = [-1e9, -filter_thickness, filter_thickness, 1e9] # for filter, in cells

        with bp.File(filename) as fh:
            ds = fh[f"/data/{ts}/particles/{species}/"]
            Np = ds["weighting"].shape[0]
            unitmom = ds["momentum/y"].unitSI.value
            bins_mom_pic = bins_mom_SI / unitmom

            # divide in chunks of less than 1e8 particles:
            nrchunks = (Np + 1) // 40000000 + 1
            if verbose:
                print(f"will read the {Np} macroparticles in {nrchunks} chunks")
            limits = list(np.linspace(0, Np, nrchunks, endpoint=False, dtype=int))[1:]
            indices = [slice(i, j, None) for (i, j) in zip([None] + limits, limits + [None])]
            hists = []
            for index in indices:
                # read the data
                weights = ds["weighting"][index]
                N = len(weights)
                pos = np.empty((N, 3))      # to store the data for taking scalar product later
                mom = np.empty((N, 3))      # to store the data for taking scalar product later
                pos[:, 0] = ds["positionOffset/x"][index]
                pos[:, 1] = ds["positionOffset/y"][index]
                pos[:, 2] = ds["positionOffset/z"][index]
                mom[:, 0] = ds["momentum/x"][index] / weights
                mom[:, 1] = ds["momentum/y"][index] / weights
                mom[:, 2] = ds["momentum/z"][index] / weights

                # compute quantities
                ps_pos = np.dot(pos - center_point, vec_ps)
                ps_mom = np.dot(mom, vec_ps)
                scalar1 = np.dot(pos - filter_point, vec_filter1)
                scalar2 = np.dot(pos - filter_point, vec_filter2)

                # do binning
                hist_raw, bin_edges = np.histogramdd(
                    (ps_pos, ps_mom, scalar1, scalar2),
                    bins=(bins_pos_cells, bins_mom_pic, bins_filter, bins_filter),
                    weights=weights
                )
                hists.append(hist_raw[:, :, 1, 1])
                if verbose:
                    nrall = np.round(np.sum(hist_raw), 1)
                    nrfilt = np.round(np.sum(hist_raw[:, :, 1, 1]), )
                    nrchunk = len(hists)
                    print(f"In {nrchunk}. chunk: {nrfilt} particles, i.e. {np.round(nrfilt/nrall*100, 1)}% of the chunk, are inside the filter region")


        hist = sum(hists)
        return hist, bins_pos_mu, bins_betagamma, bins_E


def energy_cutoff_from_hist(filename, threshold=0, retlastspec=False):
    """ return the timeline and max-value of cutoff energy """
    with open(filename, 'r') as f:
        zeilen = f.readlines()

    energies = [0] + list(map(float, zeilen[0].split()[2:-2]))
    lastline = len(zeilen)-2
    timeline = []
    # fill in the data, one dict per step
    for nrzeile in range(1, lastline+1):
        bins = np.array(list(map(float, zeilen[nrzeile].split()[1:-2])))
        thisdata = {}
        timeline.append(thisdata)
        step = int(zeilen[nrzeile].split()[0])
        thisdata['step'] = step
        thisdata['linenr'] = nrzeile
        thisdata['emax'] = energies[np.where(bins > threshold)[0][-1]] / 1000 # in MeV
        if retlastspec and nrzeile == lastline:
            lastspec = {
                "step": step,
                "linenr": nrzeile,
                "bins": bins,
                "energies": np.array(energies) / 1000,
            }

    # now compute overall maximum energies
    tuples = [(r['emax'], r['step']) for r in timeline]
    value, timestep = max(tuples)
    return (tuples, value, timestep) if not retlastspec else (tuples, value, timestep, lastspec) 

def energies_from_hist(filename, threshold=1., convolve=([0.2, 0.6, 0.2], 1)):
    """ return timeline of spectra and max-energies """
    with open(filename, 'r') as f:
        zeile = f.readline()

    energies = np.array([0] + list(map(float, zeile.split()[2:-2]))) / 1000
    arr = np.loadtxt(filename, skiprows=1)
    steps = np.array(arr[:, 0], dtype=int)
    indices = {ts: i for i, ts in enumerate(steps)}
    liste, anz = convolve
    arr = arr[:, 2:-2]
    
    def last_bin(ts=steps[-1], threshold=threshold):
        zeile = np.convolve(arr[indices[ts]], liste)[anz:-anz]
        where = np.where(zeile > threshold)[0]
        if not len(where)>0:
            return energies[0]
        return energies[where[-1]+anz]
        
    def first_gap(ts=steps[-1], threshold=threshold):
        zeile = np.convolve(arr[indices[ts]], liste)[anz:-anz]
        where = np.where(zeile <= threshold)[0]
        if not len(where)>0:
            return energies[-1]
        else:  
            return energies[where[0]+anz]
    
    return energies, steps, arr, last_bin, first_gap


def angular_spectrum_horizontplane_from_calo(filename, ts=None):
    if ts is None: # then try to guess
        ts = filename.split('_')[-4]
    
    with h5py.File(filename, "r") as fh:
        ds = fh[f"/data/{ts}/calorimeter/"]

        # helper block for accessing the attributes of the dataset
        attrnames = dict(
            Emin="minEnergy[keV]", 
            Emax="maxEnergy[keV]", 
            Elog="logScale", 
            pmax="maxPitch[deg]", 
            ymax="maxYaw[deg]", 
            ppos="posPitch[deg]", 
            ypos="posYaw[deg]", 
            unit="unitSI"
        )
        for var, name in attrnames.items():
            exec(f'{var} = ds.attrs["{name}"]')

        Enum, pnum, ynum = ds.shape

        # check if the assumptions are fulfilled
        if ypos or ppos:
            raise ValueError("I expect the calorimeter to be centered, but it's not; some analysis may not make sense")
        if pmax > 15:
            raise ValueError(f"I am going to sum over all Pitch values, the maximal pitch of {pmax} is too big for this to make sense")

        # create the arrays for the energy- and yaw-coordinates
        binsy = np.linspace(-ymax, ymax, ynum+1)
        if Elog:
            binsE = np.logspace(np.log10(Emin), np.log10(Emax), Enum+1)
        else:
            binsE = np.linspace(Emin, Emax, Enum+1)
        
        data = np.sum(ds, axis=1).T

    plt.pcolormesh(binsE, binsy, data, norm=LogNorm())


def slice_z(filename, ts=None, slicewidth=4):
    """ returns 2d-array of the central slice """
    filename = f'{params.folder.folder}{params.outputs.fields.filename.format(timestep=timestep)}'
    self.log(f"reading {filename}")
    with bp.File(filename) as fh:
        field = fh[f'/data/{timestep}/fields/{fieldname}']
        zstart, zend = params.pos_to_cell(-slicewidth/2, axis=0), params.pos_to_cell(slicewidth/2, axis=0)
        self.log(f"doing average in z-direction over {zend-zstart} cells, {zstart}-{zend}")
        rawdata = np.mean(field[zstart:zend, :, :], axis=0) # slice in z
        unit_factor = field.attrs['unitSI'].value / normfactor
        extent=[params.pos_from_cell(*args) for args in [(0, ), (rawdata.shape[0]-1, ), (0, 2), (rawdata.shape[1]-1, 2)]]
        data = unit_factor * rawdata.transpose()
        image = axes.imshow(
            data,
            origin='upper',
            aspect=1.,
            extent=extent,
            norm=plotnorm,
        )
        self.log(f"drawn axes with {whichfield}")

        if cbar_ax:
            cbar_ax.figure.colorbar(image, cax=cbar_ax)

        if title: 
            axes.set_title(title.format(time=time))
        
        return data



def angular_spectrum_conic_from_particles(filename, ts=None, Emax=None):
    if ts is None: # then try to guess
        ts = filename.replace('.bp', '').split('_')[-1]
    if Emax is None:
        Emax = 160
    
    # create the arrays for the energy- and angle-coordinates
    # definitions:
    bins_E = np.linspace(0, Emax, 51)   # Emax in MeV
    bins_theta = np.linspace(0, 80, 81) # in deg
    bins_py = [0., 1e9]                 # one bin - filter out backward

    # unit conversions:
    m_p = 1.6726219e-27 # in SI
    c = 3e8 # in SI
    J2MeV = 6.242e+12

    # new bins
    E_0 = m_p * c**2                    # in SI
    bins_E_SI = bins_E / J2MeV          # in SI
    bins_p2_SI = ((bins_E_SI + E_0)**2 - E_0**2) / c**2 # in SI
    bins_tan2 = np.tan(bins_theta / 180 * np.pi) ** 2

    diffs_E = np.diff(bins_E)
    values_E = (bins_E[1:] + bins_E[:-1]) / 2
    diffs_theta = np.diff(bins_theta) / 180 * np.pi
    values_theta = (bins_theta[:-1] + bins_theta[1:]) / 360 * np.pi
    diffs_sterad = 2*np.pi*np.sin(values_theta) * diffs_theta
    
    with bp.File(filename) as fh:
        px = fh[f'/data/{ts}/particles/H_all/momentum/x'][:]
        py = fh[f'/data/{ts}/particles/H_all/momentum/y'][:]
        pz = fh[f'/data/{ts}/particles/H_all/momentum/z'][:]
        un = fh[f'/data/{ts}/particles/H_all/momentum/y'].unitSI.value
        we = fh[f'/data/{ts}/particles/H_all/weighting'][:]

        py1 = py / we
        py2 = py**2 / we**2
        px2 = px**2 / we**2
        pz2 = pz**2 / we**2
        pt2 = px2 + pz2

        bins_p2 = bins_p2_SI / un**2
        hist, bin_edges = np.histogramdd(
            (pt2 + py2, pt2 / py2, py1),
            bins=(bins_p2, bins_tan2, bins_py),
            weights=we
        )
        
    # norm and remove the last dimension of length 1
    hist = hist[:, :, 0] / (            # norm to particles per MeV and sterad
        diffs_E.reshape((-1, 1))        # per MeV
        * diffs_sterad.reshape((1, -1)) # per sterad in cyllinder symmetry
    )
    bin_edges = bin_edges[:-1]

    return hist, bins_E, bins_theta
    
