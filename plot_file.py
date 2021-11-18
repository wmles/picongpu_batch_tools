#!/usr/bin/env python3

""" Functions to produce plots from single PIConGPU output files """

import adios as bp
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm

"""
TODO: documentation
TODO: extract all common stuff into decorator
"""




def ax_and_cbar(polar=False, retfig=False):
    """ standard figure with one normal axes and one small colorbar-axes """
    fig = plt.figure(constrained_layout=False, figsize=(12, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[15,1])

    if polar:
        ax1 = plt.subplot(gs[0], projection="polar", aspect=1.)
    else:
        ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    return (ax1, ax2) if not retfig else (ax1, ax2, fig)


class PlotAngularDistributions(object):
    """ collects methods for visualizing the angular distributions and spectra """
    @staticmethod
    def spectrum_conic(hist, bins_E, bins_theta, bins_phi, axandfig=None, plotkwargs={}):
        """ angular spectrum supposing cylinder symmetry, i.e. the
        distribution from energy and theta
        expects same input as the polar plots routines
        """
        if axandfig is None: # construct axes
            ax1, ax2, fig = ax_and_cbar(retfig=True)
        else:
            ax1, ax2, fig = axandfig
                    
        im = ax1.pcolormesh(bins_E, bins_theta, np.sum(hist, axis=2).T, norm=LogNorm(), **plotkwargs)
        ax1.set_ylabel("angle to laser propagation, in °")
        ax1.set_xlabel("Energy of $H^+$ in MeV")
        plt.colorbar(im, cax=ax2)

        return ax1, ax2, fig

#    @staticmethod
#    def angular_spectrum(hist, bins_E, bins_theta, axlist=None, plotkwargs={}):
#        """ plot angular spectrum, expects as input the output of 
#        analyze_file.angular_spectrum_*
#        """
#        if axlist is None: # construct axes
#            axlist = ax_and_cbar()
#
#        ax1, ax2 = axlist
#        
#        im = ax1.pcolormesh(bins_E, bins_theta, hist.T, norm=LogNorm(), **plotkwargs)
#        #plt.title("Angular spectra of protons, 120fs after peak")
#        ax1.set_ylabel("angle to laser propagation, in °")
#        ax1.set_xlabel("Energy of $H^+$ in MeV")
#        plt.colorbar(im, cax=ax2)
#
#        return ax1, ax2

    @staticmethod
    def rectangle(hist, bins_thetax, bins_thetaz, axlist=None):
        """ plot angular distribution in a 'rectangle' representation, i.e. the input 
        is similar to the output of analyze_file.AnalyzeAngularDistribution.rectangle_*
        but summed over energy bins.
        """
        if axlist is None: # construct axes
            axlist = ax_and_cbar()
        
        ax1, ax2 = axlist

        im = ax1.pcolormesh(bins_thetax, bins_thetaz, hist.T, norm=LogNorm())
        ax1.set_title("Angular distribution of protons")
        ax1.set_xlabel("angle 'in x-direction' to laser propagation, in °")
        ax1.set_ylabel("angle 'in z-direction' to laser propagation, in °")
        plt.colorbar(im, cax=ax2)

        return ax1, ax2


    
    @staticmethod
    def polar_default(hist, bins_E, bins_theta, bins_phi, indices, axandfig=None):
        """ computes histogram of (E, theta, phi) 
        plots several E-"slices" of this distribution """
        # decide which groups in E-direction to make
        if not indices:
            indices = PlotAngularDistributions.find_indices(hist)

        if axandfig is None: # construct axes
            ax1, ax2, fig = ax_and_cbar(retfig=True)
        else:
            ax1, ax2, fig = axandfig

        NE, Nt, Np = hist.shape
        plotdata = np.empty((Nt, Np))
        circleparts = [ # inner parts of the circle are averaged over more phi-bins
            (np.s_[ 0: 1, :], 180), # theta-indices, nr of phi bins to average over
            (np.s_[ 1: 3, :], 15),
            (np.s_[ 3: 8, :], 9),
            (np.s_[ 8:16, :], 3),
            (np.s_[16:-1, :], 1), 
        ]

        # find maximal E-index that contains any nonzero elements
        # should not take higher, since the colornorm breaks then

        for ind in indices:
            Eindex = slice(ind, None, None)
            Evalue = int(round(bins_E[ind]))

            for thetaind, averagenr in circleparts:
                data = np.sum(hist[Eindex, :, :], axis=0) # filter and sum E
                data = data[thetaind]
                numtheta, numphi = data.shape
                plotdata[thetaind] = np.mean(data.reshape((numtheta, -1, averagenr)), axis=-1).reshape(numtheta, -1, 1).repeat(averagenr, axis=-1).reshape((numtheta, -1))

            ax1.cla()
            ax2.cla()
            try:
                im = ax1.pcolormesh(bins_phi, bins_theta[:-1], plotdata[:-1, :], norm=LogNorm())
                plt.colorbar(im, cax=ax2)
                ax1.set_title(f"$H^+$ with Energy over {Evalue} MeV")
            except ValueError:
                Evalue = None
            yield Evalue, ax1, ax2, fig

    # war ein Versuch, die indices zu suchen; ist bissl Quatsch, weil das nicht der Anteil der Teilchen insgesamt, sondern nur an den im Histogramm vorhandenen
    @staticmethod
    def find_indices(hist, targetfractions=[0.001, 0.1]):
        sumfromE = np.cumsum(np.sum(hist, axis=(1, 2))[::-1]) # the amount of particles with higher energy
        fractions = sumfromE / sumfromE[-1]
        indandfrac = []
        for fract in targetfractions:
            for i, frac in enumerate(fractions):
                if frac >= fract:
                    indandfrac.append((i, frac))
                    break

        indandfrac = dict(indandfrac) # removes duplicates, if two targets reached at once
        indices = len(sumfromE) - 1 - np.array(sorted(indandfrac.keys()))
        return indices


    @staticmethod
    def angular_distribution_polar(hist, bins_theta, bins_phi, axlist=None):
        """ plot angular distribution in a polar representation, i.e. with respect
        to the sperical angles theta and phi. The input is similar to the output of
        analyze_file.AnalyzeAngularDistribution.rectangle_* but summed over E-bins.
        """
        if axlist is None: # construct axes
            axlist = ax_and_cbar(polar=True)
        
        ax1, ax2 = axlist

        im = ax1.pcolormesh(bins_thetax, bins_thetaz, hist.T, norm=LogNorm())
        ax1.set_title("Angular distribution of protons")
        ax1.set_xlabel("angle 'in x-direction' to laser propagation, in °")
        ax1.set_ylabel("angle 'in z-direction' to laser propagation, in °")
        plt.colorbar(im, cax=ax2)

        return ax1, ax2

    
    @staticmethod
    def angular_distributions_from_energy(hist, bins_E, bins_thetax, bins_thetaz, Elims=[0, None], axlist=None):
        """ plot energy-filtered angular distribution; takes as input the output of 
        analyze_file.angular_distribution_*
        """

        if axlist is None: # construct axes
            axlist = ax_and_cbar()
        
        ax1, ax2 = axlist

        # find the indices of energy bins to use for the filtering/summation
        Emin, Emax = Elims
        if Emin in bins_E:
            indmin = np.where(bins_E == Emin)[0][0]
        elif Emin < min(bins_E):
            raise ValueError("Emin may not be smaller than the minimal energy bin")
        else:
            indmin = np.searchsorted(bins_E, Emin) - 1
        if Emax in bins_E:
            indmax = np.where(bins_E == Emax)[0][0]
        elif Emax in [None, -1]: # special cases
            indmax = -1
        elif Emax > max(bins_E):
            raise ValueError("Emax may not be greater than the maximal energy bin")
        else:
            indmax = np.searchsorted(bins_E, Emax)

        values = np.sum(hist[indmin:indmax, :, :], axis=0)

        ax1, ax2 = PlotAngularDistributions.angular_distribution_rectangle(
            values, bins_thetax, bins_thetaz, axlist=[ax1, ax2]
        )
        ax1.set_title(f"Angular distribution of $H^+$ between {bins_E[indmin]} and {bins_E[indmax]}")

        return ax1, ax2











