# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
from __future__ import (absolute_import, division, print_function)
import AbinsModules
import numpy as np


# noinspection PyPep8Naming
class Instrument(object):

    _name = None

    def calculate_q_powder(self, input_data=None):
        """
        Calculates Q2.


        :param  input_data: data from which Q2 should be calculated
        :returns:  numpy array with Q2 data
        """

        return None

    def convolve_with_resolution_function(self, frequencies=None, s_dft=None):
        """
        Convolves discrete spectrum with the  resolution function for the particular instrument.

        :param frequencies: frequencies for which resolution function should be calculated (frequencies in cm-1)
        :param s_dft:  discrete S calculated directly from DFT

       """
        return None

    @staticmethod
    def _broaden_spectrum(frequencies=None, bins=None, s_dft=None, sigma=None, scheme='legacy_plus', prebin='auto'):
        """Convert frequency/S data to broadened spectrum on a regular grid

        Several algorithms are implemented, for purposes of
        performance/accuracy optimisation and demonstration of legacy
        behaviour.

        :param frequencies: input frequency series; these may be irregularly spaced
        :type frequencies: 1D array-like
        :param bins: Evenly-spaced frequency bin values for the output spectrum. If *frequencies* is None, s_dft is assumed to correspond to mid-point frequencies on this mesh.
        :type bins: 1D array-like
        :param s_dft: scattering values corresponding to *frequencies*
        :type s_dft: 1D array-like
        :param sigma: width of broadening function. This may be a scalar used over the whole spectrum, or a series of values corresponding to *frequencies*.
        :type sigma: float or 1D array-like
        :param scheme:
            Name of broadening method used. Options:

            - legacy_plus: Implementation of the legacy Abins method (sum over
                  Gaussian kernels with fixed number of samples). Not
                  recommended for production calculations due to aliasing on
                  non-commensurate output grid.
        :type scheme: str
        :returns: (freq_points, broadened_spectrum)

        The *freq_points* are the mid-bin frequency values which
        nominally correspond to *broadened_spectrum*; both of these are
        1D arrays of length (N-1) corresponding to the length-N *bins*.
        """
        if (bins is None) or (s_dft is None) or (sigma is None):
            raise ValueError("Frequency bins, S data and broadening width must be provided.")

        if scheme not in ('legacy_plus'):
            raise ValueError('Broadening scheme "{}" not supported for this instrument, please correct '
                             'AbinsParameters.sampling["broadening_scheme"]'.format(scheme))

        freq_points = (bins[1:] + bins[:-1]) / 2

        #  Don't bother broadening if there is nothing here to broaden: return empty spectrum
        if (len(s_dft) == 0) or not np.any(s_dft):
            return freq_points, np.zeros_like(freq_points)

        #  Allow histogram data to be input as bins + s_dft; use the mid-bin values
        elif frequencies is None:
            if len(s_dft) == len(bins) - 1:
                frequencies = freq_points
            else:
                raise ValueError("Cannot determine frequency values for s_dft before broadening")

        if scheme == 'legacy_plus':
            fwhm = AbinsModules.AbinsParameters.instruments['fwhm']
            pkt_per_peak = AbinsModules.AbinsParameters.sampling['pkt_per_peak']
            def multi_linspace(start, stop, npts):
                """Given 1D length-N start, stop and scalar npts, get (npts, N) linspace-like matrix"""
                ncols = len(start)
                step = (stop - start) / (npts - 1)
                step_matrix = step.reshape((ncols, 1)) * np.ones((1, npts))
                start_matrix = start.reshape((ncols, 1)) * np.ones((1, npts))
                return (np.arange(npts) * step_matrix) + start_matrix
            broad_freq = multi_linspace(frequencies - fwhm * sigma,
                                        frequencies + fwhm * sigma,
                                        pkt_per_peak)
            ncols = len(frequencies)
            if np.asarray(sigma).shape == ():
                sigma = np.full(ncols, sigma)
            sigma_matrix = sigma.reshape(ncols, 1) * np.ones((1, pkt_per_peak))
            freq_matrix = frequencies.reshape(ncols, 1) * np.ones((1, pkt_per_peak))
            s_dft_matrix = s_dft.reshape(ncols, 1) * np.ones((1, pkt_per_peak))
            dirac_val = Instrument._gaussian(sigma=sigma_matrix, points=broad_freq, center=freq_matrix)

            flattened_spectrum = np.ravel(s_dft_matrix * dirac_val)
            flattened_freq = np.ravel(broad_freq)

            hist, bin_edges = np.histogram(flattened_freq, bins, weights=flattened_spectrum, density=False)
            return freq_points, hist

    @staticmethod
    def _convolve_with_resolution_function_legacy(frequencies=None, s_dft=None, sigma=None, pkt_per_peak=None,
                                                  gaussian=None):
        """Legacy implementation of gaussian convolution. 

        This method uses a kernel that is not commensurate with the
        sampling bins and is not recommended for production
        calculations.

        :param frequencies: discrete frequencies in cm^-1
        :type frequencies: numpy.ndarray
        :param s_dft: discrete values of S
        :type s_dft: numpy.ndarray
        :param sigma: resolution function width (or array of widths corresponding to each frequency)
        :type sigma: float
        :param pkt_per_peak: number of points per peak
        :type pkt_per_peak: int
        :param gaussian: 
            Gaussian-like function used to broaden peaks; this should take keyword arguments *sigma*, *points*, *center* and return an array of 
            broadening kernels with widths *sigma* evaluated at *points*
        :type gaussian: function
        :returns: frequencies for which peaks have been broadened, corresponding S
        """
        fwhm = AbinsModules.AbinsParameters.instruments['fwhm']

        # freq_num: number of transition energies for the given quantum order event
        # start[freq_num]
        start = frequencies - fwhm * sigma

        # stop[freq_num]
        stop = frequencies + fwhm * sigma

        # step[freq_num]
        step = (stop - start) / (pkt_per_peak - 1)

        # matrix_step[freq_num, pkt_per_peak]
        matrix_step = np.array([step, ] * pkt_per_peak).transpose()

        # matrix_start[freq_num, pkt_per_peak]
        matrix_start = np.array([start, ] * pkt_per_peak).transpose()

        # broad_freq[freq_num, pkt_per_peak]
        broad_freq = (np.arange(0, pkt_per_peak) * matrix_step) + matrix_start
        broad_freq[..., -1] = stop

        # sigma_matrix[freq_num, pkt_per_peak]
        sigma_matrix = np.array([sigma, ] * pkt_per_peak).transpose()

        # frequencies_matrix[freq_num, pkt_per_peak]
        frequencies_matrix = np.array([frequencies, ] * pkt_per_peak).transpose()

        # gaussian_val[freq_num, pkt_per_peak]
        dirac_val = gaussian(sigma=sigma_matrix, points=broad_freq, center=frequencies_matrix)
        
        # s_dft_matrix[freq_num, pkt_per_peak]
        s_dft_matrix = np.array([s_dft, ] * pkt_per_peak).transpose()

        # temp_spectrum[freq_num, pkt_per_peak]
        temp_spectrum = s_dft_matrix * dirac_val

        points_freq = np.ravel(broad_freq)
        broadened_spectrum = np.ravel(temp_spectrum)

        return points_freq, broadened_spectrum

    @staticmethod
    def _gaussian(sigma=None, points=None, center=None):
        """
        :param sigma: sigma defines width of Gaussian
        :param points: points for which Gaussian should be evaluated
        :param center: center of Gaussian
        :returns: numpy array with calculated Gaussian values
        """
        two_sigma = 2.0 * sigma
        sigma_factor = two_sigma * sigma
        norm = (AbinsModules.AbinsParameters.sampling['pkt_per_peak']
                / (AbinsModules.AbinsParameters.instruments['fwhm'] * two_sigma))
        return np.exp(-(points - center) ** 2 / sigma_factor) / (np.sqrt(sigma_factor * np.pi) * norm)

    def __str__(self):
        return self._name

    def get_name(self):
        return self._name
