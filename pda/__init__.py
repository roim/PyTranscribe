# Copyright 2015 Rodrigo Roim Ferreira
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

""" Module containing Pitch Detection Algorithms (PDAs). """

__all__ = ['hps', 'hwt']


import numpy as _np


_log2_500 = _np.log2(500)
_log2_3000 = _np.log2(3000)
def ear_response_rfft(x, fs=44100):
    """ The human ear attenuates certain frequencies. 
        This function produces an RFFT that mimics the human ear behavior.
        The return value is in decibels of the absolute RFFT values. """
    x = _np.asarray(x)
    N = x.size

    # RFFT of the absolute values in decibels
    X = 20*_np.log10(_np.abs(_np.fft.rfft(x)))

    # Reduce 12dB/octave below 500Hz and above 3000Hz
    for i in range(0, X.size):
        log2_f = _np.log2(i*fs/N)
        if log2_f < _log2_500:
            X[i] -= 12*(_log2_500 - log2_f)
        elif log2_f > _log2_3000:
            X[i] -= 12*(log2_f - _log2_3000)

    return X


def bin_frequency(index, binSize, fs=44100):
    """ Returns the frequency for a given FFT bin. """
    return index*fs/binSize


def bin_for_frequency(f, binSize, fs=44100):
    """ Returns the bin for a given FFT frequency. """
    fLow = (f*binSize//fs)*fs/binSize
    fHigh = (1 + f*binSize//fs)*fs/binSize

    if _np.abs(f - fLow) < _np.abs(f - fHigh):
        return f*binSize//fs

    return 1 + f*binSize//fs


def note_bins(note_array, binSize, fs=44100):
    """ Given an array with note frequencies and a bin size, return an array
        with the bin most closely related to the note at the corresponding index
        e.g. note_array [ 40, 60, 100 ]
             produces   [  1,  2,   4 ] """
    return [bin_for_frequency(f, binSize, fs) for f in note_array]
