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

""" Functions for pitch detection via Harmonic Product Spectrum (HPS). """

import numpy as _np
import scipy.signal as _sig

import mtheory as _mt


def hps(x, fs=44100, lf=255, harmonics=3, precision=2, window=lambda l:_np.kaiser(l, 7.14285)):
    """ Estimates the pitch (fundamental frequency) of the given sample array by a standard HPS implementation. """
    x -= _np.mean(x)
    N = x.size
    w = x*window(N)

    # Append zeros to the end of the window so that each bin has at least the desired precision.
    if fs/N > precision:
        delta = int(fs/precision) - N
        w = _np.append(w, _np.zeros(delta))
        N = w.size

    X = _np.log(_np.abs(_np.fft.rfft(w)))

    # Sequentially decimate 'X' 'harmonics' times and add it to itself.
    # 'precision < fs/N' must hold, lest the decimation loses all the precision we'd gain.
    hps = _np.copy(X)
    for h in range(2, 2 + harmonics):
        dec = _sig.decimate(X, h)
        hps[:dec.size] += dec*(0.9**h)

    # Find the bin corresponding to the lowest detectable frequency.
    lb = lf*N/fs

    # And then the bin with the highest spectral content.
    arg_peak = lb + _np.argmax(hps[lb:dec.size])

    # TODO: Return the full array? A ranked list of identified notes?
    return fs*arg_peak/N


def tunedhps(x, fs=44100, lf=255, harmonics=3, precision=1, window=lambda x:_np.kaiser(x, 7.14285)):
    """ Estimates the pitch (fundamental frequency) of the given sample array by an HPS implementation that evaluates
    the spectrum only in tuned note frequencies (e.g. frequencies of notes in an assumed tuning). """
    x -= _np.mean(x)
    N = x.size
    w = x*window(N)

    if fs/N > precision:
        delta = int(fs/precision) - N
        w = _np.append(w, _np.zeros(delta))
        N = w.size

    frequencies = [f for f in _mt.notes if f >= lf and f < fs/(2*harmonics)]

    X = _np.log(_np.abs(_np.fft.rfft(w)))
    Y = _np.ones(len(frequencies))
    for i in range(0, len(frequencies)):
        for h in range(1, harmonics+1):
            f_idx = int(round(frequencies[i]*h/2)*2*N/fs)
            Y[i] += X[f_idx]*(0.9**(h-1))

    arg_peak = _np.argmax(Y)
    return frequencies[arg_peak]
