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

""" Contains functions to plot graphs related to the Harmonic Product Spectrum. """

import matplotlib.pyplot as _pl
import numpy as _np
import scipy.signal as _sig

import mathhelper as _mh
import mtheory as _mt
import pda.hps as _hps
import soundfiles as _sf


def plothps(audiopath, title="Harmonic Product Spectrum", horizontal_harmonics=7, plotpath=None):
    """ Plots a visual representation of the HPS with 3 harmonics. """
    samplerate, samples = _sf.readfile(audiopath)

    X = _np.fft.fft(samples, samplerate)

    # amplitude to decibel
    dBX = 20.*_np.log10(_np.abs(X)/10e-6) - 120

    # remove mirror
    dBX = dBX[0:dBX.size/2]

    f, (ax0, ax1, ax2, ax3) = _pl.subplots(4, sharex=True, sharey=True)
    axs = (ax0, ax1, ax2, ax3)

    sum = _np.zeros_like(dBX)
    for i in range(3):
        dec = _sig.decimate(dBX, i + 1)
        sum[:dec.size] += dec
        axs[i].plot(dec, 'b')

    sum = _np.divide(sum, 3)
    ax3.plot(sum, 'b')

    ax0.set_title(title)

    reference = _np.argmax(sum)
    xlocs = _np.float32([n * reference for n in range(1 + horizontal_harmonics)])
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_xlim([0, _np.max(xlocs)])
    ax3.set_xticks(xlocs)
    ax3.set_xticklabels(["%.0f" % l for l in xlocs])

    ax0.set_ylabel("Amplitude (dB)")
    ax1.set_ylabel("Decimated by 2")
    ax2.set_ylabel("Decimated by 3")
    ax3.set_ylabel("Mean")
    ax3.set_ylim([40, 1.15*_np.max(sum)])

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()

    _pl.clf()


def plot_tracking(audiopath, title="", binsize=1470, tune=False, plotpath=None, repetitions=10):
    """ Plots the HPS tracking of an audio file. """
    samplerate, samples = _sf.readfile(audiopath)

    detections = samples.size//binsize

    p = _np.zeros(repetitions*detections)
    for i in range(detections):
        f = _hps.hps(samples[i*binsize:(i+1)*binsize])

        if tune:
            f = _mh.find_nearest_value(_mt.notes, f)

    p = _np.repeat(p, repetitions)

    _pl.plot(p)
    _pl.title(title)

    xlocs = _np.linspace(0, 10*detections, 5)
    _pl.xlabel("Time (s)")
    _pl.xlim([0, _np.max(xlocs)])
    _pl.xticks(xlocs, ["%.2f" % l for l in _np.multiply(xlocs, binsize/(repetitions*samplerate))])

    _pl.ylabel("Fundamental Frequency (Hz)")
    _pl.ylim((0.9*_np.min(p), 1.05*_np.max(p)))

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()

    _pl.clf()
