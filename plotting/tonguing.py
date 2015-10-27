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

""" Contains functions to plot graphs related to tonguing detection. """

import matplotlib.pyplot as _pl
import numpy as _np

import soundfiles as _sf
import tonguing as _tong


def plot_tonguing(audiopath, title="", duration=3, plotpath=None):
    """ Plots a visual representation of the tonguing detection algorithm. """
    samplerate, samples = _sf.readfile(audiopath)

    if samples.size/samplerate < 3:
        raise Exception("Input too short")

    samples = samples[0:samplerate*duration]
    envelope = _tong._envelope(samples)
    smooth = _tong._exponential_smoothing(envelope, x_s0=_np.mean(samples[0:50]))

    f, (ax0, ax1, ax2, ax3) = _pl.subplots(4, sharex=True)

    ax0.plot(samples)
    ax1.plot(_np.abs(samples))
    ax2.plot(envelope)
    ax3.plot(smooth)

    ax0.set_title(title)

    xlocs = _np.float32([samplerate*i/2 for i in range(2*duration + 1)])
    ax3.set_xlabel("Time (s)")
    ax3.set_xlim([0, _np.max(xlocs)])
    ax3.set_xticks(xlocs)
    ax3.set_xticklabels(["%.2f" % (l/samplerate) for l in xlocs])

    ax0.set_ylabel("Signal")
    ax1.set_ylabel("Signal (Absolute)")
    ax2.set_ylabel("Hilbert Envelope")
    ax3.set_ylabel("Smoothed Envelope")

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()

    _pl.clf()
    return


def plot_amplitude(audiopath, title="", duration=3, plotpath=None):
    """ Plots the amplitude of an audio signal over time. """
    samplerate, samples = _sf.readfile(audiopath)

    if samples.size/samplerate < 3:
        raise Exception("Input too short")

    samples = samples[0:samplerate*duration]

    _pl.figure(figsize=(10, 3))
    _pl.plot(samples)
    _pl.title(title)

    xlocs = _np.float32([samplerate*i/2 for i in range(2*duration + 1)])
    _pl.xlabel("Time (s)")
    _pl.xlim([0, _np.max(xlocs)])
    _pl.xticks(xlocs, ["%.2f" % (l/samplerate) for l in xlocs])

    _pl.ylabel("Amplitude")

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()

    _pl.clf()
    return
