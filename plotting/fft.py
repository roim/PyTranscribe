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

""" Contains functions to plot FFTs. """

import matplotlib.pyplot as _pl
import numpy as _np

import soundfiles as _sf


""" Plot the FFT for up to 3 given audio file paths. """
def plotfft(audiopath, audiopath2="", audiopath3="", binsize=44100, plotpath=None):
    samplerate, samples = _sf.readfile(audiopath)

    # Merge multiple channels
    if hasattr(samples[0], "__len__"):
        samples = _np.mean(samples, 1)

    samples = samples[0:binsize]

    X = _np.fft.fft(samples, binsize)

    # amplitude to decibel
    dBX = 20.*_np.log10(_np.abs(X)/10e-6) - 120

    # remove mirror
    dBX = dBX[0:dBX.size/2]

    pl.figure(figsize=(15, 7.5))
    pl.plot(dBX, 'b')

    if audiopath2 != "":
        # Yes, I'm lazy and just copy pasted
        samplerate2, samples2 = _sf.readfile(audiopath2)

        # Merge multiple channels
        if hasattr(samples2[0], "__len__"):
            samples2 = _np.mean(samples2, 1)

        samples2 = samples2[0:binsize]

        X2 = _np.fft.fft(samples2, binsize)

        # amplitude to decibel
        dBX2 = 20.*_np.log10(_np.abs(X2)/10e-6) - 120

        # remove mirror
        dBX2 = dBX2[0:dBX2.size/2]

        pl.plot(dBX2, 'g')

    if audiopath3 != "":
        # Yes, I'm lazy and just copy pasted
        samplerate3, samples3 = _sf.readfile(audiopath3)

        # Merge multiple channels
        if hasattr(samples3[0], "__len__"):
            samples3 = _np.mean(samples3, 1)

        samples3 = samples3[0:binsize]

        X3 = _np.fft.fft(samples3, binsize)

        # amplitude to decibel
        dBX3 = 20.*_np.log10(_np.abs(X3)/10e-6) - 120

        # remove mirror
        dBX3 = dBX3[0:dBX3.size/2]

        pl.plot(dBX3, 'r')

    pl.xlabel("Frequency (Hz)")
    pl.ylabel("Amplitude (dB)")
    pl.xlim([0, binsize])
    pl.ylim([0, _np.max(dBX)])

    # Use the highest index as the reference.
    # We assume the highest index corresponds to the fundamental.
    reference = _np.argmax(dBX if audiopath2 == "" else dBX2)
    xlocs = _np.float32([n * reference for n in range(0, 50)])
    pl.xticks(xlocs, ["%.0f" % l for l in xlocs])

    if plotpath:
        pl.savefig(plotpath, bbox_inches="tight")
    else:
        pl.show()

    pl.clf()
