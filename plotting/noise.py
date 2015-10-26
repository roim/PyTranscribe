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

""" Contains functions to plot noise graphs. """

import matplotlib.pyplot as _pl
import numpy as _np

import soundfiles as _sf


def plot_noise(audiopath, windowsize=1470, title="", plotpath=None):
    """ Too hard to explain, just call it and see what happens, or read the code. """
    samplerate, samples = _sf.readfile(audiopath)

    if samples.size < 4*samplerate:
        raise Exception("Input is too short.")

    samples = samples[0:windowsize + 4*samplerate]

    windows = samples.size//windowsize

    rms = _np.zeros(windows)
    for i in range(windows):
        w = samples[i*windowsize:(i+1)*windowsize]
        rms[i] = _np.sqrt(_np.mean(_np.square(w)))

    first3seconds = rms[0:(3*samplerate//windowsize)]
    first3seconds.sort()
    pct98 = first3seconds[int(0.98*first3seconds.size)]

    a_pct98 = _np.repeat(pct98, rms.size)
    a_noise = _np.repeat(1.1*pct98, rms.size)

    _pl.figure(figsize=(10, 3))
    _pl.title(title)
    _pl.plot(rms, 'r')
    _pl.plot(a_pct98, 'g')
    _pl.plot(a_noise, 'b')

    _pl.xlabel("Time (seconds)")
    xlocs = _np.int32([n*samplerate/(2*windowsize) for n in range(1 + 2*samples.size//samplerate)])
    xlabels = ["%.1f" % (0.5*int(n)) for n in range(xlocs.size)]
    _pl.xlim(0, 4*samplerate//windowsize)
    _pl.xticks(xlocs, xlabels)

    _pl.ylabel("RMS Power")
    _pl.ylim([0, 2*_np.max(first3seconds)])

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()

    _pl.clf()
    return
