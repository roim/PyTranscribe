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

""" Contains functions to plot graphs related to the signal windowing. """

import matplotlib.pyplot as _pl
import numpy as _np


def plot_kaiser_series(windowsize, beta=7.14285, n=4, title="", plotpath=None):
    """ Plots a series of 'n' Kaiser windows. """
    window_center = windowsize//2

    s = _kaiser_series(windowsize, beta, n)

    _pl.figure(figsize=(10, 2))
    _pl.title(title)
    _pl.plot(s, 'b')

    _pl.xlabel("Index")
    xlocs = _np.int32([n * window_center for n in range(n*2 + 1)])
    _pl.xlim([0, _np.max(xlocs)])
    _pl.xticks(xlocs)

    _pl.ylabel("Amplitude")
    _pl.ylim(0, 1)

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()

    _pl.clf()
    return


def plot_double_kaiser_series(windowsize, beta=7.14285, n=4, title="", plotpath=None):
    """ Plots a series of 'n' Kaiser windows with 0.5 superposition. """
    window_center = windowsize//2

    s = _kaiser_series(windowsize, beta, n)
    s2 = _np.roll(s, window_center)

    _pl.figure(figsize=(10, 2))
    _pl.title(title)
    _pl.plot(s, 'b')
    _pl.plot(s2, 'r')

    _pl.xlabel("Index")
    xlocs = _np.int32([n * window_center for n in range(n*2 + 1)])
    _pl.xlim([0, _np.max(xlocs)])
    _pl.xticks(xlocs)

    _pl.ylabel("Amplitude")
    _pl.ylim(0, 1)

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()

    _pl.clf()
    return


def _kaiser_series(windowsize, beta, n):
    """ Returns a series of Kaiser windows with the given parameters. """
    w = _np.kaiser(windowsize, beta)
    return _np.tile(w, n)
