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

""" Function for plots related to clustering algorithms. """

import matplotlib.pyplot as _pl
import numpy as _np

import clustering.kde as _kde


# Sample data (Ode to Joy): x = _np.log2([20, 23, 21, 23, 23, 22, 24, 22, 22, 20, 22, 22, 34, 10, 44, 21, 21, 21, 23, 23, 22, 25, 22, 50, 22, 22, 33, 10, 50, 21, 21, 20, 25, 22, 10, 11, 25, 25, 22, 10, 10, 24, 22, 25, 22, 26, 44, 23, 20, 22, 22, 21, 20, 11, 10, 50, 22, 24, 33, 11, 53])
def plot_histogram(x, bounds=(1, 12)):
    """Plots a histogram for a given array."""
    plot_width = bounds[1] - bounds[0]
    hist_points = _np.linspace(bounds[0], bounds[1], 10*plot_width)
    _pl.hist(x, bins=hist_points, range=bounds, fc='gray', histtype='stepfilled', alpha=0.5, normed=False)

    _pl.xlabel("Duração (lg da quantidade de janelas)")
    _pl.ylabel("Quantidade de notas")
    _pl.show()
    return


def plot_kde(x, bw=0.15, bounds=(1, 12)):
    """Plots a histogram and the KDE for a given array."""
    plot_width = bounds[1] - bounds[0]

    hist_points = _np.linspace(bounds[0], bounds[1], 10*plot_width)
    _pl.hist(x, bins=hist_points, range=bounds, fc='gray', histtype='stepfilled', alpha=0.5, normed=True)

    kde_points = _np.linspace(bounds[0], bounds[1], 100*plot_width)
    pdf = _kde.kde(x, kde_points, bw)
    _pl.plot(kde_points, pdf, color='red', alpha=0.75, lw=2)

    _pl.xlabel("Duração (lg da quantidade de janelas)")
    _pl.ylabel("Quantidade de notas (normalizada)")
    _pl.show()
    return
