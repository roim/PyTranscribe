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

import clustering as _clst


# Sample data (Ode to Joy): x = _np.array([5.129283017, 4, 5.044394119, 4, 4, 3.906890596, 4.857980995, 3.906890596, 5.129283017, 5.129283017, 5.087462841, 3.807354922, 5.044394119, 4, 4.087462841, 4, 4.700439718, 3.906890596, 4.087462841, 4, 4.857980995, 4.584962501, 3.906890596, 3.321928095, 3.807354922, 3.807354922, 3.807354922, 4.169925001, 2.807354922, 3.807354922, 3.906890596, 3.906890596, 4.169925001, 3.906890596, 3.700439718, 3.700439718, 5.129283017])
def plot_kde(x, bw=0.15, bounds=(1,12)):
    """Plots a histogram and the KDE for a given array."""
    plot_width = bounds[1] - bounds[0]

    hist_points = _np.linspace(bounds[0], bounds[1], 10*plot_width)
    _pl.hist(x, bins=hist_points, range=bounds, fc='gray', histtype='stepfilled', alpha=0.5, normed=True)

    kde_points = _np.linspace(bounds[0], bounds[1], 100*plot_width)
    pdf = _clst.kde(x, kde_points, bw)
    _pl.plot(kde_points, pdf, color='red', alpha=0.75, lw=2)

    _pl.show()
