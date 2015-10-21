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

""" Functions for clustering based on Kernel Density Estimation (KDE). """

import numpy as _np

from scipy.signal import argrelmax as _argrelmax
from scipy.stats import gaussian_kde as _gkde


def kde(x, points, bw=0.15):
    """ Kernel Density Estimation on a given array of occurrences. """
    # SciPy's KDE weighs bandwidth by the input's covariance.
    # Divide 'bw' by the sample stddev to be consistent with the literature.
    kde = _gkde(x, bw_method=bw/x.std(ddof=1))
    return kde.evaluate(points)


def kde_clusterize(x, bw=0.15, bounds=(1,10), bins_per_unit=100):
    """ Returns clusters obtained by peaks on the KDE of the given array. """
    bound_width = bounds[1] - bounds[0]
    total_points = bound_width * bins_per_unit
    x_points = _np.linspace(bounds[0], bounds[1], total_points)

    # Estimate the kernel density
    pdf = kde(x, x_points, bw)

    # Extract local maxima peaks of at least 10% of the highest
    max = _np.max(pdf)
    maxima = _np.array([arg_peak for arg_peak in _argrelmax(pdf)[0] if pdf[arg_peak] > 0.1*max])

    return x_points[maxima]
