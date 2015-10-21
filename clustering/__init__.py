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

""" Module containing clustering algorithms. """

__all__ = ['kde']


import mathhelper as _mh
import numpy as _np


def evaluate_clustering(x, clusters):
    """ Returns the root mean square of the differences between array points and their closest clusters. """
    x = _np.asarray(x)
    error = 0
    # Can be heavily optimized, but let's assume there are few clusters.
    for point in x:
        c = _mh.find_nearest_value(clusters, point)
        error += (point - c)**2

    return _np.sqrt(error/x.size)


def equidistant_clusterize(x, interval=1, bounds=(1,10)):
    """ Returns clusters between 'bounds' that are equidistant by 'interval' with minimum RMS error. """
    candidates = _np.linspace(0, interval, 1000)

    min_error = -1
    best_clusters = []
    for candidate in candidates:
        clusters = []
        c = bounds[0] + candidate
        while c <= bounds[1]:
            clusters.append(c)
            c += interval

        error = evaluate_clustering(x, clusters)
        if error < min_error or min_error < 0:
            min_error = error
            best_clusters = clusters

    return _np.array(best_clusters)
