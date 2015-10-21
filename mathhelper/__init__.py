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

""" Module with math helper functions, including operations on arrays. """

import numpy as _np


def round(x, base):
    """ Rounds x to a given base. """
    return int(_np.round(x/base))


def find_nearest_idx(array, value):
    """ Returns the index of the array element closest to value. """
    return (_np.abs(array - value)).argmin()


def find_nearest_value(array, value):
    """ Returns the array value closest to the input value. """
    return array[find_nearest_idx(array, value)]
