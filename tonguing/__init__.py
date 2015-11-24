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

""" Module containing utilities for tonguing detection. """

import numpy as _np

from scipy.signal import hilbert as _hilbert


def _envelope(x):
    """Calculates the signal's _envelope through its analytic representation's absolute value."""
    return _np.abs(_hilbert(x))


def _exponential_smoothing(x, x_s0=0, alpha=0.1):
    """Performs exponential smoothing of a given series."""
    x_s = _np.multiply(x, alpha)
    x_s[0] += (1 - alpha)*x_s0
    # Slow. Bad. Ugly.
    for t in range(1, x_s.size):
        x_s[t] += (1 - alpha)*x_s[t-1]

    return x_s


class TonguingDetector(object):
    def __init__(self, min_duration=0.01, threshold=0.107, fs=44100, step=5):
        # Slice the input arrays by this step.
        # I know, I know... Last minute optimizations...
        self.step = step

        # Minimum amount of consecutive samples to consider a state (noisy or silent) detected.
        self.min_samples = min_duration*fs/step

        # Minimum amplitude level to consider a sample noisy.
        self.threshold = threshold

        # State implied by the last analyzed sample alone.
        self.current_tentative_state = False
        # Number of samples that consecutively implied this state.
        self.current_tentative_samples = 0

        # Last detected state.
        self.last_detected_state = False

        # Last smoothed sample (state kept between feeds so we can continue the smoothing from a previous point).
        self.x_s0 = 0

    def feed(self, x):
        """Feeds x into the detector."""
        # Reduce input array.
        x = x[::self.step]

        # We return True if we detected a tonguing during 'x'
        # i.e. a transition True -> False on 'self.last_detected_state'
        e = _envelope(x)
        e_s = _exponential_smoothing(e, self.x_s0)
        self.x_s0 = e_s[-1]

        tongued = False
        for s in e_s:
            s_state = s > self.threshold
            if s_state == self.current_tentative_state:
                self.current_tentative_samples += 1
            else:
                self.current_tentative_state = s_state
                self.current_tentative_samples = 1

            if self.current_tentative_samples > self.min_samples:
                if self.last_detected_state and not self.current_tentative_state:
                    tongued = True

                self.last_detected_state = self.current_tentative_state

        return tongued
