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

""" Functions for pitch detection via Harmonic Wavelet Transforms (HWT). 
    NOTE: Implementation of this submodule is incomplete as there is no proper pitch detection function.
          This submodule is kept, nonetheless, as it contains functions that are not trivial to implement yet useful
          for analyzing whether harmonic wavelets are appropriate for detecting pitch in a given input. """

import mathhelper as _math

import numpy as _np

import mtheory as _mt


def stht(sig, lowest_note=256, octaves=3, sampleRate=44100):
    """ Short Time Harmonic Transform on an input signal array. """
    window_size = sampleRate
    
    notes_per_window = 15 # to change this, we'd need to considerably change the implementation below (round/floor/ceil)
    windows = int(_np.ceil(len(sig)/window_size))
    harmonic_map = _np.zeros((octaves*12, windows*notes_per_window*(2**(octaves-1))))

    for w in range(0, windows):
        # First we calculate an FFT where each bin contains 1Hz
        # Then we calculate the IFFT of each semitone band
        #   each resulting IFFT corresponds to a harmonic wavelet coefficient series
        # Since each series has a different length, we normalize them all to match the smallest
        windowed_sig = sig[window_size*w:window_size*(w+1)]

        X = _np.fft.fft(windowed_sig, window_size)

        low_freq = lowest_note
        high_freq = lowest_note*_mt.semitone

        i = 0
        stop_note = lowest_note*(2**octaves)
        while high_freq < stop_note:
            interval = _np.round(high_freq) - _np.round(low_freq)
            segment_ifft = _np.fft.ifft(X[_np.round(low_freq):_np.round(high_freq)])
            
            current_bin = 0
            next_bin = notes_per_window/interval
            for t in range(0, segment_ifft.size):
                if _np.floor(next_bin) == _np.floor(current_bin):
                    harmonic_map[i][notes_per_window*w+int(_np.floor(current_bin))] += _np.abs(segment_ifft[t])*15/interval
                else:
                    harmonic_map[i][notes_per_window*w+int(_np.floor(current_bin))] += _np.abs(segment_ifft[t])*(_np.floor(next_bin)-current_bin)
                    if next_bin - _np.floor(next_bin) > 1e-5:
                        harmonic_map[i][notes_per_window*w+int(_np.floor(next_bin))] += _np.abs(segment_ifft[t+1])*(next_bin - _np.floor(next_bin))

                current_bin = next_bin
                next_bin += notes_per_window/interval

            low_freq = high_freq
            high_freq = high_freq*_mt.semitone
            i += 1

    return harmonic_map

def fht(sig, lowest_note=246.94*2**(1/24), octaves=3, sampleRate=44100):
    """ Fast Harmonic Transform on an input signal array. """
    X = _np.fft.fft(sig)

    bin_f = sampleRate/X.size
    low_freq = lowest_note
    high_freq = lowest_note*_mt.semitone
    base_notes = _math.round(high_freq, bin_f) - _math.round(low_freq, bin_f)
    p_notes_last_octave = 2**(octaves-1)
    
    hmap = _np.zeros((octaves*12, base_notes*p_notes_last_octave))

    i = 0
    stop_note = lowest_note*(2**octaves)
    while high_freq < stop_note:
        octave_scale = int(2**(i//12))
        note_repeat = p_notes_last_octave//octave_scale
        time_bins = _math.round(high_freq, bin_f) - _math.round(low_freq, bin_f)
        bin_dt = base_notes*octave_scale/time_bins

        segment_ifft = _np.fft.ifft(X[_math.round(low_freq, bin_f) : _math.round(high_freq, bin_f)])
            
        current_bin = 0
        next_bin = bin_dt
        n = 0
        for t in range(0, segment_ifft.size):
            f_curr_bin = int(_np.floor(current_bin))
            f_next_bin = int(_np.floor(next_bin))
            if _np.floor(next_bin) == _np.floor(current_bin):
                hmap[i][f_curr_bin*note_repeat:(f_curr_bin + 1)*note_repeat] += _np.abs(segment_ifft[t])*octave_scale/segment_ifft.size
            else:
                hmap[i][f_curr_bin*note_repeat:(f_curr_bin + 1)*note_repeat] += _np.abs(segment_ifft[t])*(_np.floor(next_bin)-current_bin)*octave_scale/(bin_dt*segment_ifft.size)
                if (next_bin - _np.floor(next_bin))/bin_dt > 0.05:
                    print(segment_ifft.size)
                    print(t)
                    print(next_bin)
                    hmap[i][f_next_bin*note_repeat:(f_next_bin + 1)*note_repeat] += _np.abs(segment_ifft[t+1])*(next_bin - _np.floor(next_bin))*octave_scale/(bin_dt*segment_ifft.size)

            current_bin = next_bin
            next_bin += bin_dt

        low_freq = high_freq
        high_freq = high_freq*_mt.semitone
        i += 1

    return hmap
