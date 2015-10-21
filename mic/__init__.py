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

""" Module containing utilities to read samples from a microphone. """

import numpy as _np
import pyaudio


class MicListener(object):

    def __init__(self, samples_per_read, channels=1, rate=44100, debug_wave=False, debug_perf=False, print=False):
        if debug_perf:
            global time
            import time

        self.samples_per_read = samples_per_read
        self.channels = channels
        self.rate = rate
        self.debug_wave = debug_wave
        self.debug_perf = debug_perf
        self.print = print

        self.total_ticks = 0
        
        if self.debug_perf:
            self.read_time = -1

        if self.debug_wave:
            self.wave = _np.array([], dtype=_np.float32)

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(format = pyaudio.paFloat32,
                                     channels = self.channels,
                                     rate = self.rate,
                                     input = True,
                                     input_device_index = self._find_input_device(),
                                     frames_per_buffer = self.samples_per_read)

        return

    def close(self):
        """ Closes the audio resources. """
        self._stream.close()
        if self.debug_wave:
            self._write_wave()
        return

    def detect_noise(self, noise_detection_reads=None):
        """ Detects a safe noise RMS level (threshold) based on the highest RMS values during the detection reads. """
        if not noise_detection_reads:
            noise_detection_reads = 3.0*self.rate/self.samples_per_read

        rms_noise_values = _np.zeros(noise_detection_reads)
        for i in range(rms_noise_values.size):
            samples = self.listen()
            RMS = _np.sqrt(_np.mean(_np.square(samples)))
            rms_noise_values[i] = RMS

        rms_noise_values.sort()
        pct98i = int(0.98*noise_detection_reads)
        pct98rms = rms_noise_values[pct98i]
        return 1.1*pct98rms

    def listen(self):
        """ Returns an nparray with samples from the mic, or an array of zeros if the mic can't be read. """
        self.total_ticks += 1
        if self.debug_perf:
            read_start_time = time.time()

        try:
            buffer = self._stream.read(self.samples_per_read)
        except IOError as e:
            if self.print():
                print("\tError recording: %s" % e)
            return _np.zeros(self.samples_per_read)

        samples = _np.frombuffer(buffer, _np.float32)

        if self.debug_wave:
            self.wave = _np.append(self.wave, samples)

        if self.debug_perf:
            self.read_time = time.time() - read_start_time

        # Convert the input buffer into an np array.
        return samples

    def _find_input_device(self):
        """ Finds the default input device's index. """
        for i in range(self._pa.get_device_count()):
            devinfo = self._pa.get_device_info_by_index(i)
            if self.print:
                print("Device %d: %s" % (i, devinfo["name"]))

            for keyword in ["mic", "input"]:
                if keyword in devinfo["name"].lower():
                    if self.print:
                        print("Found input: device %d - %s" % (i,devinfo["name"]))

                    return i

        if self.print:
            print("Using default input device.")

        return None

    def _write_wave(self):
        """ Writes the recorded wave array to a numpy compressed file (.npz). """
        filename = "wave"
        if self.print:
            print("### Writing %s.npz" % filename)

        _np.savez_compressed(filename, self.wave)
        return
