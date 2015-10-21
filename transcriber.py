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

""" At the moment this file should only be used as a code entry point.
Although this file has a transcriber that can be used to obtain the transcription of a musical signal, no work has
been done to isolate it into a reusable module, as I spent no time thinking of an appropriate API for such a module.
Submodules used by the transcriber (PDAs, clustering algorithms, etc) are isolated and ready to be reused. """

# Debugging parameters
DEBUG_PERF  = False
DEBUG_NOISE = False
DEBUG_NOTE = True
DEBUG_TICK = False
DEBUG_TONG = True
DEBUG_WAVE = True

# Output parameters
OUT_FILENAME = 'out.txt'
MIDI_FILENAME = 'out.midi'
WRITE_OUT = True
WRITE_MIDI = True
WRITE_XML = True

print("### Importing")

# Windows dependent - used *only* to finalize on keyboard interaction.
# If you want to use this on another platform you're smart enough to figure out what to do.
from msvcrt import kbhit
from msvcrt import getch

# Python
import math
import os
import time

# External
import music21
import numpy as np
import scipy.stats

# Internal
import clustering as clst
import mathhelper as mh
import mic
import mtheory as mt
import pda.hps
import soundfiles as sf
import tonguing as tong

class Transcriber(object):
    """ Class to retrieve samples from the default microphone.
    Includes utilities such as noise level detection. """

    def __init__(self, blocks_per_sec, samples_per_block, noise_detection_duration):
        """ Initializes a microphone listener object.
        NOTE: guidelines for defining the initializer parameters:
            'samples_per_block == int(44100/blocks_per_sec)' -> no sample overlapping between blocks, every sample received is used.
            'samples_per_block > int(44100/blocks_per_sec)'  -> sample overlapping between blocks, every sample received is used, some are used multiple times.
            'samples_per_block < int(44100/blocks_per_sec)'  -> no sample overlapping, some samples are discarded (will raise).
        We generally want 'samples_per_block' to be an integer multiple of '44100/samples_per_read', so that every sample is used the same amount of times. """

        if samples_per_block < int(44100/blocks_per_sec):
            raise ValueError("samples_per_block must be >= int(44100/blocks_per_sec)")

        # Channels read by the mic.
        self.channels = 1

        # Input rate.
        self.rate = 44100

        # Blocks processed per second. A block is a set of samples that will be processed by PDAs.
        self.blocks_per_sec = blocks_per_sec

        # Amount of samples in each block.
        self.samples_per_block = samples_per_block

        # A buffer read is the retrieval of set of samples from the mic.
        # Keep in mind the amount of samples per read is not necessarily the same as the amount of samples in an
        # input block processed by the PDA, since there might be sample overlapping on the PDA but never on the mic.
        # The amount of overlapping is determined implicitly by the variables given in this initializer.
        self.samples_per_read = int(self.rate/blocks_per_sec)

        # Mic Listener.
        self.mic = mic.MicListener(self.samples_per_read, self.channels, self.rate, debug_wave=DEBUG_WAVE, print=True)

        # Reads necessary to detect noise levels.
        self.noise_detection_reads = noise_detection_duration*self.rate/self.samples_per_read

        self.block = np.zeros(samples_per_block)
        self.tong = None
        self.noise_threshold = None

        self.total_ticks = 0

        self.notes = []
        self.current_note = "NOVALUE"
        self.previous_note = "NOVALUE"
        self.current_ticks = 0
        self.currently_slurring = False

        if DEBUG_PERF:
            self.hps_time = -1
            self.read_time = -1

        if WRITE_OUT:
            self.out = ""

        return

    def close(self):
        self.mic.close()
        return

    def detect_noise(self):
        self.noise_threshold = self.mic.detect_noise(self.noise_detection_reads)
        self.tong = tong.TonguingDetector(threshold=1.25*self.noise_threshold)
        return self.noise_threshold

    def transcribe(self):
        if not self.tong:
            raise AssertionError("Please initialize the tonguing detector first. (missing a call to detect_noise()?)")

        self.total_ticks += 1
        new_samples = self.mic.listen()

        if DEBUG_PERF:
            rms_start_time = time.time()

        rms = np.sqrt(np.mean(np.square(new_samples)))

        if DEBUG_PERF:
            self.rms_time = time.time() - rms_start_time
            tong_start_time = time.time()

        # Feed the new samples to the Tonguing Detector.
        # Beware we shouldn't send repeated samples, so we send the new_samples and not the entire block.
        tongued = self.tong.feed(new_samples)

        if DEBUG_PERF:
            self.tong_time = time.time() - tong_start_time

        if tongued:
            if self.current_ticks > 2:
                # We detected tonguing, so split the current note.
                # TODO: if 'previous_note' is considered noisy, account for it in the duration.
                # TODO: Consider whether we should increment the current tick partially (proportionally to the audible portion?).
                self.notes.append({"name":self.current_note, "duration":np.log2(self.current_ticks), "ticks":self.current_ticks, "slur":"stop" if self.currently_slurring else False})
                self.currently_slurring = False
                if DEBUG_NOTE:
                    print("%s\t %d\t %.3fs"%(self.current_note, self.current_ticks, self.current_ticks/self.blocks_per_sec))
                if DEBUG_TONG:
                    print("TONG")
                if WRITE_OUT:
                    self.out += "%d\t: TONG\n" % self.total_ticks

            self.current_ticks = 0

        # Add the new_samples at the beginning of the block, so they replace the oldest values.
        self.block[0:self.samples_per_read] = new_samples

        # Now roll the block back so that it is in chronological order.
        # Not strictly necessary as we're discarding phase, but it makes replacing old values easier and also ensures
        # usual windowing will smooth discontinuities at the borders.
        self.block = np.roll(self.block, -self.samples_per_read)

        # No need to proceed if we're to discard the pitch due to insufficient RMS power in the block.
        if not DEBUG_NOISE and rms < self.noise_threshold:
            self.out += "%d\t: 'rms < self.noise_threshold'\n" % self.total_ticks
            return;

        # We want pitch, so pass the block to the PDA
        if DEBUG_PERF:
            hps_start_time = time.time()

        perceived_f = pda.hps.hps(self.block, fs=self.rate, harmonics=3, precision=2)

        if DEBUG_PERF:
            self.hps_time = time.time() - hps_start_time

        # and tune the pitch down to a known note.
        tuned_f = mh.find_nearest_value(mt.notes, perceived_f)
        note = mt.note_name[tuned_f]

        # TODO: rough error percentage estimate
        error = perceived_f - tuned_f
        percentage = np.sign(error) * 2 * error/(tuned_f*(1 + mt.semitone) if error > 0 else tuned_f*(1 - mt.semitone))

        if note == self.current_note:
            if self.current_note == self.previous_note:
                # We're receiving a new sample of the current note.
                self.current_ticks += 1
            else:
                # Our 'previous_note' measurement was probably noisy.
                # Pretend it was a measurement of 'current_note', and account ticks for both.
                self.current_ticks += 2
        elif note == self.previous_note:
            # Keep in mind that all notes are 'tentative' until their tick count is > n, so:
            #   - C5 C5 C5 D5 D5 means we successfully identified a C5 and the beginning of a D5, assuming n is 1.
            if self.current_ticks > 2:
                self.notes.append({"name":self.current_note, "duration":np.log2(self.current_ticks), "ticks":self.current_ticks, "slur":"continue" if self.currently_slurring else "start"})
                self.currently_slurring = True
                if DEBUG_NOTE:
                    print("%s\t %d\t %.3fs"%(self.current_note, self.current_ticks, self.current_ticks/self.blocks_per_sec))

            self.current_note = note
            self.current_ticks = 2
        elif self.previous_note != self.current_note:
            # Experimentally, it's pretty rare to have 2 noisy detections in a row, so if we find 2 different measurements
            # we can assume the old note has ended.
            #   - C5 C5 C5 D5 E5 means we identified a C5 end, but we don't know the next note yet.
            if self.current_ticks > 2:
                self.notes.append({"name":self.current_note, "duration":np.log2(self.current_ticks), "ticks":self.current_ticks, "slur":"continue" if self.currently_slurring else False})
                if DEBUG_NOTE:
                    print("%s\t %d\t %.3fs"%(self.current_note, self.current_ticks, self.current_ticks/self.blocks_per_sec))

            # We currently have no idea of the note being played, so assign an error string to it.
            # When we have k identical detections in a row (with k defined in the elifs above) we will successfully
            # assign the current note.
            self.current_note = "NOISE_ERR"
            self.current_ticks = 0

        self.previous_note = note

        if DEBUG_TICK:
            print("%s\t (%.3f)\t@ %.2f" % (note, percentage, rms))
        if WRITE_OUT:
            self.out += "%d\t: %s\t (%.3f)\t@ %.2f\r\n" % (self.total_ticks, note, percentage, rms)
        return

    def finalize(self):
        self.close()

        # Extract the last note.
        # TODO: Extract function
        if self.current_ticks > 2:
            self.notes.append({"name":self.current_note, "duration":np.log2(self.current_ticks), "ticks":self.current_ticks})
            if DEBUG_NOTE:
                print("%s\t %d\t %.3fs"%(self.current_note, self.current_ticks, self.current_ticks/self.blocks_per_sec))

        print("\n\n###### Detected notes:")
        for note in self.notes:
            print(note)

        durations = np.array([n["duration"] for n in self.notes])
        clusters = clst.equidistant_clusterize(durations)
        corrected_notes = [{"name":n["name"], "duration":2**mh.find_nearest_value(clusters, n["duration"]), "slur":n["slur"] if "slur" in n.keys() else None} for n in self.notes]


        print("\n\n###### Corrected notes:")
        for note in corrected_notes:
            print(note)

        most_common = scipy.stats.mode([n["duration"] for n in corrected_notes])[0][0]
        tempo = int(round(60*self.blocks_per_sec/most_common, 0))

        while tempo < 90:
            tempo *= 2
            most_common /= 2

        while tempo > 180:
            tempo /= 2
            most_common *= 2

        if WRITE_MIDI or WRITE_XML:
            s = music21.stream.Stream()
            s.append(music21.tempo.MetronomeMark(number=tempo))
            s.append(music21.meter.TimeSignature('4/4'))
            for note in corrected_notes:
                n = music21.note.Note()
                n.pitch.name = note["name"]
                n.duration.quarterLength = note["duration"]/most_common
                s.append(n)
                note["music21"] = n

            slurring = False
            slur = music21.spanner.Slur()
            for note in corrected_notes:
                if note["slur"] == "start":
                    slurring = True
                if slurring:
                    slur.addSpannedElements([note["music21"]])
                if note["slur"] == "stop":
                    slurring = False
                    s.insert(0, slur)
                    slur = music21.spanner.Slur()

            if slurring:
                s.insert(0, slur)

            s.insert(0, s.analyze('key'))

            if WRITE_MIDI:
                sf.write_m21stream_to_midi(s, MIDI_FILENAME)
            if WRITE_XML:
                s.show('musicxml')

        if WRITE_OUT:
            print("### Writing processed output file")
            f = open(OUT_FILENAME, 'w')
            f.write(self.out)
            f.flush()
            os.fsync(f)

        return

if __name__ == "__main__":
    print("### Initializing Transcriber")
    trs = Transcriber(blocks_per_sec = 60.0,
                       samples_per_block = 1470,
                       noise_detection_duration = 3.0)

    print("### Detecting noise threshold")
    noise_threshold = trs.detect_noise()

    print("Noise RMS detected at %.4f" % noise_threshold)

    if DEBUG_PERF:
        cycle = 0
        out_buffer = ""

    for i in range(3):
        print("### TRANSCRIBING")

    while(True):
        if DEBUG_PERF:
            start_time = time.time()

        trs.transcribe()

        if DEBUG_PERF:
            out_buffer += "Read:  %.4f" % trs.read_time + "\n"
            out_buffer += "RMS:   %.4f" % trs.rms_time + "\n"
            out_buffer += "Tong   %.4f" % trs.tong_time + "\n"
            out_buffer += "HPS:   %.4f" % trs.hps_time + "\n"
            out_buffer += "Total: %.4f" % (time.time() - start_time) + "\n"
            out_buffer += "-------------\n"

            cycle += 1
            if cycle % INPUT_BLOCKS_PER_SEC == 0:
                print(out_buffer)
                out_buffer = ""

        if kbhit():
            trs.finalize()
            getch()
            break
