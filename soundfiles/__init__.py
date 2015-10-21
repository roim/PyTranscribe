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

""" Functions for operating on sound files. """


def readfile(audiopath="wave.npz"):
    """ Returns the sample rate and samples contained in a given audio file. Format support is restricted. """
    import numpy as np
    from os.path import splitext

    extension = splitext(audiopath)[1].lower()
    if extension == ".wav":
        samplerate, samples = wav.read(audiopath)
    elif extension == ".npy":
        samplerate = 44100
        samples = np.load(audiopath)
    elif extension == ".npz":
        samplerate = 44100
        samples = np.load(audiopath)["arr_0"]
    else:
        raise NotImplementedError("Unknown file extension")

    return samplerate, samples


def writewav(audiopath="wave.npz", outpath="out.wav"):
    """ Write a wav file given an input sample array file that can be read with readfile. """
    import numpy as np
    import scipy.io.wavfile as wav

    samplerate, samples = readfile(audiopath)
    if np.issubdtype(samples.dtype, np.integer):
        max = np.iinfo(samples.dtype).max
        samples = np.divide(samples, max)

    wav.write(outpath, samplerate, samples)


def write_m21stream_to_midi(s, filePath='audio.midi'):
    """ Writes a Music21 stream to a midi file. """
    import music21 as _music
    mf = _music.midi.translate.streamToMidiFile(s)
    mf.open(filePath, 'wb')
    mf.write()
    mf.close()


def write_m21stream_to_xml(s, filePath='audio.xml'):
    """ Writes a Music21 sream to an xml file. """
    import music21 as _music
    mf = _music.musicxml.m21ToString(s)
    mf.open(filePath, 'wb')
    mf.write()
    mf.close()
