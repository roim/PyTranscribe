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

""" Musical theory constants. Assumes A440 tuning. """

""" Array of note frequencies. """
notes = [ 16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53, 2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07, 4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040.00, 7458.62, 7902.13 ]


""" Mapping from note frequency to note name. """
note_name = {16.35: 'C0', 17.32: 'C#0', 18.35: 'D0', 19.45: 'D#0', 20.60: 'E0', 21.83: 'F0', 23.12: 'F#0', 24.50: 'G0', 25.96: 'G#0', 27.50: 'A0', 29.14: 'A#0', 30.87: 'B0', 32.70: 'C1', 34.65: 'C#1', 36.71: 'D1', 38.89: 'D#1', 41.20: 'E1', 43.65: 'F1', 46.25: 'F#1', 49.00: 'G1', 51.91: 'G#1', 55.00: 'A1', 58.27: 'A#1', 61.74: 'B1', 65.41: 'C2', 69.30: 'C#2', 73.42: 'D2', 77.78: 'D#2', 82.41: 'E2', 87.31: 'F2', 92.50: 'F#2', 98.00: 'G2', 103.83: 'G#2', 110.00: 'A2', 116.54: 'A#2', 123.47: 'B2', 130.81: 'C3', 138.59: 'C#3', 146.83: 'D3', 155.56: 'D#3', 164.81: 'E3', 174.61: 'F3', 185.00: 'F#3', 196.00: 'G3', 207.65: 'G#3', 220.00: 'A3', 233.08: 'A#3', 246.94: 'B3', 261.63: 'C4', 277.18: 'C#4', 293.66: 'D4', 311.13: 'D#4', 329.63: 'E4', 349.23: 'F4', 369.99: 'F#4', 392.00: 'G4', 415.30: 'G#4', 440.00: 'A4', 466.16: 'A#4', 493.88: 'B4', 523.25: 'C5', 554.37: 'C#5', 587.33: 'D5', 622.25: 'D#5', 659.25: 'E5', 698.46: 'F5', 739.99: 'F#5', 783.99: 'G5', 830.61: 'G#5', 880.00: 'A5', 932.33: 'A#5', 987.77: 'B5', 1046.50: 'C6', 1108.73: 'C#6', 1174.66: 'D6', 1244.51: 'D#6', 1318.51: 'E6', 1396.91: 'F6', 1479.98: 'F#6', 1567.98: 'G6', 1661.22: 'G#6', 1760.00: 'A6', 1864.66: 'A#6', 1975.53: 'B6', 2093.00: 'C7', 2217.46: 'C#7', 2349.32: 'D7', 2489.02: 'D#7', 2637.02: 'E7', 2793.83: 'F7', 2959.96: 'F#7', 3135.96: 'G7', 3322.44: 'G#7', 3520.00: 'A7', 3729.31: 'A#7', 3951.07: 'B7', 4186.01: 'C8', 4434.92: 'C#8', 4698.63: 'D8', 4978.03: 'D#8', 5274.04: 'E8', 5587.65: 'F8', 5919.91: 'F#8', 6271.93: 'G8', 6644.88: 'G#8', 7040.00: 'A8', 7458.62: 'A#8'}


""" Mapping from note name to note frequency. """
note_frequency = {'C0': 16.35, 'C#0': 17.32, 'D0': 18.35, 'D#0': 19.45, 'E0': 20.60, 'F0': 21.83, 'F#0': 23.12, 24.50: 'G0', 25.96: 'G#0', 'A0': 27.50, 'A#0': 29.14, 'B0': 30.87, 'C1': 32.70, 'C#1': 34.65, 'D1': 36.71, 'D#1': 38.89, 'E1': 41.20, 'F1': 43.65, 'F#1': 46.25, 49.00: 'G1', 51.91: 'G#1', 'A1': 55.00, 'A#1': 58.27, 'B1': 61.74, 'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41, 'F2': 87.31, 'F#2': 92.50, 98.00: 'G2', 103.83: 'G#2', 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47, 'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 196.00: 'G3', 207.65: 'G#3', 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94, 'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 392.00: 'G4', 415.30: 'G#4', 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88, 'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25, 'F5': 698.46, 'F#5': 739.99, 783.99: 'G5', 830.61: 'G#5', 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77, 'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51, 'F6': 1396.91, 'F#6': 1479.98, 1567.98: 'G6', 1661.22: 'G#6', 'A6': 1760.00, 'A#6': 1864.66, 'B6': 1975.53, 'C7': 2093.00, 'C#7': 2217.46, 'D7': 2349.32, 'D#7': 2489.02, 'E7': 2637.02, 'F7': 2793.83, 'F#7': 2959.96, 3135.96: 'G7', 3322.44: 'G#7', 'A7': 3520.00, 'A#7': 3729.31, 'B7': 3951.07, 'C8': 4186.01, 'C#8': 4434.92, 'D8': 4698.63, 'D#8': 4978.03, 'E8': 5274.04, 'F8': 5587.65, 'F#8': 5919.91, 6271.93: 'G8', 6644.88: 'G#8', 'A8': 7040.00, 7458.62: 'A#8'}


""" Relative frequency increase in a semitone. """
semitone = 2**(1/12)


""" Lowest note playable on a western concert flute. """
lowest_flute_note = note_frequency["C4"]


""" Highest note playable on a western concert flute. """
highest_flute_note = note_frequency["D7"]


""" Array of notes playable on a western concert flute. """
flute_notes = [f for f in notes if f >= lowest_flute_note and f <= highest_flute_note]
