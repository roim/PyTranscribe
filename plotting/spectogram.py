# Code for this submodule was modified from 'Create audio spectograms with Python' by Frank Zalkow.
# Original snippet available at http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html
#
# This work is licensed under a Creative Commons Attribution 3.0 Unported License, 
# available at https://creativecommons.org/licenses/by/3.0/.
#   Frank Zalkow, 2012-2013
#   Rodrigo Roim Ferreira, 2015

""" Utilities for plotting spectograms. """


import matplotlib.pyplot as _pl
import numpy as _np
import numpy.lib.stride_tricks as _st

import mtheory as _mt
import pda as _pda


def stft(sig, frameSize=2560, overlapFac=0.5, localFac=1, window=lambda x:_np.kaiser(x, 7.14285)):
    """ Short Time Fourier Transform of audio signal. """
    # Correct the overlapping factor to account for the localization factor
    correctionFac = 1/overlapFac - localFac/overlapFac + localFac
    overlapFac = overlapFac*correctionFac

    fillSize = (1-localFac)*frameSize
    win = _np.concatenate([_np.zeros(fillSize/2), window(frameSize*localFac), _np.zeros(fillSize/2)])
    hopSize = int(frameSize - _np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = _np.append(_np.zeros(_np.floor(frameSize/2.0)), sig)
    # cols for windowing
    cols = _np.ceil((samples.size - frameSize)/float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = _np.append(samples, _np.zeros(frameSize))
    
    frames = _st.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return _np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    """ Scale frequency axis logarithmically. """
    timebins, freqbins = _np.shape(spec)

    scale = _np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = _np.unique(_np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = _np.complex128(_np.zeros([timebins, scale.size]))
    for i in range(0, scale.size):
        if i == scale.size - 1:
            newspec[:,i] = _np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = _np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = _np.abs(_np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, scale.size):
        if i == scale.size - 1:
            freqs += [_np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [_np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs


def plotgrayimage(ims, colormap='jet', plotpath=None):
    """ Plot gray image. """
    _pl.figure(figsize=(15, 7.5))
    _pl.imshow(_np.abs(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    _pl.colorbar()

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()


def plotstft(audiopath="wave.npz", binsize=1470, guidelines=False, plotpath=None, colormap="jet"):
    """ Plots the spectrogram of a given file. """
    import soundfiles as sf
    samplerate, samples = sf.readfile(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*_np.log10(_np.abs(sshow)/10e-6) # amplitude to decibel
    timebins, freqbins = _np.shape(ims)

    if guidelines:
        min_f = _np.min(ims)
        notebins = _pda.note_bins(_mt.notes, binsize)
        for t in range(len(ims)//8):
            t = t*8
            for n in range(len(notebins)):
                ims[t][notebins[n]] = min_f

    _pl.figure(figsize=(15, 7.5))
    _pl.imshow(_np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    _pl.colorbar()

    _pl.xlabel("Time (s)")
    _pl.ylabel("Frequency (Hz)")
    _pl.xlim([0, timebins-1])
    _pl.ylim([0, 0.2*freqbins])

    xlocs = _np.float32(_np.linspace(0, timebins-1, 5))
    _pl.xticks(xlocs, ["%.02f" % l for l in ((xlocs*samples.size/timebins)+(0.5*binsize))/samplerate])
    ylocs = _np.int16(_np.round(_np.linspace(0, 0.2*freqbins-1, 40)))
    _pl.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        _pl.savefig(plotpath, bbox_inches="tight")
    else:
        _pl.show()

    _pl.clf()
