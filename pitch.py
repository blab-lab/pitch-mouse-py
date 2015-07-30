import numpy as np
import math
import scipy
import pickle


class PitchFinder(object):

    @classmethod
    def analyze(cls, samples, sample_rate, min_pitch=50, max_pitch=500):
        if (rms(samples) < 15): return 0
        to_skip = sample_rate / max_pitch
        acorr =  np.correlate(samples, samples, mode='same')
        acorr = acorr[len(acorr)/2:]
        max_freq_steps = to_skip + np.argmax(acorr[to_skip:])
        ret = sample_rate / max_freq_steps
        return ret

class RMSFinder(object):

    @classmethod
    def analyze(cls, samples, sample_rate, min_pitch=50, max_pitch=1500):
        pass


import lpc
from scipy.signal import lfilter

class FormantFinder(object):

    @classmethod
    def analyze(cls, samples, sample_rate, nlpc=9):

        if (rms(samples) < 5): return 0

        """
        duration = len(samples) * 1.0 / sample_rate  * 1000
        if duration < 30:
            return 0
        """

        shorts = [int(x / 1.0 * 32768) for x in samples]
        #print "samples", len(shorts), sample_rate, max(shorts)

        shorts = shorts * np.hamming(len(shorts))
        shorts = lfilter([1], [1, .63], shorts) # pre-emphasize that stuff
        A = lpc.lpc_ref(shorts, nlpc)
        roots = np.roots(A)
        roots = roots[np.imag(roots)>=0]
        #print len(roots), "roots"
        bw = -.5 * sample_rate / 2 / math.pi * np.log(np.abs(roots))
        angs = np.arctan2(np.imag(roots), np.real(roots))
        angs = angs * [sample_rate / 2 / math.pi]
        #angs = np.absolute(angs)
        order = np.argsort(angs)

        formants = []
        for i in order:
            if angs[i] > 90 and bw[i] < 400:
                formants.append(angs[i])
               
        #print "debug", formants
        if len(formants) > 0:
            """
            if formants[0] > 250 and formants[0] < 600:
                freqz= scipy.signal.freqz(1, A, 512)
                #print np.abs(freqz)
                pickle.dump( freqz, open( "freqz.p", "wb" ) )
            """
            return formants[0]
        return 0

def rms(samples):
        rms = np.sqrt(samples.dot(samples)/samples.size) * 100
        return rms

