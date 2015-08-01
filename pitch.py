import cPickle
import numpy as np
import math
import scipy
import pickle
import theano
from runmlp import MLP, LinearOutputLayer, HiddenLayer

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
            return formants[:2]
        return None

def rms(samples):
        rms = np.sqrt(samples.dot(samples)/samples.size) * 100
        return rms

classifier = None

with open('classifier.pkl') as cfile:
    classifier = cPickle.load(cfile)

predict = theano.function(
    inputs=[classifier.hiddenLayer.input],
    outputs=classifier.linearOutputLayer.y_pred,
)

class FormantFinderNeural(object):

    @classmethod
    def analyze(cls, samples, sample_rate, nlpc=9):
        if rms(samples) < 5: return None

        samples = np.abs(np.fft.rfft(samples)),
        samples /= np.max(samples)
        p = predict(samples)[0]
        return p



def rms(samples):
        rms = np.sqrt(samples.dot(samples)/samples.size) * 100
        return rms


if __name__ == '__main__':
    samples = np.array([ 4939,  5448,  4220,  4620,  3305,  1975,  2028,   476,   -79,
       -1158, -2067, -1664, -2767, -3082, -2486, -2869, -2414, -2148,
       -1754,  -915,  -943,  -243,    68,   211,   598,   283,   434,
         231,  -609, -1021, -1812, -2639, -3478, -1909, -3844, -4823,
       -3008, -3483, -2210, -2723, -1351,   690,  -259,  1742,  2279,
        3322,  5104,  3531,  5363,  6069,  4731,  5318,  4401,  4330,
        3750,  1900,  1892,   802,  -161,  -831, -2095, -1583, -2348,
       -2996, -2442, -2734, -2397, -2126, -1867, -1041,  -911,  -412,
         -26,   200,   734,   609,   612,   617,  -240,  -613, -1426,
       -2230, -3301, -3179, -2166, -5166, -4086, -3041, -3267, -1970,
       -2856,   -15,   485,    28,  2417,  2379,  4376,  4589,  3753,
        6319,  5382,  4809,  5182,  3995,  4283,  2638,  1508,  1493,
         -44,  -523, -1591, -1939, -1594, -2790, -2756, -2173, -2420,
       -2084, -1752, -1338,  -571,  -712,   100,   398,   389,   805,
         436,   686])

    import sys


    print FormantFinderNeural.analyze(samples, 11025)
    print FormantFinder.analyze(samples, 11025)
    sys.exit()

    import time
    t0 = time.time()
    TIMES = 10000
    for i in range(TIMES):
        FormantFinder.analyze(samples, 11025)
    print "Elapsed ms for ", TIMES, " samples: ", 1000*(time.time() - t0)
