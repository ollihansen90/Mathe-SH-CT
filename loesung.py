import numpy as np
from scipy.ndimage import rotate

def drehe_bild(bild, winkel):
    return rotate(bild, angle=winkel, reshape=False, order=1)

def sinogrammzeile(bild, winkel):
    hilfsbild = drehe_bild(bild, winkel)
    return np.sum(hilfsbild, axis=0)

def sinogramm(bild, num_winkel):
    output = []
    for winkel in np.linspace(0,180,num_winkel,endpoint=False):
        output.append(sinogrammzeile(bild, winkel))
    return output

def streifenbild(zeile, winkel):
    hilfsbild = np.tile(zeile, (2*len(zeile),1))
    return drehe_bild(hilfsbild, -winkel)[len(zeile)//2:3*len(zeile)//2]

def backprojection(sinogramm):
    output = np.zeros((len(sinogramm[0]), len(sinogramm[0])))
    winkel = np.linspace(0,180,len(sinogramm),endpoint=False)
    for i, zeile in enumerate(sinogramm):
        output += streifenbild(zeile, winkel[i])
    return output/len(sinogramm)

def filter_sinogramm(sinogramm, kernel):
    output = []
    for zeile in sinogramm:
        zeile_fft = np.fft.fft(zeile)
        freqs = np.fft.fftfreq(len(zeile))
        filter = kernel(freqs)
        zeile_fft *= filter
        output.append(np.real(np.fft.ifft(zeile_fft)))
    return output

def ramp_filter(freqs):
    filter = np.abs(freqs)
    return filter

def hamming_filter(freqs):
    filter = np.abs(freqs) * (0.54 + 0.46*np.cos(2*np.pi*freqs))
    return filter

def hann_filter(freqs):
    filter = np.abs(freqs) * (0.5 + 0.5*np.cos(2*np.pi*freqs))
    return filter

def cosine_filter(freqs):
    filter = np.abs(freqs) * np.cos(np.pi*freqs/2)
    return filter

def lanczos_filter(freqs):
    filter = np.abs(freqs) * np.sinc(freqs/2)
    return filter

def parzen_filter(freqs):
    filter = np.abs(freqs) * (1 - np.abs(freqs))
    return filter

def bartlett_filter(freqs):
    filter = np.abs(freqs) * (1 - np.abs(freqs))
    return filter

def blackman_filter(freqs):
    filter = np.abs(freqs) * (0.42 + 0.5*np.cos(2*np.pi*freqs) + 0.08*np.cos(4*np.pi*freqs))
    return filter
