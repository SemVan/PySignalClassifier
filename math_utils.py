

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal as sp_sig
import pywt
from scipy.signal import butter, lfilter, lfilter_zi, welch
from scipy.signal import find_peaks_cwt
from scipy.signal import correlate
from scipy.signal import cwt
import math
import csv
from math_utils import *

def readFile(fileName):
    data = [[], []]
    with open(fileName, 'r') as f:
        for row in f:
            rowList = row.split(",")
            data[0].append(rowList[0])
            if len(rowList)==2:
                data[1].append(float(rowList[1]))
            else:
                data[1].append(float(rowList[2])/(float(rowList[1])+float(rowList[3])))
    return (get_y_reverse_signal(np.asarray(data[1])))



def get_y_reverse_signal(sig):
    sig_max = np.max(sig)
    new_sig = sig_max-sig
    return new_sig

def get_fourier_result (signal, period):
    complex_four = np.fft.fft(signal)
    spectra = np.absolute(complex_four)
    freqs = []
    for i in range(len(signal)):
        freqs.append(1/(period*len(signal))*i)
    return spectra, freqs

def get_wavelet(s):
    fs = 25
    period = 0.04
    widths = np.arange(1, 40, 1)
    cwtmatr, freqs = pywt.cwt(s, widths, 'mexh', period)
    cwtmatr = cwtmatr[:20,:]
    freqs = freqs[:20]
    # print(np.shape(cwtmatr))
    # plt.imshow(cwtmatr, extent=[0/fs, len(s)/fs, np.max(freqs), np.min(freqs)], cmap='inferno', aspect='auto',
    # vmax = (cwtmatr).max(), vmin=(cwtmatr).min())
    # plt.title("Mexican hat transform")
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    return cwtmatr

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, z = lfilter(b, a, data, zi=zi*data[0])
    return y

def normalizeSpectrum(spectra):
    nSp = spectra/spectra[0]
    return nSp

def normalizeSignal(signal):
    m = max(signal)
    new_sig = signal/m
    return new_sig

def findNearestElemInTheList(lst, elem):
    diffs = [abs(lst_el-elem) for lst_el in lst]
    return np.argmin(diffs)

def getSignalIntegral(s_y, s_x, start_x, stop_x):
    str_idx = findNearestElemInTheList(s_x, start_x)
    stp_idx = findNearestElemInTheList(s_x, stop_x)
    return np.sum(s_y[str_idx:stp_idx])

def getDixonCriteria(amps, central):
    amps = sorted(amps)
    cen_ndx = findNearestElemInTheList(amps, central)
    dix = -1
    if cen_ndx != 0:
        dix = (amps[cen_ndx] - amps[cen_ndx-1])/(amps[cen_ndx] - amps[0])

    return dix

def peaks(signal, xAx, dotSize, num):
    peaks = find_peaks_cwt(signal, dotSize)
    y = np.asarray([signal[i] for i in peaks])
    # y = [num for i in peaks]
    x = np.asarray([xAx[i] for i in peaks])
    return x, y

def delete_peak_doubling(peakX, peakY):
    truePeaksX = []
    truePeaksY = []
    prev = peakX[0]
    aver = prev
    averY = [peakY[0]]
    count = 1
    for i in range(1, len(peakX)):
        if peakX[i]-prev<10:
            aver +=peakX[i]
            averY.append(peakY[i])
            count += 1
        else:
            truePeaksX.append(aver/count)
            truePeaksY.append(max(averY))
            prev = peakX[i]
            aver = prev
            averY = [peakY[i]]
            count = 1

    return truePeaksX, truePeaksY


def compare_by_peaks(y1, x1, y2, x2):
    y1 = butter_bandpass_filter(y1, 0.5, 3, 25, 3)
    x1 = range(len(y1))
    y2 = butter_bandpass_filter(y2, 0.5, 3, 25, 3)
    x2 = range(len(y2))
    x1_p, y1_p = peaks(y1, x1, [2, 3, 4], 1)
    # x1_p, y1_p = delete_peak_doubling(x1_p, y1_p)
    x2_p, y2_p = peaks(y2, x2, [2, 3, 4], 1)
    # x2_p, y2_p = delete_peak_doubling(x2_p, y2_p)

    plt.scatter(x1_p, y1_p, color='red')
    plt.plot(x1, y1)
    plt.scatter(x2_p, y2_p, color='blue')
    plt.plot(x2, y2)
    plt.show()
    if len(x1_p)>2 and len(x2_p)>2:
        if len(x1_p) == len(x2_p):
            # return [True,len(x1_p), len(x2_p)]
            d_max = 0
            p1 = x1_p[1:-1]
            p2 = x2_p[1:-1]
            for i in range(len(p1)):
                d = abs(p1[i]-p2[i])
                if d>d_max:
                    d_max = d
            if d_max<= 10:
                return [True,len(x1_p), len(x2_p)]
    return [False, len(x1_p), len(x2_p)]

def simple_peaks(signal, xAx, dotSize):
    x = []
    y = []
    for i in range(1, len(signal)-1):
        if (signal[i]>signal[i+1] and signal[i]>signal[i-1]):
            x.append(xAx[i])
            y.append(signal[i])
    return x,y

def getPeakCount(peaks_x, peaks_y):
    peaks_cut_x = [peaks_x[i] for i in range(len(peaks_x)) if (peaks_x[i] > 0.5 and peaks_x[i] < 3.0)]
    return len(peaks_cut_x)

def getMeanValAndSD(sig):
    return np.mean(sig), np.std(sig)

def getSpectrumCentralFrequencyAndAmp(peakX, peakY):
    z = list(zip(peakY, peakX))
    peaks_cut = [z[i][0] for i in range(len(z)) if (z[i][1] > 0.5 and z[i][1] < 3.0)]
    peaks_cut_x = [z[i][1] for i in range(len(z)) if (z[i][1] > 0.5 and z[i][1] < 3.0)]
    if len(peaks_cut) == 0:
        return -1, -1
    maxY = np.max(peaks_cut)
    maxX = peaks_cut_x[np.argmax(peaks_cut)]
    #print("freq max ", maxX)
    return maxX, maxY

def getPearsonCorrelation(signal1, signal2):
    return np.corrcoef(signal1,signal2)[0][1]

def getCorrelationFuncMax(signal1, signal2):
    minlen = min(len(signal1), len(signal2))
    corrF = correlate(signal1[0:minlen], signal2[0:minlen], 'full', 'direct')
    return max(corrF)
