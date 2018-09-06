'''
@author Semchuk

GLOBAL MODULE NOTES:
@brief Here we are gonna create some functions for contactless measurement verification data grabbing.
    For the 27.05.18 the data vector to be verified contains:
    1. Central frequency amplitude (frequency peak with the highest amplitude)
    2. Number of peaks (we have to decide which range of peak width to use).
        Note: Seems that we hve to connect this range with sistole parameters and descritization frequency)
    4. Array of Ai/Ac, where Ai- amplitude of i peak, Ac- amplitude of central frequency (do we really need it?)
    5. Average amplitude of peaks (central freq peak included)
    6. Standart deviation of peaks amplitudes (central freq peak included)
    6. Difference between central freq peak amplitude and average peak amplitude (compared to standart deviation)
@param contaclessSignal - will be used to calculate all the data described above
@param contactSignal - will be used to decide whether contactless measurement can be relied on or not (true or false)
        To decide whether contactless measurement can be relied on:
        1. Campare central frequencies of contact and contactless signal spectral densities
        2. Compare peak time series of contact and contactless measurements

MODULE INTERFACE:
def prepareDataForMLprocedures(folderName)
    Note: folderName-the directory where data is stored. Lets now consider it to be a folder with subfolder each of
            which contains contact measuerement signal and contactless measurement signal.

ATOMIC FUNCTIONS REQUIRED:
def readSignal(fileName) - read file and froms array-type value with signal data
def getPeakTimeSeries(signal, width) - performs peak detections and returns peak time-series array (will be used for signal and spectrum)
def getSignalFFT(signal, descFreq) - performs FFT and returns the spectral density and corresponding frequencies array
def getPeakNumber(peakSeries) - get the count of peaks in signal or spectrum
def getAveragePeakAmplitude(peakSeries) - mean amplitude of all peaks
def getPeakAmlitudeStandartDev(peakSeries) - SD of amplitudes of peaks
def getSpectrumCentralFrequency(spectrum) - frequency with maximum peak amplitude in spectrum

MAIN PROCEDURES SEQUENCE:
1. Read both signals
2. Divide it into part(must be equal and len = 2^n)
    for each part:
    2.1. Perform FFT for both and find central frequencies for both in the range of [0.5 - 4.0] (can be varied)
    2.2. Compare them
    2.3. Perform peak detection for both
    2.4. Do smth with peak series (!!!!!)
    2.5. Get target value (if the measurement can be relied on or not)
    2.6. Get peakNumber, averagePeakAmplitude, peak amplitude AD and etc. for contactless
    2.7. Add it to huge array which will be stored in the file
3. After all data calculations are performed fill the file

GLOBAL DATA ARRAY STRUCTURE:
What do we need to store:
 part Name|Central frequency | Number of peaks | Average peak amplitude | Peak Amplitude SD| CentarFreq and SD comparison| Target

So the global data array can be:
GDA = [[a,b]]
    where a - partName (e.g. FileName+SignalPartNumber)
          b - dict with (centralFreq, peakNum etc.) keys
'''


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

def oneContactlessSignalPiece(signal, period):
    d = {}
    spectra, freqs = get_fourier_result(signal, period)
    spectra_n = normalizeSpectrum(spectra)
    pX, pY = simple_peaks(spectra_n, freqs, np.arange(1, 2))
    a, b = getSpectrumCentralFrequencyAndAmp(pX, pY)
    dixi = getDixonCriteria(pY, b)
    d['central_freq'], d['central_freq_amp'] = a, b
    d['dixi'] = dixi
    d['peak_count'] = getPeakCount(pX, pY)
    d['peak_mean'], d['peak_SD'] = getMeanValAndSD(pY)
    d['sig_mean'], d['sig_SD'] = getMeanValAndSD(signal)
    d['peak_matters_SD'] = abs(d['central_freq_amp']-d['peak_mean'])/d['peak_SD']
    d['peak_matters_Int'] = d['central_freq_amp'] / getSignalIntegral(spectra_n, freqs, 0, 3)
    return d, spectra_n, pX, pY

def oneContactSignalPiece(signal, period):
    d = {}
    spectra, freqs = get_fourier_result(signal, period)
    pX, pY = simple_peaks(spectra, freqs, np.arange(1, 3))
    Fcentr, FcentrAmp = getSpectrumCentralFrequencyAndAmp(pX, pY)
    return Fcentr, spectra, freqs, pX, pY

def oneFilePairProcedure(contact, contactless, window, name, prefix):
    per = 0.04
    fullD = []
    targets = 0
    total = 0
    mean_correlation = 0
    for i in range(min(len(contact), len(contactless))-window):
        contact_cut = normalizeSignal(contact[i:i+window])
        # contact_wave = get_wavelet(contact_cut)
        contactless_cut = normalizeSignal(contactless[i:i+window])
        # contactless_wave = get_wavelet(contactless_cut)
        # wave_matters = np.sum(np.multiply(contactless_wave, contact_wave))/np.sum(np.multiply(contact_wave, contact_wave))
        dictLess, less_sp, p_x_l, p_y_l = oneContactlessSignalPiece(contactless_cut, per)
        if (dictLess == 0):
            continue
        contactHr, contact_sp, fr, p_x, p_y = oneContactSignalPiece(contact_cut, per)
        peak_res = compare_by_peaks(contact_cut, range(len(contact_cut)), contactless_cut, range(len(contactless_cut)))
        peak_comp = peak_res[0]
        dictLess['signal_peaks'] = peak_res[2]
        # plt.plot(fr, contact_sp)
        # plt.plot(fr , less_sp)
        # plt.scatter(p_x, p_y, color='red')
        # plt.scatter(p_x_l, p_y_l, color='black')
        # plt.show()
        # correlation = getCorrelationFuncMax(contactless_cut, contact_cut)/getCorrelationFuncMax(contact_cut, contact_cut)
        #mean_correlation += correlation
        target = int(abs(dictLess['central_freq']-contactHr) <= 0.015)*peak_comp
        if target > 0.75:
            target = 1
        else:
            target = 0
        dictLess['target'] = target
        dictLess['contact_freq'] = contactHr
        piece_name = name+'_'+str(i)+'_'+prefix
        print(piece_name)
        fullD.append([piece_name, dictLess])
        if target == 1:
            targets += 1
        total += 1

    return fullD, targets, total#, mean_correlation

def write_q_pulse_report(filename, d):
    keys = ['central_freq', 'contact_freq', 'central_freq_amp', 'peak_count', 'peak_mean', 'peak_SD', 'peak_matters_SD', 'peak_matters_Int', 'dixi', 'signal_peaks', 'sig_mean','sig_SD', 'target']
    with open(filename, 'w') as csvfile:
        writing = csv.writer(csvfile, delimiter = ',')
        writing.writerow(['Name']+keys)
        for row in d:
            rowToWrite = [row[1][key] for key in keys]
            rowToWrite = [row[0]] + rowToWrite
            writing.writerow(rowToWrite)
    return


def oneFolderProcedure(foldName):
    seriesData = []
    poses = 0
    over = 0
    m_c = 0
    for subfold in os.listdir(foldName):
        for less_file in ['Contactless.txt', 'Video.txt']:
            signalCon = readFile(foldName+'/'+subfold+'/'+'Contact.txt')
            signalLess = readFile(foldName+'/'+subfold+'/'+less_file)
            oneFold, a, b = oneFilePairProcedure(signalCon, signalLess, 256, subfold, less_file[0:3])
            poses += a
            over += b
            # m_c += cor
            seriesData = seriesData + oneFold

    return seriesData, poses, over#, m_c

results, p, t = oneFolderProcedure('All_measurements')
print('positives ', p, ' total ', t)
#print("mean correlation ", c/t)
write_q_pulse_report('total_data_mining_vid.csv', results)
