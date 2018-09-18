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
import time
from math_utils import *



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
    small_window = 128

    for i in range(0, min(len(contact), len(contactless))-window, 10):
        contact_cut = normalizeSignal(contact[i:i+window])
        contactless_cut = normalizeSignal(contactless[i:i+window])
        contact_cut = butter_bandpass_filter(contact_cut, 0.5, 3, 25, 3)
        contactless_cut = butter_bandpass_filter(contactless_cut, 0.5, 3, 25, 3)

        offset = get_offset(contactless_cut, contact_cut)
        if offset>0:
            contactless_cut = contactless_cut[:-offset]
            contact_cut = contact_cut[offset:]

        av = 0
        k = 0
        for j in range(0, min(len(contact_cut), len(contactless_cut))-small_window, 5):
            s = contact_cut[j:j+small_window]
            s_l = contactless_cut[j:j+small_window]
            k_res = compare_by_peaks(s, range(len(s)), s_l, range(len(s_l)))
            if k_res[0]==True:
                av += 1
            k += 1


        if (av/k>0.5):
            target = 1
        else:
            target = -1
        dictLess, less_sp, p_x_l, p_y_l = oneContactlessSignalPiece(contactless_cut, per)
        if (dictLess == 0):
            continue
        contactHr, contact_sp, fr, p_x, p_y = oneContactSignalPiece(contact_cut, per)
        peak_res = compare_by_peaks(contact_cut, range(len(contact_cut)), contactless_cut, range(len(contactless_cut)))
        peak_comp = peak_res[0]
        p_mean, p_std, p_maxi, p_mini = get_signal_peaks_metrics(peak_res[4])
        dictLess['signal_peaks'] = peak_res[2]
        dictLess['p_mean'] = p_mean
        dictLess['p_std'] = p_std
        dictLess['p_maxi'] = p_maxi
        dictLess['p_mini'] = p_mini

        # plt.plot(range(len(peak_res[8])), peak_res[8])
        # plt.plot(range(len(peak_res[7])), peak_res[7])
        # plt.scatter(peak_res[4], peak_res[6], color = 'red')
        # plt.scatter(peak_res[3], peak_res[5], color = 'black')
        # plt.show()

        # target = peak_comp
        # if target > 0.75:
        #     target = 1
        # else:
        #     target = -1
        dictLess['target'] = target
        dictLess['contact_freq'] = contactHr
        piece_name = name+'_'+str(i)+'_'+prefix
        print(piece_name)
        print(dictLess['target'])
        fullD.append([piece_name, dictLess])
        if target == 1:
            targets += 1
        total += 1

    return fullD, targets, total#, mean_correlation

def write_q_pulse_report(filename, d):
    keys = ['central_freq', 'contact_freq', 'central_freq_amp', 'peak_count',
     'peak_mean', 'peak_SD', 'peak_matters_SD', 'peak_matters_Int', 'dixi', 'signal_peaks', 'sig_mean','sig_SD','p_mean', 'p_std', 'p_maxi', 'p_mini', 'target']
    with open(filename, 'w') as csvfile:
        writing = csv.writer(csvfile, delimiter = ',')
        writing.writerow(['Name']+keys)
        for row in d:
            rowToWrite = [row[1][key] for key in keys]
            rowToWrite = [row[0]] + rowToWrite
            writing.writerow(rowToWrite)
    return


def vizualize(sig1, sig2):
    sig1 = normalizeSignal(sig1)
    sig2 = normalizeSignal(sig2)
    plt.plot(range(len(sig1)), sig1)
    plt.plot(range(len(sig2)), sig2)
    plt.show()
    return

def oneFolderProcedure(foldName):
    seriesData = []
    poses = 0
    over = 0
    m_c = 0
    for subfold in os.listdir(foldName):
        # for less_file in ['Video.txt', 'Contactless.txt']:
        for less_file in ['Video.txt']:
            signalCon = readFile(foldName+'/'+subfold+'/'+'Contact.txt')
            signalLess = readFile(foldName+'/'+subfold+'/'+less_file)
            # vizualize(signalCon, signalLess)
            # signalCon = butter_bandpass_filter(signalCon, 0.5, 3, 25, 3)
            # signalLess = butter_bandpass_filter(signalLess, 0.5, 3, 25, 3)
            # vizualize(signalCon, signalLess)
            oneFold, a, b = oneFilePairProcedure(signalCon, signalLess, 256, subfold, less_file[0:3])
            poses += a
            over += b
            seriesData = seriesData + oneFold

    return seriesData, poses, over

results, p, t = oneFolderProcedure('Exp_measurements')
print('positives ', p, ' total ', t)
#print("mean correlation ", c/t)
write_q_pulse_report('total_data_mining_dev.csv', results)
