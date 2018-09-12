"""
@author Semchuk

GLOBAL MODULE NOTES:
@brief This module is supposed to be used as a script with set of fucntions for
contactless PPG peak detection using convo;utional neural networks.

Scheme for data preapration:
1. Get two parallel signals
2. Find out the best offset position (according to the maximum of cross correlation fucntion)
3. Get spike positions in contact PPG signal
4. Window center +- 5% is true position, otherwise it`s false
"""


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


def oneFilePairProcedure(contact, contactless, window, name, prefix):
    per = 0.04
    fullD = []
    targets = 0
    total = 0
    mean_correlation = 0

    low_freq = 0.5
    high_freq = 3
    f_disc = 25
    order = 3

    contact = butter_bandpass_filter(contact, low_freq, high_freq, f_disc, order)
    x1 = range(len(contact))
    contactless = butter_bandpass_filter(contactless, low_freq, high_freq, f_disc, order)
    x2 = range(len(contactless))

    x_p, y_p = peaks(contact, x1, [2, 3, 4], 1)

    peak_counter = 0
    start = 0

    for i in range(start, len(contactless)-window):
        contactless_cut = normalizeSignal(contactless[i:i+window])
        target = 0
        if abs(i+window/2 - x_p[peak_counter])<window/10:
            target = 1
        elif (i+window/2 - x_p[peak_counter])>window/10:
            peak_counter += 1
            if peak_counter>=len(x_p):
                break

        piece_name = name+'_'+str(i)+'_'+prefix
        print(piece_name)
        record = [piece_name]
        for sig in contactless_cut:
            record.append(sig)
        record.append(target)
        fullD.append(record)
        if target == 1:
            targets += 1
        total += 1

    return fullD, targets, total

def write_peak_report(filename, d, width):
    with open(filename, 'w') as csvfile:
        titles = ['name']
        for i in range(width):
            titles.append(i)
        writing = csv.writer(csvfile, delimiter = ',')
        writing.writerow(titles+['Target'])
        i = 0
        for row in d:
            rowToWrite = row
            writing.writerow(rowToWrite)
    return

def oneFolderProcedure(foldName, window):
    seriesData = []
    poses = 0
    over = 0
    for subfold in os.listdir(foldName):
        for less_file in ['Contactless.txt', 'Video.txt']:
            signalCon = readFile(foldName+'/'+subfold+'/'+'Contact.txt')
            signalLess = readFile(foldName+'/'+subfold+'/'+less_file)
            oneFold, a, b = oneFilePairProcedure(signalCon, signalLess, window, subfold, less_file[0:3])
            poses += a
            over += b
            seriesData = seriesData + oneFold

    return seriesData, poses, over

width = 15
results, p, t = oneFolderProcedure('All_measurements', width)
print('positives ', p, ' total ', t)
write_peak_report('peak_data_mining.csv', results, width)
