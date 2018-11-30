#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: eric.lai 2018.10.24

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import dct


def MFCC(window_size, signal, sample_rate, frame_size, shift, feature_model):
    """
    feature extraction of wav use MFCC
    12 cepstral coefficients
    12 delta cepstral coefficients
    12 double delta cepstral coefficients
    1 energy coefficient
    1 delta energy coefficients
    1 double delta energy coefficients

    example : MFCC(1024,pcm_data,16000,0.02,0.01,"mfcc")
    :param window_size:Number of processing points 1024
    :param signal: sig (array)
    :param sample_rate: 16000(Hz)
    :param frame_size: 0.02 (s)
    :param shift: 0.01 (s)
    :param feature_model:mfcc or mfcc_dtm_energy ,If you choose to mfcc return 12 mfcc ,If you choose to mfcc_dtm_energy return 39 dim
    :return: 12 mfcc ,or 12 mfcc add 12 add dtm add 12 dtmm add 3 energy
    """
    # Pre weighted weight
    alpha = 0.97
    new_signal = np.zeros(len(signal))
    for i in range(len(signal)):
        if i is 0:
            new_signal[i] = signal[0]
        else:
            new_signal[i] = signal[i] - alpha * signal[i - 1]

    frame_size = frame_size * sample_rate
    shift = shift * sample_rate
    signal_length = len(new_signal)
    frame_size = round(frame_size)
    shift = round(shift)
    num_frames = int(np.ceil(np.abs(signal_length - frame_size) / shift))

    new_signal_length = num_frames * shift + frame_size
    zero_array = np.zeros((new_signal_length - signal_length))
    saved_signal = np.append(new_signal,zero_array)

    indices = np.zeros(shape=(num_frames, frame_size))
    prevx = 0
    x = frame_size
    for i in range(num_frames):
        indices[i] = np.arange(prevx, x)
        prevx += shift
        x += shift
    indices = indices.astype(np.int32, copy=False)

    frames = saved_signal[indices]
    # hamming windowing
    frames = frames * np.hamming(frame_size)
    # rfft on the frames
    eng_frames = np.absolute(np.fft.rfft(frames, window_size))

    # number of coefficients stored after the real fourier transform
    coefficent_num = eng_frames.shape[-1]
    # find the powe of frames (average of the energy over the wave)
    pow_frames = ((1.0 / window_size) * ((eng_frames) ** 2))
    non_truncated_frames = np.absolute(np.fft.fft(frames, window_size))

    # we did not have duplicates
    mel_freq_ceil = 1127 * np.log(1 + (sample_rate / 2) / 700)  # ceiling
    mel_freq_floor = 0  # the floor
    num_filter = 22  # number of filters
    mel_points = np.linspace(mel_freq_floor, mel_freq_ceil, num_filter + 2)
    # in hertz
    hz_points = 700 * (math.e ** (mel_points / 1127) - 1)
    f = np.floor((window_size + 1) * hz_points / sample_rate)
    # filter banks, because there are in previous steps, half of 256 coefficients are truncated, we kept 129

    # coefficients
    H = np.zeros(shape=(num_filter, coefficent_num))
    for m in range(1, num_filter + 1):
        f_m_left = int(f[m - 1])  # left
        f_m = int(f[m])  # center
        f_m_right = int(f[m + 1])  # right
        denom1 = (f[m] - f[m - 1]) * (f[m + 1] - f[m - 1])
        denom2 = (f[m + 1] - f[m]) * (f[m + 1] - f[m - 1])
        for k in range(f_m_left, f_m):
            H[m - 1, k] = (2 * (k - f[m - 1])) / denom1  # implement the filtering
        for k in range(f_m, f_m_right):
            H[m - 1, k] = (2 * (f[m + 1] - k)) / denom2  # implement the filtering
        # get 1 and other parts are default zeroes
        # zeroes for other ranges
    # perform the ln sums, dot product between each with the power sum of the frame,
    final_segements = np.dot(pow_frames, H.transpose())
    # clean up the silence portions
    final_segements = np.where(final_segements == 0, np.finfo(float).eps, final_segements)
    final_segements = np.log(final_segements)  # take the logs
    # final_segements
    num_ceps = 12  # number of ceps coefficients

    # take the discrete fourier transform along the second axis and keep 12 coefficients
    mfcc = dct(final_segements, type=2, axis=-1, norm='ortho')[:, 1: (num_ceps + 1)]
    # keep cepstral coefficients from 2 to 13, because the first one is real root and disregard that

    # first difference
    mfcc_r,mfcc_l = np.shape(mfcc)
    dtm = np.zeros((mfcc_r,mfcc_l))
    for i in range(2,mfcc_r-2):
        dtm[i,:] = -2*mfcc[i-2,:]-mfcc[i-1,:]+mfcc[i+1,:]+2*mfcc[i+2,:]
    dtm = dtm/3.0
    # second order difference
    dtmm = np.zeros((mfcc_r,mfcc_l))
    for i in range(2,mfcc_r-2):
        dtmm[i,:] = -2*dtm[i-2,:]-dtm[i-1,:]+dtm[i+1,:]+2*dtm[i+2,:]
    dtmm = dtmm/3.0
    # dtm = dct(dtm, type=2, axis=-1, norm='ortho')[:, 1: (num_ceps + 1)]
    # dtmm = dct(dtmm,type=2, axis=-1, norm='ortho')[:, 1: (num_ceps + 1)]
    energy_mfcc = np.zeros((mfcc_r,1))
    energy_dtm = np.zeros((mfcc_r,1))
    energy_dtmm = np.zeros((mfcc_r,1))
    for i in range(mfcc_r):
        energy_mfcc[i] = np.sum(np.square(mfcc[i,:]))
        energy_dtm[i] = np.sum(np.square(dtm[i,:]))
        energy_dtmm[i] = np.sum(np.square(dtmm[i,:]))
    # print(np.shape(dtm),np.shape(dtmm))
    # add all feature ,12 mfcc ,12 first difference,12 second order difference,1 energy mfcc ,1 energy dtm ,1 energy dtmm
    final_mfcc = np.hstack((dtm,dtmm))
    final_mfcc = np.hstack((mfcc,final_mfcc))
    final_mfcc = np.hstack((final_mfcc,energy_mfcc))
    final_mfcc = np.hstack((final_mfcc,energy_dtm))
    final_mfcc = np.hstack((final_mfcc,energy_dtmm))
    final_mfcc_l,final_mfcc_r = np.shape(final_mfcc)
    final_mfcc = final_mfcc[2:final_mfcc_l-2,:]
    # print(np.shape(final_mfcc))
    # return mfcc and logarithimic spec representation
    if feature_model == "mfcc":
        return mfcc
    if feature_model == "mfcc_dtm_energy":
        return final_mfcc
    else:
        print("Method Error")
        exit()



def feature_normalization(feature,method):
    """
    feature normalization
    :param feature: np.array
    :param method: "average","pca","k_means","dtw","hmm"
    :return: normalization feature (np.array)
    """
    feature_r,feature_l = np.shape(feature)
    if method == "average":
        average_feature = []
        for i in range(feature_l):
            average_feature.append(np.average(feature[:, i]))
        return average_feature

    if method == "pca":
        from sklearn.decomposition import PCA
        feature = feature.T
        pca = PCA(n_components=5)
        f_data = pca.fit_transform(feature)
        f_data = f_data.T
        l,r = np.shape(f_data)
        shape_data = np.reshape(f_data, (l * r))
        # print(pca.explained_variance_ratio_)
        # print(sum(pca.explained_variance_ratio_))
        # print(len(shape_data))
        return shape_data

    if method == "k_means":
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=0).fit(feature)
        lab = kmeans.labels_
        feature0 = np.zeros(feature.shape[1])
        feature1 = np.zeros(feature.shape[1])
        feature2 = np.zeros(feature.shape[1])
        count0 = 0
        count1 = 0
        count2 = 0
        for i in range(feature.shape[0]):
            if lab[i] == 0:
                feature0 = feature0+feature[i]
                count0 += 1
            if lab[i] == 1:
                feature1 = feature1+feature[i]
                count1 += 1
            if lab[i] == 2:
                feature2 = feature2+feature[i]
                count2 += 1
        feature0 = feature0/count0
        feature1 = feature1/count1
        feature2 = feature2/count2
        feature0 = np.hstack((feature0,feature1))
        feature0 = np.hstack((feature0,feature2))
        return feature0
    else:
        print("Method Error")
        exit()


if __name__ == '__main__':
    from read_save_wave import read_wave_file
    nchannels, framerate, wave_datas = read_wave_file("lygg_test.wav")
    # plt.plot(wave_datas[1])
    # plt.show()
    mfcc = MFCC(1024,wave_datas[0],framerate,0.02,0.01,"mfcc")
    # print(mfcc)
    mfcc = feature_normalization(mfcc,"k_means")
    print(np.size(mfcc))
    # plt.plot(mfcc)
    # plt.show()
    # from python_speech_features import mfcc
    # mfcc2 = mfcc(wave_datas[0],samplerate=framerate,winlen=0.02,winstep=0.01,numcep=12,
    #      nfilt=40,nfft=1024,preemph=0.97,ceplifter=22)
    # print(mfcc2.shape)
