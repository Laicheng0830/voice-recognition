#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: eric.lai 2018.11.13

# Look for the longest audio in the data and 0 plug in the rest of the audio:1.0
#              precision    recall  f1-score   support
#
#         0.0       1.00      1.00      1.00        11
#         1.0       1.00      1.00      1.00        12
#         2.0       1.00      1.00      1.00       102
#         3.0       1.00      1.00      1.00       134
#
# avg / total       1.00      1.00      1.00       259

# average:1.0
#              precision    recall  f1-score   support
#
#         0.0       1.00      1.00      1.00        12
#         1.0       1.00      1.00      1.00        10
#         2.0       1.00      1.00      1.00       103
#         3.0       1.00      1.00      1.00       134
#
# avg / total       1.00      1.00      1.00       259

# pca:0.9768339768339769
#              precision    recall  f1-score   support
#
#         0.0       1.00      1.00      1.00         8
#         1.0       1.00      0.80      0.89        15
#         2.0       1.00      0.97      0.99       108
#         3.0       0.96      1.00      0.98       128
#
# avg / total       0.98      0.98      0.98       259

# k_means: 0.9922779922779923
#              precision    recall  f1-score   support
#
#         0.0       0.71      1.00      0.83         5
#         1.0       1.00      0.83      0.91        12
#         2.0       1.00      1.00      1.00       122
#         3.0       1.00      1.00      1.00       120
#
# avg / total       0.99      0.99      0.99       259


import numpy as np
import matplotlib.pylab as plt
from MFCC_feature import MFCC,feature_normalization
from sklearn import cross_validation, svm, metrics
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import os

def read_file(dir):
    data_len = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            pcm_path = os.path.join(root, file)
            f = open(pcm_path,'rb')
            str_data = f.read()
            wave_data = np.fromstring(str_data,dtype=np.short)
            data_len.append(len(wave_data))
    return data_len

def count_len():
    gun_len = read_file(gun_dir)
    blast_len = read_file(blast_dir)
    glass_p_len = read_file(glass_p_dir)
    glass_a_len = read_file(glass_a_dir)
    scream_p_len = read_file(scream_p_dir)
    scream_a_len = read_file(scream_a_dir)
    # print([max(gun_len),max(blast_len),max(glass_a_len),max(glass_p_len),max(scream_p_len),max(scream_a_len)])
    # print(max([max(gun_len),max(blast_len),max(glass_a_len),max(glass_p_len),max(scream_p_len),max(scream_a_len)]))
    return max([max(gun_len),max(blast_len),max(glass_a_len),max(glass_p_len),max(scream_p_len),max(scream_a_len)])

def feature_mfcc(dir,max_len):
    data = np.zeros(max_len)
    data_feature = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            pcm_path = os.path.join(root, file)
            f = open(pcm_path,'rb')
            str_data = f.read()
            wave_data = np.fromstring(str_data,dtype=np.short)
            wave_data_len = len(wave_data)
            data[0:wave_data_len] = wave_data
            mfcc_feature = MFCC(1024, data, 16000, 0.02, 0.01, "mfcc")
            # shape_mfcc = feature_normalization(mfcc_feature,'pca')
            l,r = np.shape(mfcc_feature)
            shape_mfcc = np.reshape(mfcc_feature,(l*r))
            data_feature.append(shape_mfcc)
    print(np.shape(data_feature))
    da_r,da_l = np.shape(data_feature)
    return data_feature,da_r,da_l

def get_feature(max_len):
    gun_feataure,r1,c1 = feature_mfcc(gun_dir,max_len)
    blast_feature,r2,c2 = feature_mfcc(blast_dir,max_len)
    glass_a_feature,r3,c3 = feature_mfcc(glass_a_dir,max_len)
    glass_p_feature,r4,c4 = feature_mfcc(glass_p_dir,max_len)
    scream_a_feature,r5,c5 = feature_mfcc(scream_a_dir,max_len)
    scream_p_feature,r6,c6 = feature_mfcc(scream_p_dir,max_len)
    # tag
    train_tag = np.zeros(r1)
    train_tag = np.hstack((train_tag,np.ones(r2)))
    train_tag = np.hstack((train_tag,np.ones(r3)*2))
    train_tag = np.hstack((train_tag,np.ones(r4)*2))
    train_tag = np.hstack((train_tag,np.ones(r5)*3))
    train_tag = np.hstack((train_tag,np.ones(r6)*3))
    print(np.shape(train_tag))
    train_data = gun_feataure+blast_feature+glass_a_feature+glass_p_feature+scream_a_feature+scream_p_feature
    print(np.shape(train_data))
    return train_data,train_tag

def feature_mfcc_t(dir):
    data_feature = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            pcm_path = os.path.join(root, file)
            f = open(pcm_path,'rb')
            str_data = f.read()
            wave_data = np.fromstring(str_data,dtype=np.short)
            mfcc_feature = MFCC(1024, wave_data, 16000, 0.02, 0.01, "mfcc")
            shape_mfcc = feature_normalization(mfcc_feature,'pca')
            data_feature.append(shape_mfcc)
    print(np.shape(data_feature))
    da_r,da_l = np.shape(data_feature)
    return data_feature,da_r,da_l

def get_feature_t():
    gun_feataure,r1,c1 = feature_mfcc_t(gun_dir)
    blast_feature,r2,c2 = feature_mfcc_t(blast_dir)
    glass_a_feature,r3,c3 = feature_mfcc_t(glass_a_dir)
    glass_p_feature,r4,c4 = feature_mfcc_t(glass_p_dir)
    scream_a_feature,r5,c5 = feature_mfcc_t(scream_a_dir)
    scream_p_feature,r6,c6 = feature_mfcc_t(scream_p_dir)
    # tag
    train_tag = np.zeros(r1)
    train_tag = np.hstack((train_tag,np.ones(r2)))
    train_tag = np.hstack((train_tag,np.ones(r3)*2))
    train_tag = np.hstack((train_tag,np.ones(r4)*2))
    train_tag = np.hstack((train_tag,np.ones(r5)*3))
    train_tag = np.hstack((train_tag,np.ones(r6)*3))
    print(np.shape(train_tag))
    train_data = gun_feataure+blast_feature+glass_a_feature+glass_p_feature+scream_a_feature+scream_p_feature
    print(np.shape(train_data))
    return train_data,train_tag

def svm_train(train_data,train_tag):
    # pca = PCA(n_components=5)
    # train_data = pca.fit_transform(train_data)
    # print(pca.explained_variance_ratio_)
    # print(sum(pca.explained_variance_ratio_))

    data_train, data_test, label_train, label_test = cross_validation.train_test_split(train_data, train_tag)
    best_score = 0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            # for each combination of parameters
            clf_t = svm.SVC(gamma=gamma, C=C,decision_function_shape='ovo')
            clf_t.fit(data_train, label_train)
            score = clf_t.score(data_test, label_test)
            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}

    print("best score: ", best_score)
    print("best parameters: ", best_parameters)


    clf = svm.SVC(gamma = best_parameters['gamma'],C = best_parameters['C'],decision_function_shape='ovo')
    clf.fit(data_train,label_train)
    predict = clf.predict(data_test)
    # predicted data
    ac_score = metrics.accuracy_score(label_test, predict)
    # Build test accuracy
    cl_report = metrics.classification_report(label_test, predict)
    # Generate cross validation reports
    print(ac_score)
    # Display data accuracy
    print(cl_report)
    print(confusion_matrix(label_test, predict))



if __name__ == '__main__':
    gun_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/gun'
    blast_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/blast'
    glass_a_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/glass/glass_a'
    glass_p_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/glass/glass_p'
    scream_a_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/scream/scream_a'
    scream_p_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/scream/scream_p'
    # max_len = count_len()
    # train_data, train_tag = get_feature(max_len)
    # svm_train(train_data,train_tag)

    train_data,train_tag = get_feature_t()
    svm_train(train_data,train_tag)
