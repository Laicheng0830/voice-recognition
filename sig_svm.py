#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: eric.lai 2018.11.6
# use SVM dealing with MFCC feature

# best parameters:  {'C': 10, 'gamma': 0.1}
# 0.963531669865643
#              precision    recall  f1-score   support
#
#         0.0       1.00      0.94      0.97        32
#         1.0       1.00      1.00      1.00        32
#         2.0       0.99      1.00      0.99        93
#         3.0       1.00      1.00      1.00       108
#         4.0       0.94      0.94      0.94       143
#         5.0       0.92      0.92      0.92       113
# avg / total       0.96      0.96      0.96       521
# confusion_matrix
# [[ 42   4   0   0   0   4]
#  [  1  49   0   0   0   0]
#  [  0   0 212   3   0   0]
#  [  0   0   0 220   0   0]
#  [  0   0   1   0 189  86]
#  [  0   0   1   1  59 175]]


from MFCC_feature import MFCC
from MFCC_feature import feature_normalization
import matplotlib.pylab as plt
import numpy as np
from pydub import AudioSegment
import os
from sklearn import cross_validation, svm, metrics
from sklearn.metrics import confusion_matrix


def read_file(dir):
    data_feature = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            # print(os.path.join(root, file))
            pcm_path = os.path.join(root, file)
            voice_data = AudioSegment.from_file(
                file=pcm_path,
                sample_width=2,
                frame_rate=16000,
                channels=1
            )
            pcm_data = np.array(voice_data.get_array_of_samples())
            # print(pcm_data)
            mfcc_feature = MFCC(1024,pcm_data,16000,0.02,0.01,"mfcc")
            gun_feature_t = feature_normalization(mfcc_feature,"average")
            data_feature.append(gun_feature_t)
        print(np.shape(data_feature))
        r,c = np.shape(data_feature)
    return data_feature,r,c

def get_feature():
    gun_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/gun'
    blast_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/blast'
    glass_a_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/glass/glass_a'
    glass_p_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/glass/glass_p'
    scream_a_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/scream/scream_a'
    scream_p_dir = 'C:/Users/asus/Desktop/Train_test/TrainDatabase/scream/scream_p'

    # print(np.shape(gun_feataure),np.shape(blast_feature),np.shape(glass_a_feature),np.shape(glass_p_feature),np.shape(scream_a_feature),np.shape(scream_p_feature))

    gun_feataure,r1,c1 = read_file(gun_dir)
    blast_feature,r2,c2 = read_file(blast_dir)
    glass_a_feature,r3,c3 = read_file(glass_a_dir)
    glass_p_feature,r4,c4 = read_file(glass_p_dir)
    scream_a_feature,r5,c5 = read_file(scream_a_dir)
    scream_p_feature,r6,c6 = read_file(scream_p_dir)
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

def get_test_data():
    gun_dir = 'C:/Users/asus/Desktop/Train_test/TestDatabase/gun'
    blast_dir = 'C:/Users/asus/Desktop/Train_test/TestDatabase/blast'
    glass_a_dir = 'C:/Users/asus/Desktop/Train_test/TestDatabase/glass/glass_a'
    glass_p_dir = 'C:/Users/asus/Desktop/Train_test/TestDatabase/glass/glass_p'
    scream_a_dir = 'C:/Users/asus/Desktop/Train_test/TestDatabase/scream/scream_a'
    scream_p_dir = 'C:/Users/asus/Desktop/Train_test/TestDatabase/scream/scream_p'

    gun_feataure, r1, c1 = read_file(gun_dir)
    blast_feature, r2, c2 = read_file(blast_dir)
    glass_a_feature, r3, c3 = read_file(glass_a_dir)
    glass_p_feature, r4, c4 = read_file(glass_p_dir)
    scream_a_feature, r5, c5 = read_file(scream_a_dir)
    scream_p_feature, r6, c6 = read_file(scream_p_dir)
    # tag
    test_tag = np.zeros(r1)
    test_tag = np.hstack((test_tag, np.ones(r2)))
    test_tag = np.hstack((test_tag, np.ones(r3) * 2))
    test_tag = np.hstack((test_tag, np.ones(r4) * 2))
    test_tag = np.hstack((test_tag, np.ones(r5) * 3))
    test_tag = np.hstack((test_tag, np.ones(r6) * 3))
    print(np.shape(test_tag))
    test_data = gun_feataure + blast_feature + glass_a_feature + glass_p_feature + scream_a_feature + scream_p_feature
    print(np.shape(test_data))
    return test_data, test_tag


def svm_train(train_data,train_tag,test_data,test_tag):
    # data_train, data_test, label_train, label_test = cross_validation.train_test_split(train_data, train_tag)
    # data_train, data_test, label_train, label_test = train_data, test_data, train_tag, test_tag
    data_train1, data_test1, label_train1, label_test1 = train_data,  test_data, train_tag, test_tag
    data_all = data_train1+data_test1
    label_train1 = np.hstack((label_train1,label_test1))
    data_train, data_test, label_train, label_test = cross_validation.train_test_split(data_all, label_train1)

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

def predict_svm(train_data,train_tag,test_data,test_tag):
    clf = svm.SVC(gamma=0.1, C=10, decision_function_shape='ovo',class_weight={0:4,1:4})
    clf.fit(train_data, train_tag)
    predict = clf.predict(test_data)
    ac_score = metrics.accuracy_score(test_tag,predict)
    cl_report = metrics.classification_report(test_tag,predict)
    print(ac_score)
    print(cl_report)
    print(confusion_matrix(test_tag,predict))
    print("succeed")


if __name__ == '__main__':
    train_data,train_tag = get_feature()
    test_data,test_tag = get_test_data()
    svm_train(train_data,train_tag,test_data,test_tag)
    # predict_svm(train_data,train_tag,test_data,test_tag)
