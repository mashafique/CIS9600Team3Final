#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 00:56:44 2021

@author: abelroman
"""

from dmba import plotDecisionTree, classificationSummary, regressionSummary
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score


def get_NBmetrics(test_y_est,test_y):


    TP = 0.00000000001
    TN = 0.00000000001
    FP = 0.00000000001
    FN = 0.00000000001

    for i in range(len(test_y)):
        if str(test_y_est[i])=="True" and test_y_est[i]==test_y[i]:
            TP = TP+1
        elif str(test_y_est[i])=="False" and test_y_est[i]==test_y[i]:
            TN = TN+1
        elif str(test_y_est[i])=="True" and test_y_est[i]!=test_y[i]:
            FP = FP+1
        else:
            FN = FN+1     
        
    # a. Accuracy Rate
    ACC = (TP+TN)/(TP+TN+FP+FN)
    #print("The accuracy rate is", round(ACC,4))

    # b. Precision
    PRE = TP/(TP+FP)
    #print("The precision is", round(PRE,4))

    # c. Recall
    REC = TP/(TP+FN)
    #print("The recall is", round(REC,4))

    # d. F-measure
    Fscore = 2*PRE*REC/(PRE+REC)
    #print("The F-measure is", round(Fscore,4))
    
    return ACC, PRE, REC, Fscore