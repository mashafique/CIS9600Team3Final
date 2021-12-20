#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 00:57:17 2021

@author: abelroman
"""

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from dmba import plotDecisionTree, classificationSummary, regressionSummary


def ca_score_model(train_act_y, train_pred_y, test_act_y, test_pred_y):
    print("Training Set Metrics:")
    print("Accuracy on the train is:",accuracy_score(train_act_y,train_pred_y))
    classificationSummary(train_act_y,train_pred_y)
    print('The Precision on the train is:', precision_score(train_act_y,train_pred_y))
    print('The Recall on the train is:',recall_score(train_act_y,train_pred_y))
    print('The F-Measure on the train is:',f1_score(train_act_y,train_pred_y))
    
    print("\nTesting Set Metrics:")
    print("Accuracy on the test is:",accuracy_score(test_act_y, test_pred_y))
    classificationSummary(test_act_y, test_pred_y) 
    print('The Precision on the test is:', precision_score(test_act_y, test_pred_y))
    print('The Recall on the test is:',recall_score(test_act_y, test_pred_y))
    print('The F-Measure on the test is:',f1_score(test_act_y, test_pred_y))
    
    test_acc = accuracy_score(test_act_y, test_pred_y)
    test_prec = precision_score(test_act_y, test_pred_y)
    test_recall = recall_score(test_act_y, test_pred_y)
    test_f = f1_score(test_act_y, test_pred_y)
    
    list_result = [test_acc, test_prec, test_recall, test_f]
    
    return(list_result)