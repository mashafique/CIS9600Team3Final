#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 11:23:10 2021

@author: abelroman
"""

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features