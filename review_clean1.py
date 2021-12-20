#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:45:38 2021

@author: abelroman
"""

# cleans review data
def review_clean(review): 

    lower = review.str.lower()
    pattern_remove = lower.str.replace("&#039;", "")
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
    
    return dataframe