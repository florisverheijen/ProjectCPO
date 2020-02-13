# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:55:48 2018

@author: s152040
"""



def faculteit(n):
    result = n
    while n > 1:
        result = result*(n-1)
        
        n = n-1
    print(result)
    return result

faculteit(3)
        