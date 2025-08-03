# 📁 distances.py

from scipy.spatial import distance
import numpy as np

def manhattan(v1, v2):
    return np.sum(np.abs(np.array(v1).astype('float') - np.array(v2).astype('float')))

def euclidienne(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    return np.sqrt(np.sum((v1 - v2)**2))

def chebyshev(v1, v2):
    return np.max(np.abs(np.array(v1).astype('float') - np.array(v2).astype('float')))

def canberra(v1, v2):  
    return distance.canberra(v1, v2)
