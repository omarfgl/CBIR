import cv2
import numpy as np

from BiT import bio_taxo
from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick

def glcm(chemin):
    data = cv2.imread(chemin, 0)  # Convertir l'image en NG
    co_occ = graycomatrix(data, [1], [np.pi/2], symmetric=False, normed=True)
    contrast = graycoprops(co_occ, 'contrast')[0, 0]
    dissimilarity = graycoprops(co_occ, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(co_occ, 'homogeneity')[0, 0]
    energy = graycoprops(co_occ, 'energy')[0, 0]
    correlation = graycoprops(co_occ, 'correlation')[0, 0]
    ASM = graycoprops(co_occ, 'ASM')[0, 0]
    features = [contrast, dissimilarity, homogeneity, energy, correlation, ASM]
    return [float(x) for x in features]

def haralick_feat(chemin):
    data = cv2.imread(chemin, 0)  # Convertir l'image en NG
    features = haralick(data).mean(0).tolist()
    return [float(x) for x in features]

def bit_feat(chemin):
    data = cv2.imread(chemin, 0)  # Convertir l'image en NG
    features = bio_taxo(data)
    return [float(x) for x in features]

def concatenation(chemin):
    return glcm(chemin) + haralick_feat(chemin) + bit_feat(chemin)