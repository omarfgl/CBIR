# ==============================
# 📁 extraction/extraction.py
# ==============================
# ✅ Extraction récursive RGB avec logs clairs (corrige les dossiers non lus)

import os
import numpy as np
from descripteurs import glcm, haralick_feat, bit_feat, concatenation

chemin_dossier = 'dataset/'
print("📁 Début de l'extraction des caractéristiques...")
print(f"📂 Dataset trouvé : {chemin_dossier}\n")

signatures_glcm = []
signatures_haralick = []
signatures_bit = []
signatures_concat = []

for root, dirs, files in os.walk(chemin_dossier):
    for nom_fichier in files:
        chemin_image = os.path.join(root, nom_fichier)
        try:
            print(f"🖼️ Traitement : {chemin_image}")
            glcm_vec = glcm(chemin_image) + [chemin_image]
            har_vec = haralick_feat(chemin_image) + [chemin_image]
            bit_vec = bit_feat(chemin_image) + [chemin_image]
            concat_vec = concatenation(chemin_image) + [chemin_image]

            signatures_glcm.append(glcm_vec)
            signatures_haralick.append(har_vec)
            signatures_bit.append(bit_vec)
            signatures_concat.append(concat_vec)

        except Exception as e:
            print(f"❌ Erreur {chemin_image}: {e}")

# Sauvegarde dans des fichiers .npy
np.save('GLCM_RGB.npy', np.array(signatures_glcm, dtype=object))
np.save('Haralick_RGB.npy', np.array(signatures_haralick, dtype=object))
np.save('BiT_RGB.npy', np.array(signatures_bit, dtype=object))
np.save('Concat_RGB.npy', np.array(signatures_concat, dtype=object))

print("\n✅ Extraction terminée et fichiers .npy sauvegardés")
