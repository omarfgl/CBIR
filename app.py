# ==============================
# 📁 app.py (Interface Web CBIR)
# ==============================
# ✅ Interface utilisateur avec Streamlit (conforme au projet)

import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

from descripteurs import glcm, haralick_feat, bit_feat, concatenation
from distances import euclidienne, manhattan, chebyshev, canberra

import json

st.write("✅ Interface chargée.")

# 🔐 Chargement des utilisateurs
with open("users.json", "r") as f:
    users = json.load(f)

# 🔐 Gestion de la session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Connexion requise")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Connexion réussie")
            st.rerun()
        else:
            st.error("❌ Identifiants incorrects")
    st.stop()

# =========================
# 📌 Dictionnaires de choix
# =========================
descripteur_options = {
    'GLCM_RGB': ('GLCM_RGB.npy', glcm),
    'Haralick_RGB': ('Haralick_RGB.npy', haralick_feat),
    'BiT_RGB': ('BiT_RGB.npy', bit_feat),
    'Concaténation': ('Concat_RGB.npy', concatenation)
}

distance_options = {
    'Euclidienne': euclidienne,
    'Manhattan': manhattan,
    'Tchebychev': chebyshev,
    'Canberra': canberra
}

# ===================
# 🎯 Interface Streamlit
# ===================
st.title("🔎 CBIR - Recherche d'images par contenu (RGB)")

uploaded_file = st.file_uploader("Téléverser une image", type=["png", "jpg", "jpeg"])

# 🎛️ Choix utilisateur
col1, col2 = st.columns(2)
with col1:
    descripteur_nom = st.selectbox("Choisir le descripteur", list(descripteur_options.keys()))
with col2:
    distance_nom = st.selectbox("Choisir la mesure de distance", list(distance_options.keys()))
n_resultats = st.slider("Nombre d'images similaires à afficher", 1, 10, 5)

# =====================
# 📥 Image et requête
# =====================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    st.image(image, caption="Image requête", use_container_width=True)

    # Sauvegarde temporaire
    chemin_temp = "images/temp_image.jpg"
    Image.fromarray(image_np).save(chemin_temp)

    # Extraction descripteur
    fonction_vecteur = descripteur_options[descripteur_nom][1]
    vecteur_requete = fonction_vecteur(chemin_temp)

    # Chargement des signatures
    signatures_path = descripteur_options[descripteur_nom][0]
    signatures = np.load(signatures_path, allow_pickle=True)

    # Calcul de distance
    distance_fct = distance_options[distance_nom]
    resultats = []
    for ligne in signatures:
        vecteur, chemin = ligne[:-1], ligne[-1]
        dist = distance_fct(vecteur_requete, vecteur)
        resultats.append((chemin, dist))

    # Tri + affichage
    resultats.sort(key=lambda x: x[1])
    top = resultats[:n_resultats]

    st.markdown("---")
    st.subheader("📸 Résultats similaires")
    col_images = st.columns(n_resultats)

    for i, (chemin, dist) in enumerate(top):
        with col_images[i]:
            img = cv2.imread(chemin)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, caption=f"Distance: {dist:.3f}", use_container_width=True)
