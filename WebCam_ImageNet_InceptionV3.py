##############################################################################
# Mise en oeuvre d'un réseau entrainé sur la base IMAGENET pour reconnaitre  #
# en temps réel des objets sur les images de la Webcam de l'ordinateur       #
# --> utilisation du module keras.applications                               #
# --> utilisation du réseau InceptionV3 (https://arxiv.org/abs/1512.00567)   #
##############################################################################

# Importation des modules
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv

#----------------------------------------------------------------------------#
# Chargement du réseau entrainé                                              #
#----------------------------------------------------------------------------#
Reseau_InceptionV3 = tf.keras.applications.InceptionV3()

#----------------------------------------------------------------------------#
# Ouverture du flux video de la Webcam                                       #
#----------------------------------------------------------------------------#
FluxVideo = cv.VideoCapture(0)
if not FluxVideo.isOpened():
  print("Impossible de récupérer le flux video de la caméra!")
  exit()

#----------------------------------------------------------------------------#
# Boucle de lecture d'image avec reconnaissance                              #
#----------------------------------------------------------------------------#
while (True):
  # Lecture d'une image --> format ndarray avec codage couleurs BGR
  ReadOK, ImgBGR = FluxVideo.read()

  # Vérification de la bonne lecture de l'image
  if not ReadOK:
    print(" Problème dans le flux vidéo de la caméra!")
    break

  # Conversion de l'image au format RGB
  ImgRGB = cv.cvtColor(ImgBGR, cv.COLOR_BGR2RGB)

  # Affichage imposé à 640x480 (pour être indépendant de la Webcam)
  ImgBGR_640x480 = cv.resize(ImgBGR,(640,480))

  # Ajout d'un texte dans l'image
  cv.putText(ImgBGR_640x480,
             "SPACE -> identifier    ESC -> sortir", # texte à afficher
             (10,460),                # position d'affichage
             cv.FONT_HERSHEY_SIMPLEX, # police de caractères
             1,                       # taille des caractères
             (0,0,255),               # couleur du texte
             3,                       # largeur du trait
             cv.LINE_4)               # type de trait

  # Affichage de l'image dans une fenêtre
  cv.imshow('Retour video Webcam',ImgBGR_640x480)

  # Lecture au "vol" d'une touche au clavier
  key = cv.waitKey(1)
  # 32: touche SPACE ==> lancement d'une reconnaissance
  if key == 32:
    # Instant de début des traitements (pour le calcul de la durée d'exécution)
    t1 = cv.getTickCount()
    # Re-dimensionnement de l'image au format Input de InceptionV3:299x299
    ImgRGB_299 = cv.resize(ImgRGB,(299,299))
    # Transformation en une liste (format pour 'predict') avec une seule image
    ImgRGB_299_dim4 = np.expand_dims(ImgRGB_299,axis=0)
    # Mise en forme des données dans le format des entrées attendues du réseau
    # (normalisation,...)
    Entrees = tf.keras.applications.inception_v3.preprocess_input(ImgRGB_299_dim4)
    # Calcul des sorties du réseau
    Sorties = Reseau_InceptionV3.predict(Entrees)
    # Transformation des sorties dans un format facilement exploitable
    # avec pour chaque image une liste de 3 valeurs :
    # [0] -> WordNet = identifiant associé à chaque catégorie
    # [1] -> Synset  = nom associé à chaque catégorie
    # [2] -> taux de confiance de la reconnaissance
    Conclusions = tf.keras.applications.inception_v3.decode_predictions(Sorties,top=5)
    # Instant de fin des traitements et affichage de la durée d'exécution
    t2 = cv.getTickCount()
    print("\nConclusions du réseau InceptionV3:")
    print(" ->Temps calcul = {:.2f} seconde".format((t2-t1)/cv.getTickFrequency()))
    # Affichage des 'top 5' identifiés par le réseau
    plt.imshow(ImgRGB_299),plt.show()
    for i in range(5):
      synset_ID = Conclusions[0][i][1]
      confiance = '{:.2%}'.format(Conclusions[0][i][2])
      print(synset_ID, confiance)
  # 27: touche ESC ==> sortie du mode video
  elif key == 27:
    # Libération du flux video de la caméra
    FluxVideo.release()
    # Fermeture de toutes les fenêtres ouvertes
    cv.destroyAllWindows()
    break