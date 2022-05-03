import numpy as np
import matplotlib.pyplot as plt

from manipulation_image import *
from retrouver_feature import *

taille = 24 # taille minimale des carrés balayés (24 x 24 pour la méthode de Viola-Jones)
pas = 2 # pas entre chaque sous-rectangles dans les carrés balayés
increment = 1.2 # ce par quoi on multiplie itérativement la taille des sous-rectangles
features = retrouver_features(taille,pas,increment) # toutes les features avec ces paramètres

#f = np.load("fonction_detection-5.npy") # la fonction de détection
# on ne s'intéresse ici qu'au numéro des features de la focntion de détection
#f = np.array(f[:,0],dtype=np.int32)
f = [13302]

image = np.load("test-visage.npy")[777]

for numero in f :

    t = image.copy()

    type,x,y,l,h = features[numero]

    if type == 1 :
        t[y:y+h,x:x+l] = 0
        t[y+h:y+2*h,x:x+l] = 255

    if type == 2 :
        t[y:y+h,x:x+l] = 0
        t[y:y+h,x+l:x+2*l] = 255
    
    if type == 3 :
        t[y:y+h,x:x+l] = 0
        t[y+h:y+2*h,x+l:x+2*l] = 0
        t[y+h:y+2*h,x:x+l] = 255
        t[y:y+h,x+l:x+2*l] = 255

    if type == 4 :
        t[y:y+h,x:x+l] = 0
        t[y:y+h,x+l:x+2*l] = 255
        t[y:y+h,x+2*l:x+3*l] = 0

    affiche (t)