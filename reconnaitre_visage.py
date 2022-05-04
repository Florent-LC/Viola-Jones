import numpy as np
from scipy.ndimage.interpolation import zoom
from time import time

from manipulation_image import *
from creation_features import *
from entrainement_features import *
from retrouver_feature import *


def valeur_feature (feature,t) :
    """Entree : feature : tuple -> coordonnées des points avec les largeurs à soustraire que l'on associe à un numéro de feature
                t : numpy array (2) -> une image 24x24

    Sortie : int -> la valeur de la feature (feature) pour l''image t"""

    t = t.cumsum(axis=0).cumsum(axis=1) # on priviligiera ici la fonction numpy cumsum bien plus rapide 
    #que la fonction image_integrale car incluant des parallélisations
    t = np.array (t,dtype=np.int64)

    type,x,y,l,h = feature

    if type == 1 :
        a1,_ = aire(t,x,y,l,h)
        a2,_ = aire(t,x,y+h,l,h)
        return (a1-a2)

    if type == 2 :
        a1,_ = aire(t,x,y,l,h)
        a2,_ = aire(t,x+l,y,l,h)
        return (a1-a2)
    
    if type == 3 :
        a1,_ = aire(t,x,y,l,h)
        a2,_ = aire(t,x+l,y+h,l,h)
        a3,_ = aire(t,x+l,y,l,h)
        a4,_ = aire(t,x,y+h,l,h)
        return (a1 + a2 - a3 - a4)

    if type == 4 :
        a1,_ = aire(t,x,y,l,h)
        a2,_ = aire(t,x+l,y,l,h)
        a3,_ = aire(t,x+2*l,y,l,h)
        return (a1 - a2 + a3)

    return ()


def monolithique (t,f,features,condition) :
    """Entree : t : numpy array (2) -> une image 24x24
                f : numpy array (2) -> la fonction de détection monolithique
                features : list -> liste des coordonnées des points à soustraire avec les largeurs que l'on associe à un numéro de feature
                condition : float -> si le résultat de la fonction de détection est au-dessus de cette valeur on considère que c'est un visage

    Sortie : bool -> True si l'image est considérée comme une image par la fonction de
    détection
    """

    poids = 0

    for numero,polarite,seuil,_,alpha in f :
        numero = int (numero)
        if polarite == 1 :
            if valeur_feature (features[numero],t) < seuil :
                poids += alpha
        else :
            if valeur_feature (features[numero],t) > seuil :
                poids += alpha

        if poids > condition : 
            return True

    return poids > condition


def cascade (t,f,features,conditions) :
    """Entree : t : numpy array (2) -> une image 24x24
                f : list -> la fonction de détection sous forme de cascade : chaque élément de la liste est une liste des classifieurs
                    faibles formant le classifieur fort
                features : list -> liste des coordonnées des points à soustraire avec les largeurs que l'on associe à un numéro de feature
                conditions : float list -> chaque élément correspond à la condition d'un classifieur fort, et on considère pour chaque 
                    classifieur fort et condition correspondante que si le résultat du classifieur fort est au-dessus 
                    de cette condition, c'est un visage

    Sortie : bool -> True si l'image est considérée comme une image par la fonction de
    détection
    """

    for i,classifieur_fort in enumerate(f) :

        poids = 0

        for numero,polarite,seuil,_,alpha in classifieur_fort :
            numero = int (numero)
            if polarite == 1 :
                if valeur_feature (features[numero],t) < seuil :
                    poids += alpha
            else :
                if valeur_feature (features[numero],t) > seuil :
                    poids += alpha

        if poids < conditions[i] : return False

    return True



def reconnaitre_visages (image,coef_pas,coef_taille,taille_min,pas_min,f,features,f_cascade,condition) :
    """Entree : image : numpy array (2) -> l'image où l'on vérifie s'il y a des visages
                coef_pas : float -> ce à quoi on additionne le pas entre chaque type de carrés
                coef_taille : float -> on balaye des carrés dont la taille augmente de coef_pas à chaque fois
                pas_min : int -> le pas pour les sous-carrés de largeur minimale
                taille_min : int -> taille des premiers carrés à balayer (mutiple de 24)
                f : numpy array (2) -> la fonction de détection (par défaut on suppose que c'est une cascade)
                features : list -> liste des coordonnées des points à soustraire avec les largeurs que l'on associe à un numéro de feature
                f_cascade : bool -> True si la fonction de détection est une cascade, False si elle est monolithique
                condition : float/float list -> si la fonction est monolithique : si le résultat de la fonction 
                    de détection est au-dessus de cette valeur on considère que c'est un visage / si la fonction est une cascade : IDEM
                    mais il s'agit des conditions pour chaque classifieur fort

    Sortie : liste de triplet avec les coordonnées du coin 
             en haut à gauche (x,y) et la longueur des boites (l) devant être tracées sur l'image
    """
    res = []

    hmax,lmax = np.shape(image)

    # à chaque largeur de sous-rectangle correspond un pas
    pas = pas_min

    if f_cascade :
        for largeur in np.arange (taille_min,lmax+1,coef_taille,int) :
            sous_images = subdivision (image,largeur,int(pas))
            for carre,x,y in sous_images :
                h = len (carre)
                carre = np.array(zoom(carre, float(24/h)),dtype=np.uint8)
                if cascade (carre,f,features,condition) :
                    res.append((x,y,largeur))
            pas += coef_pas

    else :
        for largeur in np.arange (taille_min,lmax+1,coef_taille,int) :
            sous_images = subdivision (image,largeur,int(pas))
            for carre,x,y in sous_images :
                h = len (carre)
                carre = np.array(zoom(carre, float(24/h)),dtype=np.uint8)
                if monolithique (carre,f,features,condition) :
                    res.append((x,y,largeur))
            pas += coef_pas

    return res




if __name__ == "__main__" :
    
    taille = 24 # taille minimale des carrés balayés (24 x 24 pour la méthode de Viola-Jones)
    pas = 2 # pas entre chaque sous-rectangles dans les carrés balayés
    increment = 1.2 # ce par quoi on multiplie itérativement la taille des sous-rectangles
    features = retrouver_features(taille,pas,increment) # toutes les features avec ces paramètres
    
    taille_min = 40 # la taille minimale des carrés balayés
    coef_taille = 1.2 # ce par quoi on augmente la taille des carrés balayés itérativement
    pas_min = 2 # le pas minimal lors du balayage des carrés
    coef_pas = 1 # ce par quoi on augmente le pas lorsqu'on augmente la taille des carrés balayés
    
    image = import_image ("chemin") # l'image sur laquelle on veut détecter des visages, chemin à compléter
    image = noir_et_blanc (image)
    
    # Reconnaissance de visage avec une cascade
    fonction_cascade =  np.load("cascade.npy")
    n = len(fonction_cascade)
    f = [fonction_cascade[i][0] for i in range(n)] # la fonction de détection
    conditions = [fonction_cascade[i][1] for i in range(n)] # les conditions pour chaque classifieur fort

    t1 = time()
    visages = reconnaitre_visages (image,coef_pas,coef_taille,taille_min,pas_min,f,features,True,conditions)
    print(time()-t1)

    boites_a_tracer (image,visages)
    
    
    f = np.load("fonction_detection.npy") # la fonction de détection monolithique
    condition = # à compléter
    
    t1 = time()
    visages = reconnaitre_visages (image,coef_pas,coef_taille,taille_min,pas_min,f,features,False,condition)
    print(time()-t1)

    boites_a_tracer (image,visages)
