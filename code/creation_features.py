import numpy as np

from manipulation_image import *



def feature1(t, pas, increment):
    """Entrée : t : numpy array (2) -> tableau à partir duquel on génère les features
                (on suppose lui avoir appliqué le principe d'image intégrale)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : float -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois (les largeurs et hauteurs
                    restant entières)
    
    Sortie : list -> liste des valeurs des features de type 1

    La première feature correspond à la différence entre 
    le sous-rectangle du haut et celui du bas :

        +   +   +
        -   -   -
        """

    hauteur = len (t)
    largeur = len (t[0,:])

    features = []
    for l in np.arange(1, largeur+1, increment,dtype=int): # les largeurs des sous-rectangles
        for h in np.arange(1, hauteur//2 + 1, increment,dtype=int): # les hauteurs des sous-rectangles
            for x in range(0, 1 + (largeur - l), pas): # les abscisses
                for y in range(0, 1 + (hauteur - 2*h), pas): # les ordonnées
                    a1,b1 = aire(t,x,y,l,h)
                    a2,b2 = aire(t,x,y+h,l,h)
                    if b1 and b2 :
                        features.append(a1 - a2)

    return features



def feature2(t, pas, increment):
    """Entrée : t : numpy array (2) -> tableau à partir duquel on génère les features
                (on suppose lui avoir appliqué le principe d'image intégrale)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : float -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois (les largeurs et hauteurs
                    restant entières)
    
    Sortie : list -> liste des valeurs des features de type 2

    La deuxième feature correspond à la différence entre 
    le sous-rectangle de gauche et celui de droite :

        +   -
        +   -
        +   -
        """

    hauteur = len (t)
    largeur = len (t[0,:])

    features = []
    for l in np.arange(1, largeur//2 + 1, increment,dtype=int): # les largeurs des sous-rectangles
        for h in np.arange(1, hauteur + 1, increment,dtype=int): # les hauteurs des sous-rectangles
            for x in range(0, 1 + (largeur - 2*l), pas): # les abscisses
                for y in range(0, 1 + (hauteur - h), pas): # les ordonnées
                    a1,b1 = aire(t,x,y,l,h)
                    a2,b2 = aire(t,x+l,y,l,h)
                    if b1 and b2 :
                        features.append(a1 - a2)

    return features



def feature3(t, pas, increment):
    """Entrée : t : numpy array (2) -> tableau à partir duquel on génère les features
                (on suppose lui avoir appliqué le principe d'image intégrale)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : float -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois (les largeurs et hauteurs
                    restant entières)
    
    Sortie : list -> liste des valeurs des features de type 3

    La troisième feature correspond à la différence entre les coins haut-gauche
    et bas-droit, et les coins haut-droit et bas-gauche :

        +   +   -   -
        +   +   -   -
        -   -   +   +
        -   -   +   +

        """

    hauteur = len (t)
    largeur = len (t[0,:])

    features = []
    for l in np.arange(1, largeur//2 + 1, increment,dtype=int): # les largeurs (et donc hauteurs) des sous-rectangles
        for h in np.arange(1, hauteur//2 + 1, increment,dtype=int): # les largeurs (et donc hauteurs) des sous-rectangles
            for x in range(0, 1 + (largeur - 2*l), pas): # les abscisses
                for y in range(0, 1 + (hauteur - 2*h), pas): # les ordonnées
                    a1,b1 = aire(t,x,y,l,h)
                    a2,b2 = aire(t,x+l,y+h,l,h)
                    a3,b3 = aire(t,x+l,y,l,h)
                    a4,b4 = aire(t,x,y+h,l,h)
                    if b1 and b2 and b3 and b4 :
                        features.append(a1 + a2 - a3 - a4)

    return features



def feature4(t, pas, increment):
    """Entrée : t : numpy array (2) -> tableau à partir duquel on génère les features
                (on suppose lui avoir appliqué le principe d'image intégrale)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : float -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois (les largeurs et hauteurs
                    restant entières)
    
    Sortie : list -> liste des valeurs des features de type 4

    La quatrième feature correspond à la différence entre un carré central
    et deux carrés situés à gauche et à droite :

        +   +   -   -   +   +
        +   +   -   -   +   +

        """

    hauteur = len (t)
    largeur = len (t[0,:])

    features = []
    for l in np.arange(1, largeur//3 + 1, increment,dtype=int): # les largeurs des sous-rectangles
        for h in np.arange(1, hauteur + 1, increment,dtype=int): # les hauteurs des sous-rectangles
            for x in range(0, (largeur - 3*l) + 1, pas): # les abscisses
                for y in range(0, 1 + (hauteur - h), pas): # les ordonnées
                    a1,b1 = aire(t,x,y,l,h)
                    a2,b2 = aire(t,x+l,y,l,h)
                    a3,b3 = aire(t,x+2*l,y,l,h)
                    if b1 and b2 and b3 :
                        features.append(a1 - a2 + a3)

    return features


def generation_features(tableau, pas, increment, verbeux):
    """Entrée : tableau : numpy array (2) -> tableau à partir duquel on génère les features
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : float -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois (les largeurs et hauteurs
                    restant entières)
                verbeux : bool -> True : permet de suivre la génération des features

    Sortie : list -> génère les quatre features décrites ci-dessus,
        en renvoyant pour chacune d'entre elles les valeurs des features
    """

    t = tableau.cumsum(axis=0).cumsum(axis=1,dtype=np.float32) # on priviligiera ici la fonction numpy cumsum bien plus rapide 
    #t = image_integrale(tableau)

    if verbeux:
        print("Génération des features 1 ...")
    f1 = feature1(t, pas, increment)

    if verbeux:
        print("Génération des features 2 ...")
    f2 = feature2(t, pas, increment)

    if verbeux:
        print("Génération des features 3 ...")
    f3 = feature3(t, pas, increment)

    if verbeux:
        print("Génération des features 4 ...\n")
    f4 = feature4(t, pas, increment)

    return f1 + f2 + f3 + f4


def eval_feature(bdd, pas=2, increment=1.2, verbeux=False):
    """Entrée : bdd : numpy array (3) -> la base de données d'images : une dimension 
                    pour le nombre d'images et les deux autres pour les dimensions de chacune des images
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : float -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois (les largeurs et hauteurs
                    restant entières)
                verbeux : bool -> True : permet de suivre la génération des features

    Sortie : numpy array (2) -> pour chaque feature (première dimension), on a la valeur de 
        cette feature pour toutes les images de la base de donnée (dimension 2)
    """

    res = [generation_features (image,pas,increment,verbeux) for image in bdd]
    res = np.array(res)
    # tableau où chaque ligne i correspond à la valeur de la ième feature pour toute la base de donnée :
    trans = np.transpose(res)

    return (trans)


if __name__ == "__main__" :

    # On peut vérifier à la main (c'est long) que les valeurs retournées sont correctes
    t = np.arange(1,26).reshape((5,5))
    print (feature1(t,3,1.5),"\n")
    print (feature2(t,3,1.5),"\n")
    print (feature3(t,3,1.5),"\n")
    print (feature4(t,3,1.5),"\n")
    
    
    # Évaluation des bases de données
    # Si les bases de données sont trop grandes, il est déconseillé de les enregistrer dans des fichiers numpy
    bdd_visage = np.load("chemin") # la base de données d'entraînement de non-visages, chemin à compléter
    bdd_non_visage = np.load("chemin") # la base de données d'entraînement de non-visages, chemin à compléter

    features_positifs = eval_feature (bdd_visage)
    features_negatifs = eval_feature (bdd_non_visage)
    
    np.savetxt("eval_feature_visage",features_positifs)
    np.savetxt("eval_feature_non_visage",features_negatifs)
