from manipulation_image import *
from creation_features import *


def retrouver_feature1(taille, pas, increment):
    """Entrée : taille : int -> taille des images carrées (par exemple taille = 24 pour des images 24x24)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : int -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois
    
    Sortie : list -> liste des coordonnées des points à soustraire avec les largeurs que l'on associe à un numéro de feature

    La première feature correspond à la différence entre 
    le sous-rectangle du haut et celui du bas :

        +   +   +
        -   -   -
        """

    features = []
    for l in np.arange(1, taille+1, increment,dtype=int): # les largeurs des sous-rectangles
        for h in np.arange(1, taille//2 + 1, increment,dtype=int): # les hauteurs des sous-rectangles
            for x in range(0, 1 + (taille - l), pas): # les abscisses
                for y in range(0, 1 + (taille - 2*h), pas): # les ordonnées
                    if x + l <= taille and y + 2*h <= taille :
                        features.append((1,x,y,l,h))

    return features


def retrouver_feature2(taille, pas, increment):
    """Entrée : taille : int -> taille des images carrées (par exemple taille = 24 pour des images 24x24)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : int -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois
    
    Sortie : list -> liste des coordonnées des points à soustraire avec les largeurs que l'on associe à un numéro de feature

    La deuxième feature correspond à la différence entre 
    le sous-rectangle de gauche et celui de droite :

        +   -
        +   -
        +   -
        """

    features = []
    for l in np.arange(1, taille//2 + 1, increment,dtype=int): # les largeurs des sous-rectangles
        for h in np.arange(1, taille + 1, increment,dtype=int): # les hauteurs des sous-rectangles
            for x in range(0, 1 + (taille - 2*l), pas): # les abscisses
                for y in range(0, 1 + (taille - h), pas): # les ordonnées
                    if x + 2*l <= taille and y + h <= taille :
                        features.append((2,x,y,l,h))

    return features


def retrouver_feature3(taille, pas, increment):
    """Entrée : taille : int -> taille des images carrées (par exemple taille = 24 pour des images 24x24)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : int -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois
    
    Sortie : list -> liste des coordonnées des points à soustraire avec les largeurs que l'on associe à un numéro de feature

    La troisième feature correspond à la différence entre les coins haut-gauche
    et bas-droit, et les coins haut-droit et bas-gauche :

        +   +   -   -
        +   +   -   -
        -   -   +   +
        -   -   +   +

        """

    features = []
    for l in np.arange(1, taille//2 + 1, increment,dtype=int): # les largeurs (et donc hauteurs) des sous-rectangles
        for h in np.arange(1, taille//2 + 1, increment,dtype=int): # les largeurs (et donc hauteurs) des sous-rectangles
            for x in range(0, 1 + (taille - 2*l), pas): # les abscisses
                for y in range(0, 1 + (taille - 2*h), pas): # les ordonnées
                    if x + 2*l <= taille and y + 2*h <= taille :
                        features.append((3,x,y,l,h))

    return features


def retrouver_feature4(taille, pas, increment):
    """Entrée : taille : int -> taille des images carrées (par exemple taille = 24 pour des images 24x24)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : int -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois
    
    Sortie : list -> liste des coordonnées des points avec les largeurs à soustraire que l'on associe à un numéro de feature

    La quatrième feature correspond à la différence entre un carré central
    et deux carrés situés à gauche et à droite :

        +   +   -   -   +   +
        +   +   -   -   +   +

        """

    features = []
    for l in np.arange(1, taille//3 + 1, increment,dtype=int): # les largeurs des sous-rectangles
        for h in np.arange(1, taille + 1, increment,dtype=int): # les hauteurs des sous-rectangles
            for x in range(0, (taille - 3*l) + 1, pas): # les abscisses
                for y in range(0, 1 + (taille - h), pas): # les ordonnées
                    if x + 3*l <= taille and y + h <= taille :
                        features.append((4,x,y,l,h))

    return features


def retrouver_features(taille, pas, increment):
    """Entrée : taille : int -> taille des images carrées (par exemple taille = 24 pour des images 24x24)
                pas : int -> pas entre deux positions de deux rectangles 
                    (en largeur et en hauteur) de même hauteur et de même largeur
                increment : int -> on augmente la largeur et la hauteur (entre deux points) 
                    de l'ensemble des deux sous-rectangles de increment à chaque fois

    Sortie : list -> génère les quatre features décrites ci-dessus,
        en renvoyant pour chacune d'entre elles les valeurs des features
    """

    f1 = retrouver_feature1(taille, pas, increment)

    f2 = retrouver_feature2(taille, pas, increment)

    f3 = retrouver_feature3(taille, pas, increment)

    f4 = retrouver_feature4(taille, pas, increment)

    return f1 + f2 + f3 + f4


if __name__ == "__main__" :
    taille = 5
    pas = 2
    increment = 1.2

    # on peut vérifier à la main que ces valeurs de features sont cohérentes
    print(retrouver_feature1(taille,pas,increment),"\n")
    print(retrouver_feature2(taille,pas,increment),"\n")
    print(retrouver_feature3(taille,pas,increment),"\n")
    print(retrouver_feature4(taille,pas,increment),"\n")

    # nombre de features sélectionnées
    taille = 24
    print(len(retrouver_features(taille,pas,increment)))
