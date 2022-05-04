import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


def import_image(chemin):
    '''Entrée : chemin : str -> nom du chemin du fichier

    Sortie : numpy array (2) -> tableau correspondant à l'image
    '''
    
    return img.imread(chemin)


def noir_et_blanc (image) :
    '''Entrée : image : numpy array (3) -> l'image à transformer

    Sortie : numpy array ->  l'image transformée en noir et blanc (2)

    Les poids correspondant (0.299, 0.587, 0.144) sont des normes à respecter'''
    
    return (np.array(image [:,:,0]*0.299 + image [:,:,1]*0.587 + image [:,:,2]*0.114,dtype = np.int64))


def affiche (image,noir_et_blanc = True) :
    '''Entrée : image : numpy array (2/3) -> l'image à afficher
                noir-et_blanc : bool -> affiche l'image en noir et blanc (True) ou en couleur (False)

    Sortie : None 
   
    Renvoie l'image (qu'elle soit en noir et blanc ou en couleur)'''
    
    if noir_et_blanc :
        plt.imshow(image,cmap=plt.get_cmap("gray"))
        plt.show ()
    else :
        plt.imshow(image)
        plt.show()


def subdivision (image, largeur, pas):
    '''Entrée : image : numpy array (2) -> l'image à subdiviser
                largeur : int -> la largeur de chaque carré
                pas : int -> le pas entre chaque carré

    Sortie : liste -> tous les carrés de dimension
    largeur*largeur espacés entre eux de pas (en hauteur et/ou en largeur) avec la position
    de ces sous-carrés dans l'image (coordonnées du coin en haut à gauche)'''
    
    h,l = np.shape(image)
    res = []
    for y in range(0,h-largeur,pas):
        for x in range(0,l-largeur,pas):
            res.append((image[y:y+largeur, x:x+largeur],x,y))
    return res


def image_integrale (tableau) :
    '''Entrée : tableau : numpy array (2) -> le tableau à transformer (matrice carrée)

    Sortie : numpy array (2) -> tableau après application du principe d'image intégrale'''

    n = len (tableau)
    res = np.array (tableau,dtype=np.int64)
    
    # on applique ce principe sur la première colonne et la première ligne
    for i in range (1,n) :
        res [i,0] += res [i-1,0]
        res [0,i] += res [0,i-1]
    
    # on peut ensuite appliquer ce principe sur les autres coefficients
    # sachant que chaque sommation aura lieu sur des coefficients qui existent
    # (aucun problème de bords) : on somme ligne par ligne
    for i in range (1,n) : # parcours sur les lignes
        for j in range (1,n) : # parcours sur les colonnes
            res [i,j] += res [i,j-1] + res [i-1,j] - res [i-1,j-1]
    return (res)


def aire (tableau,i0,j0,largeur,hauteur) :
    '''Entrée : tableau : numpy array (2) -> on suppose que le principe d'image intégrale a été appliqué
                i0 : int -> abscisse de la ligne du coin en haut à gauche du sous-rectangle considéré
                j0 : int -> ordonnée de la colonne du coin en haut à gauche du sous-rectangle considéré (graphe inversé)
                largeur : int -> largeur du sous-rectangle considéré (nombre de points sur la largeur)
                hauteur : int -> hauteur du sous-rectangle considéré (nombre de points sur la hauteur)

    Sortie : int -> l'aire du sous-rectangle à l'intérieur du tableau
             bool -> True si l'aire est définie (pas de problèmes de bords)'''

    #   A B
    #   C D

    # pour calculer l'aire d'une image intégrale, on effectue : D + A - B - C
    # si A, C, et B appartiennent à l'image : en effet, A,B et C sont les extrémités
    # non incluses du rectangle, et peuvent donc être à l'extérieur de l'image

    lx = len(tableau)
    ly = len (tableau[:,0])

    if i0 + largeur > ly or j0 + hauteur > lx :
        return (0,False)

    if i0 == 0 and j0 == 0 : # A,B et C sont tous trois à l'extérieur
        return (tableau [j0+hauteur-1,i0+largeur-1],True) # D
    elif i0 == 0 :
        return (tableau [j0+hauteur-1,i0+largeur-1] - tableau [j0-1,i0+largeur-1],True) # D-B
    elif j0 == 0 :
        return (tableau [j0+hauteur-1,i0+largeur-1] - tableau [j0+hauteur-1,i0-1],True) # D-C
    else : 
        return (tableau [j0+hauteur-1,i0+largeur-1] + tableau[j0-1,i0-1] - \
               tableau [j0-1,i0+largeur-1] - tableau [j0+hauteur-1,i0-1],True) # D+A-B-C


def boites_a_tracer (image,liste_boites) :
    '''Entrée : image : numpy array (2) -> l'image sur laquelle on va dessiner les carrés
                liste_boites : tuple list -> liste de triplet avec les coordonnées du coin 
                en haut à gauche (x,y) et la longueur des boites (l) devant être tracées sur l'image
        
    Sortie : None
    
    Affiche l'image modifiée, avec ces boîtes tracées'''

    res = np.copy(image)
    
    for (x,y,l) in liste_boites :
        # on trace les boites avec des lignes rouges
        plt.plot ([x,x],[y,y+l],'r')
        plt.plot ([x+l,x+l],[y,y+l],'r')
        plt.plot ([x,x+l],[y,y],'r')
        plt.plot ([x,x+l],[y+l,y+l],'r')

    plt.imshow(res,cmap = plt.get_cmap("gray"))
    plt.show()


if __name__ == "__main__" :
    image = import_image("chemin") # chemin à compléter
    affiche (image,False)
    image_nb = noir_et_blanc(image)
    affiche(image_nb)

    carres = subdivision (image_nb,_,_) # à compléter
    for carre,_,_ in carres[:5] : affiche(carre)

    liste_boites = [] # à compléter
    boites_a_tracer (image_nb,liste_boites)

    t = np.arange(1,577).reshape((24,24))
    t_int = image_integrale(t)
    print (aire(t_int,0,3,4,2))
    print(t[3:5,0:4])
    print (73+74+75+76+97+98+99+100 == aire(t_int,0,3,4,2)[0])
