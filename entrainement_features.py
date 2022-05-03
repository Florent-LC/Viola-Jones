import numpy as np

from manipulation_image import *
from creation_features import *

from time import time


def dichotomie (t,e) :
    """Entrée : t : liste ou numpy array (1) -> une liste / un tableau trié
                e : int -> l'élément à comparer

        Sortie : int -> l'indice du plus grand élément du tableau qui soit strictement inférieur à e
        Par convention, si tous les éléments du tableau sont plus grands que e, la fonction retourne 0"""
    
    n = len (t)
    if len(t) == 0 : return 0
    if t[-1] < e : return n-1
    if t[n//2] < e : return n//2 + dichotomie (t[n//2:],e)
    else : return dichotomie (t[:n//2],e)


def preliminaire_entrainement_classifieurs_faibles (bdd_visage,bdd_non_visage,features_positifs,features_negatifs,linspace,K=500) :
    """Entrée : bdd_visage : numpy array (3) -> la base de données d'images de visages : une dimension 
                    pour le nombre d'images et les deux autres pour les dimensions de chacune des images
                bdd_non_visage : numpy array (3) -> la base de données d'images de non-visages : une dimension 
                    pour le nombre d'images et les deux autres pour les dimensions de chacune des images
                features_positifs : numpy array (2) -> l'évaluation de toutes les features sur la base de donnée de visages
                features_negatifs : numpy array (2) -> l'évaluation de toutes les features sur la base de donnée de non-visages
                linspace : bool -> si True : on découpe les valeurs de seuils possibles avec la fonction linspace du module numpy
                                   si False : on s'intéresse à l'ensemble des valeurs de features pour la base de données pour
                                   déterminer le meilleur seuil (empiriquement, le nombre de valeur distinstes oscille autour de 600 environ)
                K : int -> caractérise la précision de la recherche des seuils si linspace = True (précision proportionnelle à K)
                

    Sortie : tuple (7) ->   indice_tri_positifs : numpy array (2) -> pour chaque feature (première dimension), indice permettant de ranger
                                les valeurs de cette feature sur la base de données de visages par ordre croissant (deuxième dimension)
                            indice_tri_negatifs : numpy array (2) -> pour chaque feature (première dimension), indice permettant de ranger
                                les valeurs de cette feature sur la base de données de non-visages par ordre croissant (deuxième dimension)
                            valeurs_feature_positifs : numpy array (2) -> pour chaque feature (première dimension), la valeur de cette
                                feature sur la base de données de visage (deuxième dimension), mais trié par valeur croissante des features
                            valeurs_feature_negatifs : numpy array (2) -> pour chaque feature (première dimension), la valeur de cette
                                feature sur la base de données de non-visages (deuxième dimension), mais trié par valeur croissante des features
                            intervalle : array list -> pour chaque feature (première dimension), l'intervalle discrétisant les valeurs de
                                seuil possible
                            i_plus : array list-> pour chaque feature (première dimension), et pour chaque seuil de intervalle (deuxième dimension)
                                l'indice de la plus grande valeur de feature de la base de données de visages qui soit strictement inférieur au seuil
                            i_moins : array list -> pour chaque feature (première dimension), et pour chaque seuil de intervalle (deuxième dimension)
                                l'indice de la plus grande valeur de feature de la base de données de non-visages qui soit strictement inférieur au seuil
    
    preliminaire_entrainement_classifieurs_faibles permet de faire toutes les opérations que devraient faire entrainement_classifieurs_faibles,
    mais qui ne dépendent pas des poids (et donc du nombre d'itérations T dans la fonction adaboost) : permet donc de gagner un temps considérable,
    toutes ces opérations étant réalisées une seule fois, au lieu de T fois"""
    
    features = np.concatenate ((features_positifs,features_negatifs),1) 

    nombre_features = len (features)

    indice_tri_positifs = np.zeros ((nombre_features,len(bdd_visage)),dtype=np.int32)
    indice_tri_negatifs = np.zeros ((nombre_features,len(bdd_non_visage)),dtype=np.int32)

    valeurs_feature_positifs = np.zeros ((nombre_features,len(bdd_visage)))
    valeurs_feature_negatifs = np.zeros ((nombre_features,len(bdd_non_visage)))

    i_plus = nombre_features*[0]
    i_moins = nombre_features*[0]
 
    intervalle = nombre_features*[0]

    for i,feature in enumerate(features):

        # la valeur de la ième feature pour les visages et les non-visages
        positifs = features_positifs[i]
        negatifs = features_negatifs[i]

        # les indices de tableau qui donne des features rangés par ordre croissant
        indice_positifs = np.argsort(positifs)
        indice_negatifs = np.argsort(negatifs)

        indice_tri_positifs[i] = indice_positifs
        indice_tri_negatifs[i] = indice_negatifs

        # on trie les features par ordre croissant des valeurs des features
        valeurs_feature_positifs[i] = positifs[indice_positifs]
        valeurs_feature_negatifs[i] = negatifs[indice_negatifs]

        if linspace :
            t_min = min ((valeurs_feature_positifs[i,0],valeurs_feature_negatifs[i,0]))
            t_max = max ((valeurs_feature_positifs[i,-1],valeurs_feature_negatifs[i,-1]))
            intervalle[i] = np.linspace(t_min,t_max,K)
        else :
            intervalle[i] = np.unique(np.concatenate((valeurs_feature_positifs[i],valeurs_feature_negatifs[i])))
            # le nombre de valeurs distinctes de features pour les deux bases de données

        i_plus_feature = np.zeros(len(intervalle[i]),dtype=np.int32)
        i_moins_feature = np.zeros(len(intervalle[i]),dtype=np.int32)

        valeurs_positifs = valeurs_feature_positifs[i]
        valeurs_negatifs = valeurs_feature_negatifs[i]

        intervalle_feature = intervalle[i]

        for j,seuil in enumerate(intervalle_feature[1:]) :

            i_plus_feature[j+1] = dichotomie (valeurs_positifs,seuil)
            i_moins_feature[j+1] = dichotomie (valeurs_negatifs,seuil)

        i_plus[i] = i_plus_feature
        i_moins[i] = i_moins_feature

    return (indice_tri_positifs,indice_tri_negatifs,valeurs_feature_positifs,valeurs_feature_negatifs,intervalle,i_plus,i_moins)


def entrainement_classifieurs_faibles (bdd_visage,bdd_non_visage,poids_visages,poids_non_visages,features_positifs,features_negatifs, \
                                       indice_tri_positifs,indice_tri_negatifs,valeurs_feature_positifs,valeurs_feature_negatifs,intervalle,i_plus,i_moins,linspace=True,K=500):
    """Entrée : bdd_visage : numpy array (3) -> la base de données d'images de visages : une dimension 
                    pour le nombre d'images et les deux autres pour les dimensions de chacune des images
                bdd_non_visage : numpy array (3) -> la base de données d'images de non-visages : une dimension 
                    pour le nombre d'images et les deux autres pour les dimensions de chacune des images
                poids_visages : numpy array (1) -> le tableau des poids des visages
                poids_non_visages : numpy array (1) -> le tableau des poids des non-visages
                features_positifs : numpy array (2) -> l'évaluation de toutes les features sur la base de donnée de visages
                features_negatifs : numpy array (2) -> l'évaluation de toutes les features sur la base de donnée de non-visages
                indice_tri_positifs : numpy array (2) -> ordre croissant des valeurs des features (première dimension)
                    sur la base de données de visage (deuxième dimension)
                indice_tri_positifs : numpy array (2) -> ordre croissant des valeurs des features (première dimension)
                    sur la base de données de non-visages (deuxième dimension)
                valeurs_feature_positifs : numpy array (2) -> pour chaque feature (première dimension), la valeur de cette
                    feature sur la base de données de visage (deuxième dimension), mais trié par valeur croissante des features
                valeurs_feature_negatifs : numpy array (2) -> pour chaque feature (première dimension), la valeur de cette
                    feature sur la base de données de non-visages (deuxième dimension), mais trié par valeur croissante des features
                intervalle : array list -> pour chaque feature (première dimension), l'intervalle discrétisant les valeurs de
                    seuil possible
                i_plus : array list -> pour chaque feature (première dimension), et pour chaque seuil de intervalle (deuxième dimension)
                    l'indice de la plus grande valeur de feature de la base de données de visages qui soit strictement inférieur au seuil
                i_moins : array list -> pour chaque feature (première dimension), et pour chaque seuil de intervalle (deuxième dimension)
                    l'indice de la plus grande valeur de feature de la base de données de non-visages qui soit strictement inférieur au seuil
                linspace : bool -> si True : on découpe les valeurs de seuils possibles avec la fonction linspace du module numpy
                                   si False : on s'intéresse à l'ensemble des valeurs de features pour la base de données pour
                                   déterminer le meilleur seuil (empiriquement, le nombre de valeur distinstes oscille autour de 600 environ)
                K : int -> caractérise la précision de la recherche des seuils si linspace = True (précision proportionnelle à K)
                

    Sortie : tuple list -> pour chaque élément de la liste (feature, polarite, seuil, erreur) :
        feature : numpy array (3) -> valeur de la feature pour toute la base de donnée
        polarité : int -> +1 si on considère une image comme un visage si la valeur de la
            feature est inférieure au seuil et -1 dans le cas contraire
        seuil : float -> cf ci-dessus
        erreur : float -> l'erreur du classifieur

    entrainement_classifieurs_faibles permet d'entraîner les polarité et seuil de chaque classifieur faible"""
    
    features = np.concatenate ((features_positifs,features_negatifs),1) 

    t_plus = np.sum(poids_visages)
    t_moins = np.sum(poids_non_visages)

    nombre_features = len (features)

    classifieurs = nombre_features*[(np.zeros(len(bdd_visage)+len(bdd_non_visage)),0,0,0)]

    for i,feature in enumerate(features):

        # on trie les tableaux des poids par valeurs croissantes des features
        indice_positifs = indice_tri_positifs[i]
        indice_negatifs = indice_tri_negatifs[i]
        poids_positifs = poids_visages[indice_positifs]
        poids_negatifs = poids_non_visages[indice_negatifs]       

        t_min = min ((valeurs_feature_positifs[i,0],valeurs_feature_negatifs[i,0]))

        intervalle_feature = intervalle[i]

        i_plus_feature = i_plus[i]
        i_moins_feature = i_moins[i]

        # dans la suite du programme (jusqu'à "meilleur_seuil = seuil"), on va déterminer les valeurs initiales
        # de seuil, erreur et polarité et considérer, jusqu'à preuve du contraire, que ce sont les meilleurs valeurs

        # si polarité = -1: visage détecté si feature > seuil
        #    erreur = s_plus                 -> positifs, feature < seuil
        #             + (t_moins - s_moins)  -> négatifs, feature > seuil
        # si polarité = +1: visage détecté si feature < seuil
        #    erreur = s_moins                -> négatifs, feature < seuil
        #             + (t_plus - s_plus)    -> positifs, feature > seuil


        # la valeur initiale du seuil impose par construction que s_plus = s_moins = 0
        seuil = t_min

        erreur_min = min (t_moins, t_plus)

        if erreur_min == t_moins : polarite_min = -1
        else : polarite_min = 1

        meilleur_seuil = seuil

        # maintenant, on va parcourir l'intervalle et chercher les valeurs de seuil et polarité
        # qui minimise l'erreur
        
        for j,seuil in enumerate(intervalle_feature[1:]) :

            s_plus = np.sum(poids_positifs[:i_plus_feature[j+1]+1])
            s_moins = np.sum(poids_negatifs[:i_moins_feature[j+1]+1])

            erreur = min (s_plus + t_moins - s_moins, s_moins + t_plus - s_plus)

            if erreur < erreur_min :
                erreur_min = erreur
                meilleur_seuil = seuil
                if erreur_min == s_plus + t_moins - s_moins : polarite_min = -1
                else : polarite_min = +1

        classifieurs[i] = (feature, int(polarite_min), int(meilleur_seuil), erreur_min)

    return classifieurs


def visage_classifieur (classifieur,image) :
    '''Entrée : classifieur : tuple list -> la valeur de la ième feature pour la base de données totale,
                    sa polarité, son seuil et son erreur
                image : int -> l'indice de l'image considérée dans la base de données concaténée 
                    (qui peut donc être autant un visage qu'un non-visage)

    Sortie : bool -> True si le classifieur considère que l'image est un visage et False sinon
    '''

    feature, polarite, seuil,_ = classifieur

    if polarite == 1 :
        if feature[image] < seuil : return True
        else : return False
    else :
        if feature[image] > seuil : return True
        else : return False


def adaboost (bdd_visage,bdd_non_visage,T,linspace,K = 500,verbeux = True,chargement = True) :
    '''Entrée : bdd_visage : numpy array (3) -> la base de données d'images de visages : une dimension 
                    pour le nombre d'images et les deux autres pour les dimensions de chacune des images
                bdd_non_visage : numpy array (3) -> la base de données d'images de non-visages : une dimension 
                    pour le nombre d'images et les deux autres pour les dimensions de chacune des images
                T : int -> nombre d'itérations (soit le nombre de classifieurs faibles dans la fonction de détection monolithique)
                linspace : bool -> si True : on découpe les valeurs de seuils possibles avec la fonction linspace du module numpy
                                   si False : on s'intéresse à l'ensemble des valeurs de features pour la base de données pour
                                   déterminer le meilleur seuil (empiriquement, le nombre de valeur distinstes oscille autour de 600 environ)
                K : int -> caractérise la précision de la recherche des seuils si linspace = True (précision proportionnelle à K)
                verbeux : bool -> True : permet de suivre la construction de la fonction de détection
                chargement : bool -> True : permet de diminuer le temps d'exécution en chargeant l'évaluation
                des features pour la base de donnée, le nom des fichiers étant "eval_feature_visage", "eval_feature_non_visage"
                    

    Cette fonction adaboost renvoie un unique classifieur fort monolithique avec T classifieurs faibles (ce n'est donc pas une cascade)

    Sortie : tuple list -> la fonction de détection finale (feature, polarite, seuil, erreur, alpha) :
        feature : int -> le numéro de feature
        polarité : int -> -1 si on considère une image comme un visage si la valeur de la
            feature est supérieure au seuil et +1 dans le cas contraire
        seuil : float -> cf ci-dessus
        erreur : float -> l'erreur du classifieur
        alpha : float -> le poids du classifieur
        '''

    nombre_images_visages = len (bdd_visage)
    nombre_images_non_visages = len (bdd_non_visage)

    # pour l'heure, aucune image n'a de raison d'avoir plus de poids que les autres
    poids_visages = (1/(nombre_images_visages+nombre_images_non_visages)) * np.ones (nombre_images_visages)
    poids_non_visages = (1/(nombre_images_visages+nombre_images_non_visages)) * np.ones (nombre_images_non_visages)

    # la fonction de détection finale
    f = T*[[0,0,0,0.,0.]]

    t1 = time()

    if chargement :
        features_positifs = np.loadtxt("eval_feature_visage")
        features_negatifs = np.loadtxt("eval_feature_non_visage")
    else :
        features_positifs = eval_feature(bdd_visage)
        features_negatifs = eval_feature(bdd_non_visage)

    features = np.concatenate ((features_positifs,features_negatifs),1)

    if verbeux : print ("Temps de chargement de la base de données : ",int((time()-t1)*100)/100,"secondes\n\n\n")

    t1 = time()

    indice_tri_positifs,indice_tri_negatifs,valeurs_feature_positifs,valeurs_feature_negatifs,intervalle,i_plus,i_moins = \
        preliminaire_entrainement_classifieurs_faibles (bdd_visage,bdd_non_visage,features_positifs,features_negatifs,linspace,K)

    if verbeux : print ("Temps nécessaire à l'évaluation des préliminaires à l'entraînement des classifieurs faibles : ",int((time()-t1)*100)/100,"secondes\n\n\n")

    for t in range (T) :

        t1 = time()

        somme = np.sum(poids_visages)+np.sum(poids_non_visages)
        poids_visages /= somme
        poids_non_visages /= somme

        # on entraîne itérativement les classifieurs faibles avec les poids des images qui évoluent
        classifieurs = entrainement_classifieurs_faibles (bdd_visage,bdd_non_visage,poids_visages,poids_non_visages,features_positifs,features_negatifs, \
                                       indice_tri_positifs,indice_tri_negatifs,valeurs_feature_positifs,valeurs_feature_negatifs,intervalle,i_plus,i_moins,linspace,K)

        # le tableau des erreurs de chaque classifieur
        erreurs = np.array([e[3] for e in classifieurs])

        indice_erreur_min = int(np.argmin(erreurs))
        
        # on rajoute 10^-300 à l'erreur minimale car une erreur nulle pose des problèmes de définition de fonctions
        erreur_min = erreurs[indice_erreur_min] + 1E-300
        meilleur_classifieur = classifieurs[indice_erreur_min]

        beta = erreur_min / (1. - erreur_min)
        alpha = np.log(1. / beta)

        # on parcourt les images pour mettre à jour leur poids
        # si la classification est correcte, le poids est multiplié par beta < 1, sinon il est inchangé
        # on rappelle que meilleur_classifieur contient comme premier élément la valeur de la ième feature
        # pour toute la base de données de visage (indice 0 à nombre_images_visages - 1) et de la base de
        # données de non-visages (des indices nombre_images_visages à nombre_images_visages + nombre_images_non_visages)
        
        for i in range(nombre_images_visages) :
            if visage_classifieur(meilleur_classifieur,i) :  # classification correcte
                poids_visages[i] *= beta

        for i in range(nombre_images_non_visages) :
            if not visage_classifieur(meilleur_classifieur,i+nombre_images_visages) :  # classification correcte
                poids_non_visages[i] *= beta

        # on n'oublie pas de considérer l'erreur à laquelle on a rajouté 10^-300
        feature, polarite, seuil,_ = meilleur_classifieur
        f[t] = [indice_erreur_min, polarite, seuil, erreur_min, alpha]

        if verbeux : print ("Le classifieur faible numéro ",t," défini par :\n      numéro de feature : ",indice_erreur_min,"\n", \
            "     polarité : ",polarite,"\n","     seuil : ",seuil,"\n","     erreur : ",erreur_min,"\n", "     alpha : ",alpha,"\n", \
           "a été déterminé en ",int((time()-t1)*100)/100," secondes","\n\n")

    return f



if __name__ == "__main__" :
    #Tests unitaires

    bdd_visage = np.load("entrainement-visage.npy") # la base de données de visage pour l'apprentissage
    bdd_non_visage = np.load("entrainement-non-visage.npy") # la base de données de non-visages pour l'apprentissage

    T = 200 # nombre de classifieurs faibles dans la fonction de détection
    linspace = False # la méthode de recherche du seuil

    f = adaboost (bdd_visage,bdd_non_visage,T,linspace,chargement=False)
    print(f)

    t = np.arange(0,100)
    print(dichotomie(t,54.3),t[dichotomie(t,54.3)])
