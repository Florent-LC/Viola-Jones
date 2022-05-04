# Viola-Jones
L'algorithme de Viola et Jones sous Python

La fonction de détection monolithique ("fonction_detection.npy") a été modifiée par rapport à sa 
sortie d'Adaboost. Sur les 200 classifieurs faibles sortis, nous n'en avons conservé que 141, en 
enlevant tous les classifieurs faibles avec des features trop petites.

La fonction de détection en cascade a été déterminée avec les valeurs suivantes (cf 
construction_classifieurs_forts.py) : f = 0.4, d = 0.95, F_target = 0.01
