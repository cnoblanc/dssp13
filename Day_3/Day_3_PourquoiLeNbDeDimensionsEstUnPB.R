
# Cf théorème : page 19 sur 102

# Avec 2 coordonnées
set.seed(42)
X <- array(runif(1000*1000),dim = c(1000, 1000))
dim(X)
class(X)
distorigin <- apply(X[, 1, drop = FALSE],1 , norm, type = "2")
length(distorigin)
head(distorigin)
class(X[, 1, drop = FALSE])
class(distorigin) # vecteur de numeric




# générer 1000 points aléatoirement dans un espace de 1000 colonnes
set.seed(42)
X <- array(runif(1000*1000),dim = c(1000, 1000))

# Regardons les distances sur la premières coordonnées

# X[,1] : selectinner toutes les lignes et toutes les colonnes. 
# Cela donnerais un vecteur
# X[,1, drop=FALSE] : Drop = FALSE : garde la forme matrice, mais avec qu'une seule colonne
#
# On va appliquer une fonction à un objet : apply (array)
# appliquer une fonction sur la première dimension (par ligne) : apply (array,1)
# je calcul la norme de mon scalaire, type =2 pour avoir la norme euclidienne : norm, type = "2"
# norm 2 = norme euclidienne = racine(somme(carré))
# norm 1 = somme(valeur absolue)
# En R, les indices commencent à 1.

distorigin <- apply(X[, 1, drop = FALSE],1 , norm, type = "2")

# Apply (X, dim, FUN)
# X = matrice n,p
# extraire la dimension spécifiée
# appliquer la fonction sur la dimension 1 (les lignes)
# J'obtient le résultat en sortie.

min(distorigin) # Calcul du minimum
max(distorigin) # Calcul du max

# Voyons maintenant les 2 premières colonnes
# Norm : norm d'un vecteiur de taille 2
distorigin <- apply(X[, 1:2, drop = FALSE],1 , norm, type = "2")
min(distorigin) # Calcul du minimum
max(distorigin) # Calcul du max

distorigin <- apply(X[, 1:1000, drop = FALSE],1 , norm, type = "2") /sqrt(1000)
min(distorigin) # Calcul du minimum
max(distorigin) # Calcul du max

# Donc, plus on augmente le nombre de dimensions, moins les distances sont visibles 
# c.a.d. la différence entre min et max est plus petite.


library(tidyverse)
# Je crée une liste avec des nombres equi-répartis qui va me donner une liste 
# de nombre de dimensions.
# Je prends d'abord les 10 premiers entiers, puis une liste
# equirépartis d'éléments de 11 à 1.000
d_list <- c(1:10, round(exp(seq(log(11), log(1000), length.out = 40))))
d_list <- c(1:10, round(10^(seq(log10(11), log10(1000), length.out = 40)))) # idem précédent
d_list

compute_dist_origin <- function(d, mat) {
    apply(mat[, 1:d, drop = FALSE],1 ,norm, type = "2")
} 

suppressPackageStartupMessages(library(tidyverse))
# je prends d_list et je lui applique compute_dist_origin auquel je lui passe la matrice X
dist_origin <- map(d_list, compute_dist_origin, mat = X) # aquivalent du map/reduce
min_dist_origin <- map_dbl(dist_origin, min)
min_dist_origin
max_dist_origin <- map_dbl(dist_origin, max)
max_dist_origin


# R est fait pour définir des vecteurs ou matrices
# et lui appliquer des trucs, des opérateurs
# R est fait pour ça : appliquer des opérations à des matrices ou des vecteurs.


# On représente les données pour visualiser

# d_list : nombre de dimensions
# Min & max
dists <- tibble(d = d_list, min_dist_origin,max_dist_origin)

ggplot(data = dists, aes(x = d)) +
    geom_line(aes(y = max_dist_origin/sqrt(d), # je normalise par racine (d)
                  color = "max")) +
    geom_line(aes(y = min_dist_origin/sqrt(d), # je normalise par racine (d)
                  color = "min")) +
    geom_hline(yintercept = sqrt(1/3), linetype = "dashed") + # 1/3 est l'espérance obtenue (valeur théorique)
    xlab("dimension") + ylab("distance")

# Donc plus on a de dimensions, plus j'aurais du mal à différencier les observations

# Voyons le ratio entre min et max vs le nbr de dimensions.
ggplot(data = dists,
       aes(x = d, y = max_dist_origin/min_dist_origin)) + geom_line() +
    geom_hline(yintercept = 1, linetype = "dashed") +
    xlab("dimension") + ylab("distance ratio") +
    ylim(0,10)

# Du coup, on voit que + on a de dimensions, 
# plus il est difficile de trouver des éléments différentiateurs (des groupes)
# les distances entre éléments se retrouvent êtres similaires de plus en plus
# partout.
# avec peu de dimensions, on trouve des éléments différentiateurs.






