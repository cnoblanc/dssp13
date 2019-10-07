
# built-in : especes de fleurs d'iris.
iris

library(tidyverse)

# On voit qur : tidyr (pour faire des pivots)
# dplyr : pour faire des opérations sur les jeux de données (filter, join, merge, ...)

# ggplot2 :
ggplot(iris) # spécifier le jeux de données à ggplot
ggplot(iris)  +
    aes(x = Sepal.Length, y=Sepal.Width) +
    geom_point()

# Ajouter l'espèce en couleur et en forme de point
ggplot(iris)  +
    aes(x = Sepal.Length, y=Sepal.Width,color=Species, shape=Species) +
    geom_point()
# Avoir la forme, c'est utile pour les daltoniens et en impression n&B

# On ajoute les labels
ggplot(iris)  +
    aes(x = Sepal.Length, y=Sepal.Width,color=Species, shape=Species) +
    geom_point() +
    labs(x="Longeur des pétales", y="Largeur des pétales"
         , color="Espèces", shape="Espèces",title="Iris de Fisher")

# On ajoute le thème (par défault = grey) et centrage du titre
ggplot(iris)  +
    aes(x = Sepal.Length, y=Sepal.Width,color=Species, shape=Species) +
    geom_point() +
    labs(x="Longeur des pétales", y="Largeur des pétales"
         , color="Espèces", shape="Espèces",title="Iris de Fisher") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5))

# Couleurs de l'echelle + pettre en place un axe a echelle logarithmique
ggplot(iris)  +
    aes(x = Sepal.Length, y=Sepal.Width,color=Species, shape=Species) +
    geom_point() +
    scale_color_manual(values=c("setosa"="red","versicolor"="blue","virginica"="green")) +
    scale_x_log10() +
    labs(x="Longeur des pétales", y="Largeur des pétales"
         , color="Espèces", shape="Espèces",title="Iris de Fisher") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5))

# Facets : regroupements
ggplot(iris)  +
    aes(x = Sepal.Length, y=Sepal.Width,color=Species, shape=Species) +
    geom_point() +
    scale_color_manual(values=c("setosa"="red","versicolor"="blue","virginica"="green")) +
    facet_wrap(vars(Species)) +
    labs(x="Longeur des pétales", y="Largeur des pétales"
         , color="Espèces", shape="Espèces",title="Iris de Fisher") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5))

# Enlever la legende
ggplot(iris)  +
    aes(x = Sepal.Length, y=Sepal.Width,color=Species, shape=Species) +
    geom_point() +
    scale_color_manual(values=c("setosa"="red","versicolor"="blue","virginica"="green")) +
    facet_wrap(vars(Species)) +
    labs(x="Longeur des pétales", y="Largeur des pétales"
         , color="Espèces", shape="Espèces",title="Iris de Fisher") +
    theme_minimal() +
    theme(plot.title = element_text(hjust=0.5),legend.position = "none")




