
#
# 10 epreuves du décathlon.
# 
# Représenter les athlètes sur les 10 épreuves du décatlhon
#
# On peut le faire avec chaque couple d'épreuves
# Mais ça fait beaucoup

#
library("tidyverse")
# install.packages("FactoMineR")
library("FactoMineR")

# Ce jeu de données est dans la library "FactoMineR"
data(decathlon) # data frame
colnames(decathlon)
# coup d'oeil sur les données
glimpse(decathlon) # en indiquant le type des variables
# summary en fonction des variables
summary(decathlon)

 
# install.packages("ggrepel")
library("ggrepel")
# On enchaine 2 opération
# on met le nom des lignes comme colonne individuelle
decathlon2 <- decathlon %>% rownames_to_column()

ggplot(data= decathlon2, aes(x = `100m`,
                             y = Long.jump,
                             color = Points)) + # couleur : performance totale
    geom_smooth()+ # modèle approximation lisse au plus proche du nuage de points
    geom_smooth(method="lm") + # modèle en linéar modèle, avec ecart-type en gris
    geom_text_repel(aes(label = rowname),
                    box.padding = unit(0.75, "lines")) + # "repel" pour écarter les labels des points
    geom_point(size = 5)

# De là, on peut avoir envi de calculer une notion de "performance" :
# car au 100m, bien performer, c'est aller vite (vitesse)
# au saut en longeur, c'est aller plus loin (distance)


# Il y  a til des liens entre 2 variables  ?
# voit-on une corélation ? -> une droite ?

# install.packages("GGally")
# install.packages("ggfortify")
library(GGally)
library(ggfortify)

ggpairs(data =decathlon , columns=1:10, aes(color=Competition))



# Y a t il une corrélation entre les épreuves ?
cor(decathlon[,1:10])
# pour toutes les paires  de variables : calcul d'un coéficient
# entre -1 et 1 : proche de 1 : dépendance (-1 : corrélation négative)
# proche de zero : pas de corrélation

heatmap(abs(cor(decathlon[,1:10])),symm=T)
# on trouve un groupe de 4 épreuves qui sont proches : 400m, 110m haie, 100m et saut en longueur
# Un autre bloc : le disque et lancé de poids (shot.put & Discus)

#install.packages("corrplot")
library(corrplot)
corrplot(cor(decathlon[,1:10]))
corrplot(cor(decathlon[,1:10]),method="color",order="hclust")


# Corrélation : les sportifs qui sont bon en 100m sont aussi bon en saut longueur.
# Il n'y a pas de causalité.

# Normalisation des variables pour éviter les distortions
#install.packages("rgl")
#install.packages("scales")
library(rgl)
library(scales)
clear3d()
plot3d(as.matrix(decathlon[,1:3]),
       type = "s", size = 5, 
       # scale va normaliser les données
       col = cscale(decathlon$Points,
                    seq_gradient_pal("#132B43",
                                     high = "#56B1F7")),
       xlim = c(min(decathlon[,1:3])-0.1,max(decathlon[,1:3])+0.1),
       ylim=c(min(decathlon[,1:3])-0.1,max(decathlon[,1:3])+0.1),
       zlim=c(min(decathlon[,1:3])-0.1,max(decathlon[,1:3])+0.1))

rglwidget(elementId = "plot3d_2")         






