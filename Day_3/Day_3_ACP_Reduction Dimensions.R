
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

# Normalisation
decathlonR <- decathlon
decathlonR[1:10] <- scale(decathlonR[1:10])
decathlonR2 <- decathlonR %>% rownames_to_column()

#-----------------------
# PCA
# ----------------------
library("FactoMineR")
PCADecathlonR <- PCA(decathlonR[1:10], graph = FALSE)
str(PCADecathlonR)

CDecathlonR <- cov(as.matrix(decathlonR[1:10]))
EDecathlonR <- eigen(CDecathlonR)
UDecathlonR <- EDecathlonR$vectors
LambdaDecathlonR <- EDecathlonR$values

PCADecathlonR2 <- as.matrix(decathlonR[1:10]) %*% UDecathlonR

# Le plan qui maximize le mieux la représentation des sportifs ?
ggplot(data= data.frame(X1 = PCADecathlonR$ind$coord[,1],
                        X2 = PCADecathlonR$ind$coord[,2],
                        Col = decathlon$Points),
       aes(x = X1, y = X2, color = Col)) + geom_point(size = 5) +
    geom_text_repel(label = row.names(decathlon)) +
    scale_x_continuous(expand = c(.15,0)) + scale_y_continuous(expand = c(.1,0)) +
    guides(color = FALSE) +
    coord_fixed()
# Voyons les valeurs propres pour voir de quelle façon les 2 axes choisits repréentent l'ensemble ?
EDecathlonR$values
# ce sont les inerties sur les axes respectifs
# inertie totale est forcement 10 car c'est le nombre de dimension
# Donc, 3.27 + 1.73 = 5 -> 50%
# Si on garde les 1, 2 3 et 4 axes, alors on obtient : 74% des valeurs

# Le premier axe, parle de 3,27 variables.
# Le second axe, parle de 1,73 variable.
# X1 et X2 sont les composantes principales.

# Pour savoir quelle Axe représente quelle variable ?
t(PCADecathlonR$var$coord)

# Pour représenter cette explication du choix de la Dimension1 et 2 par rapport 
# aux variables d'origine
# Axe 1 : je veux le maximal d'information
# Axe 2 : je veux le maximal d'info (moins que Axe 1)

# En axe : les 2 axes de l'ACP

#install.packages("ggforce")
library(ggforce)
ggplot(data.frame(X1 = PCADecathlonR$var$coord[,1], X2 = PCADecathlonR$var$coord[,2]))+ 
    ggforce::geom_circle(aes(x0 = 0, y0 = 0, r = 1)) + 
    geom_segment(aes(xend = X1, yend = X2),
                 x = 0 , y = 0, arrow = grid::arrow()) + 
    geom_text(aes(x = X1, y = X2),
              label = names(decathlonR[1:10]),
              vjust = "outward", hjust = "outward") +
    coord_fixed()

# Interprétation
# Pour gagner le décathlon, il vaut mieux suivre l'axe X1:
# être bon en saut en longueur et 100m, 110mhaie, 
# on ne sait pas dire grand chose sur saut en hauteur (pole.vault) et 1500m (car projection faible)
# 

# on ajoute une notion de performance
# une notion de vitesse
library(dplyr)
#install.packages(c("egg", "factoextra"))
library(egg)
decathlonTime = decathlon[,1:10]

decathlonTime <- mutate(decathlonTime,
                        `100m` = 100/`100m`,
                        `400m` = 400/`400m`,
                        `110m.hurdle` = 110/`110m.hurdle`,
                        `1500m` = 1500/`1500m`)
rownames(decathlonTime)  <- rownames(decathlon)

PCAdecathlontime <- PCA(decathlonTime, graph = FALSE)
egg::ggarrange(factoextra::fviz_pca_ind(PCAdecathlontime, repel = TRUE),
               factoextra::fviz_pca_var(PCAdecathlontime, repel = TRUE),
               ncol = 2)

# ou aussi :
library(ggfortify)
prcomp(decathlon[,1:10],scale.=TRUE) %>% autoplot(loadings=TRUE,loadings.label=TRUE)



# Autre façon de faire :
library("FactoMineR")
acp=PCA(decathlon[,1:10],scale.unit=TRUE,graph=FALSE)
library(factoextra)
fviz_eig(acp) # pour visualiser le pourcentage de l'inertie pour chaque axe

acp$eig # Pour voir les valeurs propres
# Et cela donne aussi le pourcentage cumilé en dernière colonne.

# pour visualiser la "qualité" de représentation de chaque individu (cos(angle))
# Si ce sont des points bien représentés, si ils sont proches, ils doivent être proches dans l'espace l'origine
# Mais si ils sont moins bien représentés, même si ils sont proches, il ne sont pas
# forcement proches dans l'espace de départ.
fviz_pca_ind(acp,col.ind="cos2")

# le cercle des dimensions
fviz_pca_var(acp)

fviz_pca_biplot(acp, axes=c(3,4)) # pour aller  chercher les Dimensions 3 et 4
# donc les 3 sports mal représentés avant sont bien représentés ici.

grid.arrange(
    fviz_pca_biplot(acp, axes=c(1,2), col.ind="cos2"),
    fviz_pca_biplot(acp, axes=c(1,3), col.ind="cos2"),
    nrow=1)

# --------------------
# CLUSTERING
# -------------------
decathlonR
selectDect<-decathlonR[1:10] # pour ne garder que les 10 premirè!res variables numériques
reskmeans=kmeans(selectDect,4) # K-Means
reskmeans$cluster

fviz_pca_ind(acp,col.ind=factor(reskmeans$cluster))

# On calcul une matrice de distance
distd=dist(selectDect, method="euclidean") # on peut choisir une autre distance, si besoin
reshclust=hclust(distd,method="ward.D2") # on lance un H_Clust
plot(reshclust)

plot(hclust(distd,method="complete"),labels=FALSE)

plot(hclust(distd,method="average"),labels=FALSE)









