# ------------
# Accidents
# -----------

# Dimension reduction
library("FactoMineR")
library(ggplot2)

caracteristiques_quali<-caracteristiques_Dummy
summary(caracteristiques_quali)

PCA_Carac <- PCA(caracteristiques_quali[,7:11], graph = FALSE)
str(PCA_Carac)

C_Carac <- cov(as.matrix(caracteristiques_quali[,7:11]))
E_Carac <- eigen(C_Carac)
U_Carac <- E_Carac$vectors
Lambda_Carac <- E_Carac$values

PCA_Carac_2 <- as.matrix(caracteristiques_quali[,7:11]) %*% U_Carac

# Le plan qui maximize le mieux la représentation des Caracteristique des accidents
ggplot(data= data.frame(X1 = PCA_Carac$ind$coord[,1],
                        X2 = PCA_Carac$ind$coord[,2]
                        #,Col = decathlon$Points
                        ),
       aes(x = X1, y = X2)) + geom_point(size = 1) +
    #geom_text_repel(label = row.names(decathlon)) +
    scale_x_continuous(expand = c(.15,0)) + scale_y_continuous(expand = c(.1,0)) +
    guides(color = FALSE) +
    coord_fixed()
# Voyons les valeurs propres pour voir de quelle façon les 2 axes choisits repréentent l'ensemble ?
E_Carac$values

# Pourcentage cumulé des valeurs propres 
t(PCA_Carac$eig)
# Les 2 premiers axes représentent 47%.

library(factoextra)
fviz_eig(PCA_Carac)

# le cercle des dimensions
fviz_pca_var(PCA_Carac)
fviz_pca_ind(PCA_Carac, col.ind="cos2",label = "none")  # ça prends un temps fou à tracer car trop de points.
# fviz_pca_ind(PCA_Carac, axes=c(1,2), col.ind="cos2", alpha.ind="cos2", alpha.label="cos2")
fviz_pca_biplot(PCA_Carac, axes = c(1, 2), habillage=caracteristiques_quali$Agglomeration,label = "var") + theme_minimal() +
    xlim(-4, 4) + ylim (-5, 4)




# -----------
# Clustering
# -----------
ClustersKmeans <- kmeans(caracteristiques_Dummy, 3) # 3 clusters pour commencer

# Accès aux données du cluster 1
mode(ClustersKmeans$cluster)

Cluster1<-caracteristiques_filter[ClustersKmeans$cluster==1,]
summary(Cluster1)
Cluster2<-caracteristiques_filter[ClustersKmeans$cluster==2,]
summary(Cluster1)
Cluster3<-caracteristiques_filter[ClustersKmeans$cluster==3,]
summary(Cluster1)

# Jointure sur 2017
ClusterNo<-as.character(ClustersKmeans$cluster)
caracteristiques_WithCluster<-cbind(caracteristiques_filter,ClustersKmeans$cluster,ClusterNo)

names(caracteristiques_WithCluster)
ggplot(caracteristiques_WithCluster) +
    aes(x = Lumière, color=ClustersKmeans$cluster) +
        geom_histogram(stat="count") +
        coord_flip()+
        labs(x="Lumière", y="Count"
         , color="ClustersKmeans$cluster", shape="ClustersKmeans$cluster",title="Titre")

# Regardons le niveau de luminosité par Cluster
Horiz_bar_count_group(data=caracteristiques_WithCluster,colonne="Lumière",group="ClusterNo",strTitle="Niveau de Luminosité")
#Les accidents de plein jour sont principalement dans le Cluster 1. Bon.OK.


# Clustering : idée / notes prises au moment du debriefing
ClustersKmeans_5 <- kmeans(caracteristiques_Dummy, 5)
# puis selectionner les centres : ClustersKmeans$centers
# puis faire une ACP sur les centres
# et espérer que le résultat de 'ACP fonctionne bien


