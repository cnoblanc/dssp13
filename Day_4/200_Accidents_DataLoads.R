
# ----------------
# Jeu de données : Accidents
# ---------------
# 1) Choix du problème
# 2) Calculs / Visualisations
# 3) Visualisation Shiny ou Dasboard RMarkdown
# --------------------------------------------
library(tidyverse)

source("100_Accidents_ETL.R")

# Check the loading warning :
# Caracteristiv 2009, CSV rownum =8 : Num_Acc='200900000008'
#filter(caracteristiques,Num_Acc=='200900000008')
# record loaded with NA for longitude.

summary(caracteristiques)
summary(usagers)
# Récupérer les valeurs distincts
usagers %>% 
    distinct(catu)
summary(vehicules)

# --------
# Préparation de la table dénormalisée
# ----------
denorm<-caracteristiques %>%
    inner_join(usagers, by="Num_Acc") %>%
    inner_join(vehicules,by=c("Num_Acc", "num_veh")) %>%
    inner_join(lieux, by="Num_Acc") 

summary(denorm)

# --------
# Transformation des variables qualitative
# ----------
# Usagers
summary(usagers)
mode(usagers)
names(usagers)
length(usagers) # nombre de colonnes
glimpse(usagers)

# Codage de la gravité
grav_ord <- c("Indemne", "Blessé léger", "Blessé hospitalisé", "Tué")
grav_code<-c(0, 10, 50, 200)

library(plyr)
grav_code<-mapvalues(usagers$grav,from=grav_ord,to=grav_code)
grav_num<-as.numeric(grav_code)
usagers[,"grav_num"]<-grav_num

#usagers[,"grav_num"] <- apply(usagers[1:200,"grav"], 1
#                              , mapvalues, from = grav_ord, to = grav_code) %>% as.numeric

# Vehicules
summary(vehicules)
# remove colonnes : senc, occutc, num_veh
# Garder : catv, choc (Other), manv (Other)
# Enlever NA : obs, obsm
mode(vehicules)
names(vehicules)
length(vehicules) # nombre de colonnes

# Copier Vehicules
vehicules_dq<-vehicules
summary(vehicules_dq)
# Supprimer des colonnes
vehicules_dq$senc<-NULL
vehicules_dq$num_veh<-NULL
vehicules_dq$occutc<-NULL
vehicules_dq$obs<-NULL
vehicules_dq$obsm<-NULL
summary(vehicules_dq)

# remplacer les valeurs NULL
# is.na(vehicules_dq$choc)
# vehicules_dq[is.na(vehicules_dq$choc)]
# vehicules_dq$choc[is.na(vehicules_dq$choc)]<-"(Other)"
# Choc est un type level.
levels(vehicules_dq$choc) # Afficher les levels
levels(vehicules_dq$choc)<-c(levels(vehicules_dq$choc),"(Other)") # Ajouter un autre level
vehicules_dq$choc[is.na(vehicules_dq$choc)]<-"(Other)" # remplacer les NULL

levels(vehicules_dq$manv) # Afficher les levels
levels(vehicules_dq$manv)<-c(levels(vehicules_dq$manv),"(Other)") # Ajouter un autre level
vehicules_dq$manv[is.na(vehicules_dq$manv)]<-"(Other)" # remplacer les NULL
# Renommage colonne
names(vehicules_dq)<-c("Num_Acc","vehic_category","vehic_choc","vehic_manoeuvre")

#TODO : les LIEUX
# to do : le nombre voies : caper à 10 voies ? (pas normal d'en avoir 99 !)

# ------------------------
# Les Caracteristiques des accidents
# ------------------------
mode(caracteristiques)
names(caracteristiques)
length(caracteristiques) # nombre de colonnes

# Copier Caracteristiques
caracteristiques_dq<-caracteristiques
summary(caracteristiques_dq)

# Supprimer les colonnes inutiles
#caracteristiques_dq$date<-NULL
caracteristiques_dq$com<-NULL
caracteristiques_dq$adr<-NULL
caracteristiques_dq$gps<-NULL
caracteristiques_dq$lat<-NULL
caracteristiques_dq$long<-NULL

# intersections est un type level.
levels(caracteristiques_dq$int) # Afficher les levels
levels(caracteristiques_dq$int)<-c(levels(caracteristiques_dq$int),"(Inconnu)") # Ajouter un autre level
caracteristiques_dq$int[is.na(caracteristiques_dq$int)]<-"(Inconnu)" # remplacer les NULL

# atm : est un level aussi
levels(caracteristiques_dq$atm)
caracteristiques_dq$atm[is.na(caracteristiques_dq$atm)]<-"Autre"

ggplot(caracteristiques_dq, aes(x=atm)) + geom_histogram(stat="count", aes(fill = int))
# map values : transformer une colonne en une autre
library(plyr)
atm_code<-mapvalues(caracteristiques_dq$atm
    ,from=c("Normale","Pluie légère", "Pluie forte","Neige - grêle","Brouillard - fumée"    
            ,"Vent fort - tempête","Temps éblouissant","Temps couvert"      
            ,"Autre" )
    ,to=c(0,10,50,100,60
          ,70,80,5,NA)
)


# col : collisions
levels(caracteristiques_dq$col)
levels(caracteristiques_dq$col)<-c(levels(caracteristiques_dq$col),"(Inconnu)") # Ajouter un autre level
caracteristiques_dq$col[is.na(caracteristiques_dq$col)]<-"(Inconnu)" # remplacer les NULL

# Departement
caracteristiques_dq$departement=as.numeric(caracteristiques_dq$dep)
caracteristiques_dq$dep<-NULL

# Ajout de l'année
install.packages(c("lubridate", "magrittr"))
library("lubridate")
library("magrittr")
caracteristiques_dq$year<-year(caracteristiques_dq$date)
caracteristiques_dq$date<-NULL

# Renommage des colonnes
names(caracteristiques_dq)<-c("Num_Acc","Lumière","Agglomération","Intersection","Atmosphère","Collision","Département","Année")
summary(caracteristiques_dq)



# ------------------------
# Les Caracteristiques des accidents : One Hot encoding
# ------------------------
#install.packages("lme4")
library(lme4)

# Filter Data Source
#caracteristiques_filter<-caracteristiques_dq[caracteristiques_dq$Année==2017]
#df<- as.data.frame(caracteristiques_dq)
#caracteristiques_filter=df[,Année==2017]
caracteristiques_filter<-filter(caracteristiques_dq, Année==2017)

# Transformer les colonnes en Dummy
#caracteristiques_Dcummy<-as.list(caracteristiques_filter$Num_Acc)
#names(caracteristiques_Dummy)<-c("Num_Acc")

caracteristiques_Dummy<-NULL
for (colonnes in c("Lumière","Agglomération","Intersection","Atmosphère","Collision")) {
    df <- dummy(caracteristiques_filter[[colonnes]])
    caracteristiques_Dummy<-cbind(caracteristiques_Dummy,df)
}

summary(caracteristiques_Dummy)
