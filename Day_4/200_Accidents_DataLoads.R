
# ----------------
# Jeu de données : Accidents
# ---------------a
# 1) Choix du problème
# 2) Calculs / Visualisations
# 3) Visualisation Shiny ou Dasboard RMarkdown
# --------------------------------------------
library(tidyverse)
library(plyr)

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
# Transformation des variables qualitative
# ----------
# Usagers
# ----------
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


# ----------
# Vehicules
# ----------
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

# agg : agglomération ou pas. Create a numeric codification
levels(caracteristiques_dq$agg)
caracteristiques_dq$agg_code<-as.numeric(as.character(mapvalues(caracteristiques_dq$agg
                    ,from=c("Hors agglomération","En agglomération" )
                    ,to=c(-100,100)
)))
# lum : Niveau de luminosité. Create a numeric codification
levels(caracteristiques_dq$lum)
caracteristiques_dq$lum_code<-as.numeric(as.character(mapvalues(caracteristiques_dq$lum
                ,from=c("Plein jour","Crépuscule ou aube","Nuit avec éclairage public allumé"
                        ,"Nuit avec éclairage public non allumé","Nuit sans éclairage public")
                ,to=c(100,50,20,5,0)
)))

# int : intersections est un type level. On les regroupes pour en faire un code
levels(caracteristiques_dq$int) # Afficher les levels
caracteristiques_dq$int_code = forcats::fct_collapse(caracteristiques_dq$int,
        "Other"=NA, "Hors Intersection"="Hors intersection",
        "Avec Intersection"=c("Intersection en X","Intersection en T","Intersection en Y",
                           "Intersection à plus de 4 branches","Giratoire","Place",
                           "Passage à niveau","Autre intersection"),
        group_other=TRUE)
levels(caracteristiques_dq$int_code)<-c(levels(caracteristiques_dq$int_code),"(Inconnu)") # Ajouter un autre level
caracteristiques_dq$int_code[is.na(caracteristiques_dq$int_code)]<-"(Inconnu)" # remplacer les NULL
caracteristiques_dq$int_code<-as.numeric(as.character(mapvalues(caracteristiques_dq$int_code
                ,from=c("(Inconnu)","Other","Hors Intersection","Avec Intersection")
                ,to=c(0,0,-100,100)
)))
#fct_count(caracteristiques_dq$int_code)

# atm : atmosphère. transformer une colonne en une autre
levels(caracteristiques_dq$atm)<-c(levels(caracteristiques_dq$atm),"(Inconnu)") # Ajouter un autre level
caracteristiques_dq$atm[is.na(caracteristiques_dq$atm)]<-"(Inconnu)" # remplacer les NULL
caracteristiques_dq$atm_code<-as.numeric(as.character(mapvalues(caracteristiques_dq$atm
            ,from=c("Normale","Pluie légère", "Pluie forte","Neige - grêle","Brouillard - fumée"    
                    ,"Vent fort - tempête","Temps éblouissant","Temps couvert"      
                    ,"Autre","(Inconnu)" )
            ,to=c(100,10,-90,-100,-10
                  ,-50,50,8,0,0)
)))
#fct_count(caracteristiques_dq$atm_code)

# col : collisions. Regroupement et codage : avec/sans collision
levels(caracteristiques_dq$col) # Afficher les levels
caracteristiques_dq$col_code = forcats::fct_collapse(caracteristiques_dq$col,
         "Other"=NA, "Sans collision"="Sans collision",
         "Avec collision"=c("Deux véhicules - frontale","Deux véhicules – par l’arrière",
                            "Deux véhicules – par le coté","Trois véhicules et plus – en chaîne",
                            "Trois véhicules et plus - collisions multiples","Autre collision"),
         group_other=TRUE)
levels(caracteristiques_dq$col_code)<-c(levels(caracteristiques_dq$col_code),"(Inconnu)") # Ajouter un autre level
caracteristiques_dq$col_code[is.na(caracteristiques_dq$col_code)]<-"(Inconnu)" # remplacer les NULL
caracteristiques_dq$col_code<-as.numeric(as.character(mapvalues(caracteristiques_dq$col_code
                                        ,from=c("(Inconnu)","Other","Avec collision","Sans collision")
                                        ,to=c(0,0,-100,100)
)))
#fct_count(caracteristiques_dq$int_code)


# Departement
#caracteristiques_dq$departement=as.numeric(caracteristiques_dq$dep)
caracteristiques_dq$dep<-NULL # Enlever le département original

# Ajout de l'année
#install.packages(c("lubridate", "magrittr"))
library("lubridate")
library("magrittr")
caracteristiques_dq$year<-year(caracteristiques_dq$date)
caracteristiques_dq$date<-NULL

# Renommage des colonnes
names(caracteristiques_dq)
names(caracteristiques_dq)<-c("Num_Acc","Lumiere","Agglomeration","Intersection","Atmosphere",
                              "Collision","Code_Agglomeration","Code_Lumiere","Code_Intersection",
                              "Code_Atmosphere","Code_Collision","Annee")
summary(caracteristiques_dq)

# ------------------------
# Les Caracteristiques des accidents : One Hot encoding
# ------------------------
#install.packages("lme4")
library(lme4)

# Filter Data Source
caracteristiques_filter<-filter(caracteristiques_dq, Annee==2017)

# Transformer les colonnes en Dummy (plus besoin car convertit en numérique déjà)
#caracteristiques_Dummy<-NULL
#for (colonnes in c("Lumière","Agglomération","Intersection","Atmosphère","Collision")) {
#    df <- dummy(caracteristiques_filter[[colonnes]])
#    caracteristiques_Dummy<-cbind(caracteristiques_Dummy,df)
#}
#caracteristiques_Dummy<-caracteristiques_filter[,c("Num_Acc","Code_Agglomeration","Code_Lumiere","Code_Intersection",
#                                              "Code_Atmosphere","Code_Collision")]
caracteristiques_Dummy<-caracteristiques_filter
summary(caracteristiques_Dummy)


# --------
# Préparation de la table dénormalisée
# ----------
#enorm<-caracteristiques %>%
#    inner_join(usagers, by="Num_Acc") %>%
#    inner_join(vehicules,by=c("Num_Acc", "num_veh")) %>%
#    inner_join(lieux, by="Num_Acc") 
#summary(denorm)
