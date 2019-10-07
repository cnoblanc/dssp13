
# -------------------
# First Plot
# -------------------

library(ggplot2)
#install.packages("rlang")
library(rlang)


Horiz_bar_count <- function(data, colonne,strTitle){
    # convert strings to symbols
    x_var <- rlang::sym(colonne)
    graph<-ggplot(data) +
        aes(x=!! x_var) +
        geom_bar(stat="count")+
        coord_flip()+
        labs(title=strTitle) +
        theme_minimal()
    return(graph)
}
Horiz_bar_count_group <- function(data, colonne,group,strTitle){
    # convert strings to symbols
    x_var <- rlang::sym(colonne)
    group_var <- rlang::sym(group)
    graph<-ggplot(data) +
        aes(x=!! x_var, shape = !! group_var, color = !! group_var) +
        geom_bar(stat="count",aes(fill = !! group_var))+
        coord_flip()+
        labs(title=strTitle) +
        theme_minimal()
    return(graph)
}


# Table caracteristiques
mode(caracteristiques)
names(caracteristiques)
summary(caracteristiques)

Horiz_bar_count(data=caracteristiques,colonne="agg",strTitle="Agglomération")
Horiz_bar_count_group(data=caracteristiques,colonne="lum",group="agg",strTitle="Niveau de Luminosité")
Horiz_bar_count_group(data=caracteristiques,colonne="int",group="agg",strTitle="Intersection")
Horiz_bar_count_group(data=caracteristiques,colonne="atm",group="agg",strTitle="Conditions Météo")
Horiz_bar_count_group(data=caracteristiques,colonne="col",group="agg",strTitle="Avec ou sans Collision")
#Horiz_bar_count_group(data=caracteristiques,colonne="com",group="agg",strTitle="")
#Horiz_bar_count_group(data=caracteristiques,colonne="dep",group="agg",strTitle="")

# Table lieux
mode(lieux)
names(lieux)
summary(lieux)

Horiz_bar_count(data=lieux,colonne="catr",strTitle="Catégorie de Voie")
Horiz_bar_count(data=lieux,colonne="circ",strTitle="Type de circulation")
Horiz_bar_count(data=lieux,colonne="nbv",strTitle="Nombre de voies")
Horiz_bar_count(data=lieux,colonne="vosp",strTitle="Existance piste cyclable")
Horiz_bar_count(data=lieux,colonne="prof",strTitle="Profil de voie")

# Table Usagers
mode(usagers)
names(usagers)
summary(usagers)

Horiz_bar_count_group(data=usagers,colonne="place",group="sexe",strTitle="Place dans vehicule")
Horiz_bar_count_group(data=usagers,colonne="catu",group="sexe",strTitle="Categorie de l'Usager")
Horiz_bar_count_group(data=usagers,colonne="grav",group="sexe",strTitle="Gravité")
Horiz_bar_count_group(data=usagers,colonne="trajet",group="sexe",strTitle="Type de trajet")
Horiz_bar_count_group(data=usagers,colonne="secup",group="sexe",strTitle="Dispositif de sécurité")
Horiz_bar_count_group(data=usagers,colonne="secuu",group="sexe",strTitle="Existance Dispositif de sécurité")
Horiz_bar_count_group(data=usagers,colonne="locp",group="sexe",strTitle="localisation Usager")
Horiz_bar_count_group(data=usagers,colonne="actp",group="sexe",strTitle="?")
Horiz_bar_count_group(data=usagers,colonne="etatp",group="sexe",strTitle="Groupe ou Seul")
Horiz_bar_count_group(data=usagers,colonne="an_nais",group="sexe",strTitle="Année de naissance") # Quantitatif, mesure
Horiz_bar_count_group(data=usagers,colonne="grav_num",group="sexe",strTitle="Gravité Codé Numérique") # !! non proportionnel à la gravité
Horiz_bar_count_group(data=usagers,colonne="num_veh",group="sexe",strTitle="numero véhicule ?") # Fait partie de la clé vers véhicule

# Table Vehicules
mode(vehicules)
names(vehicules)
summary(vehicules)

Horiz_bar_count(data=vehicules,colonne="num_veh",strTitle="Numero de vehicule") # Fait partie de la clé ?
Horiz_bar_count(data=vehicules,colonne="senc",strTitle="?") # interressant ?
Horiz_bar_count(data=vehicules,colonne="catv",strTitle="Categorie Vehicule") # A regrouper ?
Horiz_bar_count(data=vehicules,colonne="occutc",strTitle="?") # Qu'est-ce que c'est ? un piège ?
Horiz_bar_count(data=vehicules,colonne="obs",strTitle="Obstacle") # peu renseigné.
Horiz_bar_count(data=vehicules,colonne="obsm",strTitle="?") # Qu'est-ce que c'est ? 
Horiz_bar_count(data=vehicules,colonne="choc",strTitle="Type de choc")
Horiz_bar_count(data=vehicules,colonne="manv",strTitle="Type de manoeuvre") # Prévoir des regroupements ?





