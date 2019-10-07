
# Day_1_Lab/table_hrsample.html

# Installation package d'accès base de données
#install.packages("DBI")
library(DBI)

# Installer la base de données fournie
#install.packages("digest")
library(digest)

# Standard way :
#install.packages("/Users/Shared/Documents Christophe/DSSP/DSSP_Cours/Day_1_Lab/MonetDBLite_0.6.0.tgz", repos = NULL, type = .Platform$pkgType,dependencies = TRUE)
#install.packages("MonetDBLite")

#-----------------------
# Installation MonetDBLite : https://www.rdocumentation.org/packages/MonetDBLite/versions/0.6.0
# Pour cela, il faut d'abord installer les "devtools" :
# install.packages("devtools")
# Mais cela nécessite aussi d'installer des "outils de developement complémentaires" sur macOS :
# https://www.cnet.com/how-to/install-command-line-developer-tools-in-os-x/
#   Therefore, to install these tools, simply open the Terminal,
#   type "make" or any desired common developer command, and press Enter, 
#   and then when prompted you can install the developer tools, and be up and running.
# (c'est pas si simple, mais on y arrive)

# Pour vérifier que les devtools sont là : 
pkgbuild::check_build_tools(debug = TRUE)

# A partir de là, on peut installer les devtools :
#install.packages("devtools")
library(devtools)

# Puis récupérer les sources sur GitHub :
#devtools::install_github("hannesmuehleisen/MonetDBLite-R")
# Là, c’est un peu long, car il compile.
# Puis, ensuite :
library(MonetDBLite)
# c’est passé !

# Et là, on peut (re)commencer.
# Définir son répertoire de travail, où se trouve la base de données
setwd("/Users/Shared/Documents Christophe/DSSP/DSSP_Cours/Day_1_Lab")
conn <- dbConnect(MonetDBLite::MonetDBLite(),"hrsample_db")

dbListTables(conn)
dbGetQuery(conn,"SELECT * FROM deskjob LIMIT 10")

# Maximum Salary
dbGetQuery(conn,"SELECT max(salary) as maxSalary FROM salaryhistory")
# 1.882.292

# Who was paid this salary?
dbGetQuery(conn,"
SELECT * from salaryhistory
order by salary desc LIMIT 1
")

# What is the average salary by position?
resultat<-dbGetQuery(conn,"
SELECT job_name 
from deskjob d
    left join deskhistory dh on d.desk_id=dh.desk_id
")
View(resultat)


# Accès à la base de données avec des fonctions R
library(dbplyr)
load("tbls.RData")
#install.packages("tidyverse")
library(tidyverse)

deskhistory_tbl
glimpse(deskhistory_tbl)

filter(salaryhistory_tbl, 
       salary_effective_date >= lubridate::dmy("01-01-2000"),
       salary_effective_date <= lubridate::dmy("31-12-2000"))

select(salaryhistory_tbl,
       employee_num, salary)

# Créer une nouvelle colonne 
mutate(salaryhistory_tbl,
       salary_effective_year = lubridate::year(salary_effective_date))
derniere_valeur=.Last.value
derniere_valeur


# CSV
write_csv(salaryhistory_tbl, "salaryhistory_test.csv")
salaryhistory_csv <- read_csv("salaryhistory_test.csv")



