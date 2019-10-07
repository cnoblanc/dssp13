
#affectation de variable
variable=3

# générer une suite de nombre aléatoires
set.seed(42) # j'impose un point de départ
runif(1)
sample.int(10000,1)


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





