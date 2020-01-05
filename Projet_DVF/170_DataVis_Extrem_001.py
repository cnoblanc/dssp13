import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import dvfdata
#df=dvfdata.loadDVF_Maisons(departement='77',refresh_force=False,add_commune=False)
df=dvfdata.loadDVF_Maisons(departement='All',refresh_force=False,add_commune=False)

# Get the list of columns
df.info()

# Get list of columns by type
cat_cols= df.select_dtypes([np.object]).columns
for col in cat_cols:
    print("Column'",col,"' values (",len(df[col].unique()),") are:",df[col].unique()[:10])
num_cols = df.select_dtypes([np.number]).columns


# ------------------------
# Get stats about our dataset
# ------------------------
#df_describe=df.describe()
print(cat_cols)
print(num_cols)

# General PairPlot
#g=sns.pairplot(data=df[num_cols])
#g.fig.suptitle("France : Pairplot of numeric features", y=1.05) 

# ------------------------
# Price outliers
# Analyze Valeur_fonciere
# ------------------------
sns.distplot(df["valeurfonc"],kde=False, rug=True)
plt.suptitle("France : Répartition du prix de vente", fontsize=12,y=0.95) 
plt.show()

# We see here that we have Outliers after 
# 1e9 : 1.000.000.000 = 1.000 millions euros = 1 milliard

# Less than 10 Millions
df_lessthan_10M=df[df["valeurfonc"]<10000000]
sns.distplot(df_lessthan_10M["valeurfonc"],kde=False, rug=True)
plt.suptitle("France : Répartition du prix inférieurs à 10 millions", fontsize=12,y=0.95) 
plt.show()

# Less than 1 million
df_lessthan_1ME=df[df["valeurfonc"]<1000000]
sns.distplot(df_lessthan_1ME["valeurfonc"],kde=False, rug=True)
plt.suptitle("France : Répartition du prix inférieurs à 1 millions", fontsize=12,y=0.95) 
plt.show()

print("nb de transactions total:",df.shape[0])
print("Count Transactions sup 1 million :", df.shape[0]-df_lessthan_1ME.shape[0])
print("% Transactions sup 1 million :", 100*(df.shape[0]-df_lessthan_1ME.shape[0])/df.shape[0])


# ------------------------
# Surface de Terrain outliers
# Analyze Surface du terrains
# ------------------------
sns.boxplot(data=df['sterr'], orient="v").set_title('Répartition des surfaces de terrains')
plt.show()

df_lessthan_Terr_1ha=df[df["sterr"]<10000]
sns.boxplot(data=df_lessthan_Terr_1ha['sterr'], orient="v").set_title('Répartition des surfaces de terrains (<1 ha)')
plt.show()

sns.distplot(df_lessthan_Terr_1ha["sterr"],kde=False, rug=True)
plt.suptitle("France : Répartition des surfaces de terrains (<1 ha)", fontsize=12,y=0.95) 
plt.show()

print("Count Transactions sup 1 ha :", df.shape[0]-df_lessthan_Terr_1ha.shape[0])
print("% Transactions sup 1 million :", 100*(df.shape[0]-df_lessthan_Terr_1ha.shape[0])/df.shape[0])


# ------------------------
# Nb de Pièces principales outliers
# Analyze des pièces principales
# ------------------------
sns.distplot(df["nbpprinc"],kde=False, rug=True)
plt.suptitle("France : Répartition du nombre de pièces principales", fontsize=12,y=0.95) 
plt.show()

df_lessthan_10_pieces=df[df["nbpprinc"]<=10 ]
sns.distplot(df_lessthan_10_pieces["nbpprinc"],kde=False, rug=True)
plt.suptitle("France : Répartition du nombre de pièces principales (<=10)", fontsize=12,y=0.95) 
plt.show()

print("Count Transactions sup 10 pieces :", df.shape[0]-df_lessthan_10_pieces.shape[0])
print("% Transactions sup 10 pieces :", 100*(df.shape[0]-df_lessthan_10_pieces.shape[0])/df.shape[0])

df_lessthan_10_pieces_sup_zero=df[df["nbpprinc"]>0 ]
print("Count Transactions à 0 pieces :", df.shape[0]-df_lessthan_10_pieces_sup_zero.shape[0])
print("% Transactions à 0 pieces :", 100*(df.shape[0]-df_lessthan_10_pieces_sup_zero.shape[0])/df.shape[0])

# ------------------------
# Surface bâtie outliers
# Analyze des surfaces bâties
# ------------------------
sns.distplot(df["sbati"],kde=False, rug=True)
plt.suptitle("France : Répartition de la surface bâtie", fontsize=12,y=0.95) 
plt.show()

df_lessthan_500_bati=df[df["sbati"]<=500 ]
sns.distplot(df_lessthan_500_bati["sbati"],kde=False, rug=False)
plt.suptitle("France : Répartition de la surface bâtie (<500 m2)", fontsize=12,y=0.95) 
plt.show()
print("Count Transactions inf 500 m2 :", df.shape[0]-df_lessthan_500_bati.shape[0])
print("% Transactions inf 500 :", 100*(df.shape[0]-df_lessthan_500_bati.shape[0])/df.shape[0])

# ------------------------
# Analyze the selected transactions
#  prix < 1Meuros, terrain<1Ha, 0< pieces <=10, bâtie <=500 m2 
# ------------------------
selected_df=df[(df["valeurfonc"]<1000000) & (df["sterr"]<10000) & (df["nbpprinc"]<=10 ) & (df["nbpprinc"]>0) & (df["sbati"]<=500)]
sns.pairplot(data=selected_df[num_cols])
plt.suptitle("France : Pairplot of numeric features for limited transactions", fontsize=14,y=1) 
plt.show()

# Très consommateur en ressources !!
#g = sns.PairGrid(selected_df[num_cols], diag_sharey=False)
#g.map_lower(sns.kdeplot)
#g.map_upper(sns.scatterplot)
#g.map_diag(sns.kdeplot, lw=3)
#plt.show()

