import os
import math
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from math import sin, cos, sqrt, atan2, radians

from sklearn.metrics import mean_squared_error,mean_absolute_error

base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF/"

def loadDVF_Maisons(departement='All',refresh_force=False,add_commune=True
                    ,year='All',filterColsInsee="None"):
    engine = create_engine('postgresql://christophe:christophe@localhost:5432/dv3f')
    mutation_fileName=base_path+"data_parquet/mutation_france.parquet"
    local_fileName=base_path+"data_parquet/local_france.parquet"
    parcelle_fileName=base_path+"data_parquet/parcelle_france.parquet"
    adresse_fileName=base_path+"data_parquet/adresse_france.parquet"
    all_fileName=base_path+"data_parquet/all_france.parquet"
    
    mutation_f = Path(mutation_fileName)
    if refresh_force==True or not mutation_f.is_file():
        print("Refreshing : Mutations")
        sql_query=("select m.idmutation,m.idmutinvar,m.datemut,m.valeurfonc,m.sterr" 
        " ,ST_X(ST_Transform (ST_Centroid(m.geomlocmut),4326)) as geoLong"
        " ,ST_Y(ST_Transform (ST_Centroid(m.geomlocmut),4326)) as geoLat"
        " from dvf.mutation as m " 
        " where m.codtypbien ='111' "  # only one House
        " and m.idnatmut = 1 and m.nbcomm=1 and m.vefa=false"
        " and m.valeurfonc>10 and m.sterr != 0 "
        )
        mutation_df = pd.read_sql_query(sql_query,con=engine)
        
        mutation_df['idmutation']=mutation_df['idmutation'].astype(int)
        mutation_df['idmutinvar']=mutation_df['idmutinvar'].astype(str)
        mutation_df['datemut'] = pd.to_datetime(mutation_df['datemut'], format="%Y-%m-%d")
        mutation_df['valeurfonc']=mutation_df['valeurfonc'].astype(int)
        mutation_df['sterr']=mutation_df['sterr'].astype(int)
        mutation_df['geolong']=mutation_df['geolong'].astype(float)
        mutation_df['geolat']=mutation_df['geolat'].astype(float)
        # Save File to Parquet
        print("Save to local parquet file")
        mutation_df.to_parquet(mutation_fileName, engine='fastparquet',compression='GZIP')
        mutation_refresh=True
    else:
        print("Read Mutations")
        mutation_df=pd.read_parquet(mutation_fileName, engine='fastparquet')
        mutation_refresh=False

    local_f = Path(local_fileName)
    if refresh_force==True or not local_f.is_file():
        print("Refreshing : Local")
        sql_query=("select l.nbpprinc,l.sbati,l.iddispoloc,l.iddispopar,l.idmutation" 
        " from dvf.local as l"  
        " where l.codtyploc=1 and l.nbmutjour =1 "  
        )
        loc_df = pd.read_sql_query(sql_query,con=engine)
        #loc_df['nbpprinc']=loc_df['nbpprinc'].astype(int)
        #loc_df['sbati']=loc_df['sbati'].astype(int)
        loc_df['iddispoloc']=loc_df['iddispoloc'].astype(int)
        loc_df['iddispopar']=loc_df['iddispopar'].astype(int)
        loc_df['idmutation']=loc_df['idmutation'].astype(int)
        # Save File to Parquet
        print("Save to local parquet file")
        loc_df.to_parquet(local_fileName, engine='fastparquet',compression='GZIP') 
        loc_refresh=True
    else:
        print("Read Local")
        loc_df=pd.read_parquet(local_fileName, engine='fastparquet')
        loc_refresh=False

    parcelle_f = Path(parcelle_fileName)
    if refresh_force==True or not parcelle_f.is_file():
        print("Refreshing : Parcelle")
        sql_query=("select dp.idmutation,dp.iddispopar,p.idparcelle "
        " ,p.coddep||p.codcomm||p.prefsect||p.nosect as quartier"
        " ,p.coddep||p.codcomm as commune ,p.coddep as departement,p.noplan"
        " from dvf.disposition_parcelle as dp" 
        "      left join dvf.parcelle as p on dp.idparcelle=p.idparcelle"
        ) 
        par_df = pd.read_sql_query(sql_query,con=engine)
        par_df['idmutation']=par_df['idmutation'].astype(int)
        par_df['iddispopar']=par_df['iddispopar'].astype(int)
        par_df['idparcelle']=par_df['idparcelle'].astype(int)
        par_df['quartier']=par_df['quartier'].astype(str)
        par_df['commune']=par_df['commune'].astype(str)
        par_df['departement']=par_df['departement'].astype(str)
        par_df['noplan']=par_df['noplan'].astype(str)
        # Save File to Parquet
        print("Save to local parquet file")
        par_df.to_parquet(parcelle_fileName, engine='fastparquet',compression='GZIP') 
        par_refresh=True
    else:
        print("Read Parcelle")
        par_df=pd.read_parquet(parcelle_fileName, engine='fastparquet')
        par_refresh=False

    adresse_f = Path(adresse_fileName)
    if refresh_force==True or not adresse_f.is_file():
        print("Refreshing : Adresse")
        sql_query=("select al.iddispoloc,a.idadresse "
        " ,a.commune as communeLabel, a.codepostal, a.typvoie"
        " from dvf.adresse_local as al " 
        "      left join dvf.adresse as a on al.idadresse=a.idadresse" 
        ) 
        adr_df = pd.read_sql_query(sql_query,con=engine)
        adr_df['iddispoloc']=adr_df['iddispoloc'].astype(int)
        adr_df['idadresse']=adr_df['idadresse'].astype(int)
        adr_df['communelabel']=adr_df['communelabel'].astype(str)
        adr_df['codepostal']=adr_df['codepostal'].astype(str)
        adr_df['typvoie']=adr_df['typvoie'].astype(str)
        # Save File to Parquet
        print("Save to local parquet file")
        adr_df.to_parquet(adresse_fileName, engine='fastparquet',compression='GZIP') 
        adr_refresh=True
    else:
        print("Read Adresse")
        adr_df=pd.read_parquet(adresse_fileName, engine='fastparquet')
        adr_refresh=False
        
    # Add the Communes INSEE :
    communes_insee_refresh=False  
    if add_commune==True:
        communes_insee_df=load_communes_insee()
        # Drop un-necessary columns
        if filterColsInsee=="None":
            communes_insee_df = communes_insee_df.drop(columns=['LIBGEO','DEP','CP','REG'])
        if filterColsInsee=="Manual":
            # Add new features in Comunes
            communes_insee_df=transform_INSEE_features(communes_insee_df)
            # Drop un-necessary columns     
            Drop_column_list=['LIBGEO','DEP','SEG Environnement Démographique Obsolète','CP'\
                              ,'Score Croissance Entrepreneuriale', 'SYN MEDICAL'\
                              , 'Seg Cap Fiscale', 'Nb Camping', 'SEG Croissance POP'\
                              , 'DYN SetC', 'Score Démographique', 'Score équipement de santé BV'\
                              , 'Seg Dyn Entre', 'Score VA Région', 'Fidélité'\
                              , 'Evolution Pop %', 'Score Ménages', 'Nb Log Vacants'\
                              , 'Indice Démographique', 'Indice Ménages', 'Indice Fiscal Partiel'\
                              , 'Nb Ménages', 'Nb Homme', 'Nb Femme'\
                              , 'Capacité Fisc', 'Nb Occupants Résidence Principale']
            #print("colums to Drop =",Drop_column_list)
            communes_insee_df = communes_insee_df.drop(columns=Drop_column_list)
            
            Drop_column_list_TxNb=['Taux de Entreprises Secteur Services'\
                    ,'Taux de Majeurs','Taux de Création Industrielles','Nb Hotel'\
                    ,'Taux de Femme','Taux de pharmaciens Libéraux BV','Nb Atifs'\
                    ,'Nb pharmaciens Libéraux BV','Taux de Infirmiers Libéraux BV'\
                    ,'Nb Entreprises Secteur Services','Taux de Occupants Résidence Principale'\
                    ,'Taux de Ménages','Taux de Etudiants','Taux de Hotel','Taux de Résidences Secondaires'\
                    ,'Taux de Logement','Nb Création Services','Taux de Omnipraticiens BV'\
                    ,'Nb Majeurs','Nb de Commerce','Taux de Entreprises Secteur Commerce'\
                    ,'Taux de Logement Secondaire et Occasionnel','Taux de Création Services'\
                    ,'Taux de dentistes Libéraux BV','Taux de Atifs','Taux de Entreprises Secteur Industrie'\
                    ,'Taux de Mineurs','Taux de Actifs Non Salariés','Taux de Camping'\
                    ,'Taux de Entreprises Secteur Construction','Nb Infirmiers Libéraux BV'\
                    ,'Taux de propriétaire','Nb Logement Secondaire et Occasionnel'\
                    ,'Nb de Services aux particuliers','Nb Actifs Non Salariés'\
                    ,'Nb Résidences Principales','Taux de Actifs Salariés','Nb Entreprises Secteur Commerce'\
                    ,'Nb Entreprises Secteur Industrie','Nb Etudiants','Nb Education, santé, action sociale'\
                    ,'Nb Omnipraticiens BV','Nb propriétaire','Taux de Services personnels et domestiques'\
                    ,'Nb Entreprises Secteur Construction','Taux de Santé, action sociale'\
                    ,'Nb Création Industrielles','Taux de de Commerce','Nb dentistes Libéraux BV'\
                    ,'Taux de Education, santé, action sociale','Taux de Pharmacies et parfumerie'\
                    ,'Taux de Industries des biens intermédiaires','Nb Industries des biens intermédiaires'\
                    ,'Nb Création Enteprises','Nb Création Construction','Nb Mineurs'\
                    ,'Nb Santé, action sociale','Nb Résidences Secondaires'\
                    ,'Taux de institution de Education, santé, action sociale, administration'\
                    ,'Nb institution de Education, santé, action sociale, administration'\
                    ,'Nb Pharmacies et parfumerie']
            communes_insee_df = communes_insee_df.drop(columns=Drop_column_list_TxNb)
            
            Drop_column_list_001=['Score Croissance Population','Indice Evasion Client'\
                    ,'Nb Création Commerces','Capacité Camping','Dep Moyenne Salaires Employé Horaires'\
                    ,'Taux de Résidences Principales','Score Urbanité','Population','REG'\
                    ,'Dynamique Entrepreneuriale Service et Commerce','Score PIB','Dep Moyenne Salaires Horaires'\
                    ,'Dep Moyenne Salaires Cadre Horaires','Dynamique Démographique BV','Taux de Homme'\
                    ,'Reg Moyenne Salaires Employé Horaires','Taux de Log Vacants','Taux étudiants'\
                    ,'Score Fiscal','Dep Moyenne Salaires Prof Intermédiaire Horaires','Evolution Population'\
                    ,'Taux Evasion Client','Densité Médicale BV','Score Synergie Médicale'\
                    ,'Reg Moyenne Salaires Horaires','Reg Moyenne Salaires Prof Intermédiaire Horaires'\
                    ,'Dynamique Entrepreneuriale','Capacité Fiscale','Taux Propriété','Score Evasion Client'\
                    ,'Taux de Création Construction','Nb Logement','Synergie Médicale COMMUNE'\
                    ,'Reg Moyenne Salaires Ouvrié Horaires','Indice Synergie Médicale','Reg Moyenne Salaires Cadre Horaires'\
                    ,'Orientation Economique','Dep Moyenne Salaires Ouvrié Horaires','Moyenne Revenus Fiscaux Départementaux'\
                    ,'Capacité Hotel','Moyenne Revnus fiscaux'\
                    ,'Taux de de Services aux particuliers','Taux de Création Enteprises'\
                    ,'Nb Services personnels et domestiques']
            communes_insee_df = communes_insee_df.drop(columns=Drop_column_list_001)
            
            #Columns_to_keep=['Nb Services personnels et domestiques' \
            #        ,'Valeur ajoutée régionale','Moyenne Revenus Fiscaux Régionaux' \
            #        ,'Taux de Création Commerces','Nb Actifs Salariés','Environnement Démographique' \
            #        ,'PIB Régionnal','Urbanité Ruralité','Dynamique Démographique INSEE'\
            #        ,'commune']
            #communes_insee_df=communes_insee_df[Columns_to_keep]
            
            # END OF : filterColsInsee="Manual":
        if filterColsInsee=="Permutation":
            # Add new features in Comunes
            communes_insee_df=transform_INSEE_features(communes_insee_df)
            Columns_to_keep=['Taux Evasion Client','Nb Création Commerces'
                    ,'Reg Moyenne Salaires Prof Intermédiaire Horaires','Urbanité Ruralité'
                    ,'Nb Ménages','Reg Moyenne Salaires Cadre Horaires','Taux de Hotel'
                    ,'Taux de Mineurs','Taux de dentistes Libéraux BV','Nb Création Enteprises'
                    ,'Dep Moyenne Salaires Horaires','Taux de Occupants Résidence Principale'
                    ,'Taux de Homme','commune']   
            communes_insee_df=communes_insee_df[Columns_to_keep]
            # END OF : filterColsInsee="Permutation":
        
    # After getting the tables : Merge All
    print("Make the join for DVF")
    mut_loc = pd.merge(mutation_df, loc_df, how='left', on=['idmutation'])
    mut_loc_par = pd.merge(mut_loc, par_df, how='left', on=['idmutation','iddispopar'])
    mut_dvf_all = pd.merge(mut_loc_par, adr_df, how='left', on=['iddispoloc'])
  
    # Filter data
    print("Filter data:")
    if departement=='All':
        pass
    elif departement=='Metropole':
        mut_dvf_all=mut_dvf_all[mut_dvf_all['departement']<='96'].copy()
    else: # We consider a filter on the departement
        mut_dvf_all=mut_dvf_all[mut_dvf_all['departement']==departement].copy()
    mut_dvf_all = mut_dvf_all[mut_dvf_all['departement']!='None'] # remove when we do not know where is the transaction

    # Filtre sur l'année
    if year!='All':
        mut_dvf_all=mut_dvf_all[str(mut_dvf_all['datemut'].year)==year]
    
    # Add infos from Communes INSEE
    if add_commune==True:
        print("Joining to add Commune INSEE")
        dvf_all = pd.merge(mut_dvf_all, communes_insee_df, how='left', on=['commune'])
    else:
        dvf_all = mut_dvf_all
    
    # Work on final data
    print("Final Calculations")
    dvf_all['n_days']=(dvf_all['datemut'].dt.date-pd.to_datetime("2013-01-01").date()).dt.days
    dvf_all['quarter']= pd.to_datetime(dvf_all['datemut']).dt.quarter
    #dvf_all['month']= pd.to_datetime(dvf_all['datemut']).dt.month
    
    dvf_all = dvf_all.drop(columns=['datemut','idmutation','idmutinvar','iddispoloc','iddispopar'
            ,'typvoie','idparcelle','noplan','idadresse'])
    dvf_all.dropna(axis=0, subset=['geolong'], inplace=True) # remove records which we do not know the geolong
    dvf_all.dropna(axis=0, subset=['geolat'], inplace=True) # remove records which we do not know the geolat

    
    if mutation_refresh or loc_refresh or par_refresh or adr_refresh or communes_insee_refresh:
        # Save to parquet file
        print("Save All France")
        dvf_all.to_parquet(all_fileName, engine='fastparquet',compression='GZIP')
        
    return dvf_all

def transform_INSEE_features(df):
    print("transform Features from INSEE")
    num_cols= df.select_dtypes([np.number]).columns
    col_name_list=num_cols[num_cols.str.startswith('Nb ')]
    col_name_list=np.array(col_name_list)
    #print("list of Nb columns",col_name_list)
    
    for col in col_name_list:
        # remove the "Nb " from the col name
        col_name=col[3:]
        # create the new Taux column
        #print("create new column:","Taux de "+col_name)
        df['Taux de '+col_name]= df['Nb '+col_name] / df['Population'] *100    
    
    return df

def update_category_features(df):
    cat_cols= df.select_dtypes([np.object]).columns
    
    # Convert 'SEG Croissance POP' to numeric
    #if 'SEG Croissance POP' in cat_cols:
        #map_val={'en déclin démographique': -1.0, 'en croissance démographique': 1.0}
        #df['Croissance POP']=df['SEG Croissance POP'].map(map_val, na_action='ignore')
        #df = df.drop(columns=['SEG Croissance POP'])
        
    # Convert category values None & NaN
    cat_cols= df.select_dtypes([np.object]).columns  
    for column in cat_cols:
        #print("->Column:",column)
        #print("NoNe values:",df[column].isnull())
        #df[df[column].isnull()][column]='None'
        df[column].replace(to_replace=[None], value=np.nan, inplace=True)
        values = df[column].unique()
        for value in values:
            if not isinstance(value, str) and np.isnan(value):
                df[column]=df[column].fillna('missing')   

    return df

def prepare_df(df, remove_categories=True):
    # Remove/filter the extrem values
    print("Prepare : filter extrem values")
    #selected_df=df[(df["valeurfonc"]<1000000) & (df["sterr"]<10000) & (df["nbpprinc"]<=10 ) & (df["nbpprinc"]>0) & (df["sbati"]<=500)]
    selected_df=df[ (df["sterr"]<10000) & (df["nbpprinc"]<=10 ) & (df["nbpprinc"]>0) & (df["sbati"]<=500)]
    
    print("Add Big cities per Departements Distance")
    departement_geoloc=load_geo_communes()
    df_dep_geo = pd.merge(selected_df, departement_geoloc, how='left', on=['departement'])
    #selected_df['department_city_dist']='geolong', 'geolat', 'DepBigCity_lat','DepBigCity_long'
    #cols_geo=df_dep_geo.columns
    # approximate radius of earth in km
    R = 6373.0
    lat1 = np.radians(df_dep_geo['geolat'])
    lon1 = np.radians(df_dep_geo['geolong'])
    lat2 = np.radians(df_dep_geo['DepBigCity_lat'])
    lon2 = np.radians(df_dep_geo['DepBigCity_long'])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan(np.sqrt(a) / np.sqrt(1-a))
    
    df_dep_geo['department_city_dist']= R * c
    #test=df_dep_geo[df_dep_geo['codepostal']=='77160']
    
    print("Prepare : drop geo categories")
    selected_df = df_dep_geo.drop(columns=['quartier','commune','departement' \
            ,'communelabel','codepostal','DepBigCity_lat','DepBigCity_long'])
    
    print("Prepare : update categories")
    selected_df=update_category_features(selected_df)
    
    if remove_categories == True :
        # Transform
        print("Prepare : transform categories")
        cat_cols= selected_df.select_dtypes([np.object]).columns
        selected_df = selected_df.drop(columns=cat_cols)
        
    return(selected_df)

def print_cols_infos(df):
    # Get list of columns by type
    cat_cols= df.select_dtypes([np.object]).columns
    num_cols = df.select_dtypes([np.number]).columns
    print_col_vals=20
    
    print("-> Category Variables are :",cat_cols)
    for col in cat_cols:
        # print first 20 
        print("#### Column'",col,"' values (",len(df[col].unique()),") are:",df[col].unique()[:print_col_vals])
    print("-> Numeric Variables are :",num_cols)
    
    return None

def load_communes_insee():
    print("Read Communes INSEE")
    communes_insee_fileName=base_path+"data_parquet/communes_insee.parquet"
    com_insee_f = Path(communes_insee_fileName)
    if not com_insee_f.is_file():
        print("Refreshing : Communes INSEE")
        folder_path=os.path.join("data_INSEE_Communes")
        file_path=folder_path+"/MDB-INSEE-V2_2016.xls"
        communes_insee_df=pd.read_excel(io=file_path)   
        communes_insee_df = communes_insee_df.rename(columns={'CODGEO':'commune'})
        print("Save to local parquet file")
        communes_insee_df.to_parquet(communes_insee_fileName, engine='fastparquet',compression='GZIP')
        communes_insee_refresh=True
    else:
        print("Read Communes INSEE")
        communes_insee_df=pd.read_parquet(communes_insee_fileName, engine='fastparquet')   
    return communes_insee_df

def load_geo_communes():
    com_dep_geo_fileName=base_path+"data_parquet/communes_dep_geo.parquet"
    com_dep_geo_f = Path(com_dep_geo_fileName)
    if not com_dep_geo_f.is_file():
        print("Refreshing : Biggest Communes per Departement")
        # Prepare the list of main cities per departements
        folder_path=os.path.join("data_INSEE_Communes")
        file_path=folder_path+"/MDB-INSEE-V2_2016.xls"
        communes_insee_df=pd.read_excel(io=file_path)   
        communes_insee_df = communes_insee_df.rename(columns={'CODGEO':'commune'})
        communes_insee_df = communes_insee_df.rename(columns={'DEP':'departement'})
        
        commune_size_df=communes_insee_df[['commune','Population','LIBGEO','departement']]
        commune_size_df=commune_size_df.sort_values(by='Population', ascending=False)
        # Keep as well the biggest city for each departments
        # Get the Biggest cities by Departments
        idx = commune_size_df.groupby(['departement'])['Population'].transform(max) == commune_size_df['Population']
        cities_department=commune_size_df[idx]
    
        file_path=base_path+"/data_Geo/EU_Geo_Circos_Regions_departements_circonscriptions_communes_gps.csv"
        df_cities_csv=pd.read_csv(file_path,sep=';',decimal=".",dtype=np.str,encoding="utf_8")
        df_cities_csv = df_cities_csv.rename(columns={'code_insee':'commune'})
        df_geo_cities = df_cities_csv[['commune','latitude','longitude','éloignement']]
        df_geo_cities['latitude']=pd.to_numeric(df_cities_csv['latitude'].replace('NaN',''), errors='raise')
        df_geo_cities['longitude']=pd.to_numeric(df_cities_csv['longitude'].replace('NaN',''), errors='raise')
        # Missing geos
        missing_geo=df_cities_csv[pd.isna(df_cities_csv['latitude'])]
        # 2.834 missing on 36.840 total
        
        # remove duplicates
        df_geo_cities = df_geo_cities.drop_duplicates(['commune'],keep='first')
        cities_department_geo = pd.merge(cities_department, df_geo_cities
                                         , how='left', on=['commune'])
        cities_department_geo = cities_department_geo.rename(columns={'latitude':'DepBigCity_lat'})
        cities_department_geo = cities_department_geo.rename(columns={'longitude':'DepBigCity_long'})
        df=cities_department_geo[['departement','DepBigCity_lat','DepBigCity_long']]
        print("Save to local parquet file")
        df.to_parquet(com_dep_geo_fileName, engine='fastparquet',compression='GZIP')
    else:
        print("Read Biggest Communes per Departement")
        df=pd.read_parquet(com_dep_geo_fileName, engine='fastparquet')   
        
    return df


def get_predict_errors(y, y_pred):
    y_absolute_error=(y_pred-y).abs()
    y_squared_error=y_absolute_error**2
    y_absolut_error_pct= 100*((y_pred-y)/y).abs()
    
    #y_logarithm_error=
    
    return y_absolute_error.mean(), y_absolute_error.std() \
        ,y_absolut_error_pct.mean(), y_absolut_error_pct.std() \
        ,y_squared_error.mean(),y_squared_error.std() \
        ,math.sqrt(y_squared_error.mean()),math.sqrt(y_squared_error.std())
        


    