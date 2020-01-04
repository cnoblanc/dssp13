import os
import math
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

from sklearn.metrics import mean_squared_error,mean_absolute_error

base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF"

def loadDVF_Maisons(departement='All',refresh_force=False,add_commune=True,year='All'):
    engine = create_engine('postgresql://christophe:christophe@localhost:5432/dv3f')
    mutation_fileName="data_parquet/mutation_france.parquet"
    local_fileName="data_parquet/local_france.parquet"
    parcelle_fileName="data_parquet/parcelle_france.parquet"
    adresse_fileName="data_parquet/adresse_france.parquet"
    all_fileName="data_parquet/all_france.parquet"
    communes_insee_fileName="data_parquet/communes_insee.parquet"
    
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
        com_insee_f = Path(communes_insee_fileName)
        if refresh_force==True or not com_insee_f.is_file():
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
        dvf_all = dvf_all.drop(columns=['LIBGEO','DEP','SEG Environnement Démographique Obsolète','CP'])
    else:
        dvf_all = mut_dvf_all
    
    # Work on final data
    print("Final Calculations")
    #dvf_all['n_days'] = dvf_all['datemut'].apply(lambda date: (date - pd.to_datetime("2013-01-01")).days)
    dvf_all = dvf_all.drop(columns=['datemut','idmutation','idmutinvar','iddispoloc','iddispopar'
            ,'typvoie','idparcelle','noplan','idadresse'])
    
    if mutation_refresh or loc_refresh or par_refresh or adr_refresh or communes_insee_refresh:
        # Save to parquet file
        print("Save All France")
        dvf_all.to_parquet(all_fileName, engine='fastparquet',compression='GZIP')
        
    return dvf_all

def category_features(df):
    cat_cols= df.select_dtypes([np.object]).columns
    
    if 'SEG Croissance POP' in cat_cols:
        if df['SEG Croissance POP']== 'en déclin démographique':
            df['Croissance POP']=-1.0
        elif df['SEG Croissance POP']== 'en croissance démographique':
            df['Croissance POP']=1.0
        else:
            df['Croissance POP']=0
        df = df.drop('SEG Croissance POP')
    
    return df


def get_predict_errors(y, y_pred):
    y_absolute_error=(y_pred-y).abs()
    y_squared_error=y_absolute_error**2
    y_absolut_error_pct= 100*((y_pred-y)/y).abs()
    
    return y_absolute_error.mean(), y_absolute_error.std() \
        ,y_absolut_error_pct.mean(), y_absolut_error_pct.std() \
        ,y_squared_error.mean(),y_squared_error.std() \
        ,math.sqrt(y_squared_error.mean()),math.sqrt(y_squared_error.std())
        


    