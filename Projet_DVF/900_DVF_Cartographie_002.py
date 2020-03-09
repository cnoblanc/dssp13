# Default imports
import numpy as np
%matplotlib nbagg
import matplotlib.pyplot as plt

import dvfdata, dvfmap


Departements=['01','02','03','04','05','06','07','08','09']

for dep in Departements:
    print("Start data for:",dep)
    df=dvfdata.loadDVF_Maisons(departement=dep,refresh_force=False,add_commune=False)
    print("Start map for:",dep)
    dvfmap.create_map(df,name="map_"+dep,subset=0)

for dep in range(10,100,1):
    print("Start data for:",str(dep))
    df=dvfdata.loadDVF_Maisons(departement=str(dep),refresh_force=False,add_commune=False)
    print("Start map for:",dep)
    dvfmap.create_map(df,name="map_"+str(dep),subset=0)

