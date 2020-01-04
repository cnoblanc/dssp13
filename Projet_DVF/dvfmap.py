import os
import folium as flm
from folium.plugins import MarkerCluster

base_path="/Users/christophenoblanc/Documents/ProjetsPython/DSSP_Projet_DVF"

def create_map(df,name="map_france",subset=0):
    world_map = flm.Map(location=[0, 0], zoom_start=2)
    mc = MarkerCluster()

    if subset ==0 :
        df_geolocalized=df.dropna(subset=['geolat', 'geolong'])
        print("Start adding ",df_geolocalized.shape[0],"markers")
    else :
        df_geolocalized=df.dropna(subset=['geolat', 'geolong'])[:subset]
        print("Start adding ",subset,"markers")
    
    # Draw markers on the map.
    for index, row in df_geolocalized.iterrows():    
        popup_str="prix:"+str(row["valeurfonc"])+" euros<br>" \
            + "surface batie:"+str(row["sbati"])+" m2<br>"\
            + "nb pièces principales:"+str(row["nbpprinc"])+"<br>"\
            + "surface terrain:"+str(row["sterr"])+" m2<br>"
        mc.add_child(flm.Marker(location=[row["geolat"], row["geolong"]],
                     popup=popup_str))
    world_map.add_child(mc)
    print("Start saving HTML file")
    # Create and show the world_map.save('airports.html')
    fileName=os.path.join(base_path+"/map_saved", name + ".html")
    world_map.save(outfile=fileName)
    print("Map saved")
    return None


#def display(m, height=300):
#    """Takes a folium instance and embed HTML."""
#    m._build_map()
#    srcdoc = m.HTML.replace('"', '&quot;')
#    embed = HTML('<iframe srcdoc="{0}" '
#                 'style="width: 100%; height: {1}px; '
#                 'border: none"></iframe>'.format(srcdoc, height))
#    return embed
#display(world_map)