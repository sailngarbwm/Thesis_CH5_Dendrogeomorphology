import os
import geopandas as gpd 
from rasterstats import zonal_stats

import pandas as pd 


cat_list = ['Bris', 'Mary', 'NPR', 'kobble']

multi_lidar = ['Bris', 'Mary']

Mary_lid = {'mary_lidar_1.asc':['MR1'],
            'mary_lidar_1.asc':['MR2','MR3']
            }
Bris_lid = {
            'bris_dem_1.asc':['UBR1', 'UBR2', 'UBR3'],
            'bris_dem_2.asc':['UBR4']
}

ml_dict = {k:v for k, v in zip(multi_lidar, [Bris_lid, Mary_lid])}

sl = {'NPR':'pine_lidar_clip.asc',
        'kobble':'kobble_lidar.asc' }
stat = ['min', 'max', 'mean', 'std']
for cat in cat_list:

    mbuf = gpd.read_file(cat+'_merged_buf.geojson')

    if cat in sl.keys():
        stats = zonal_stats(list(mbuf.geometry),sl[cat], stats=stat)
        stats = pd.DataFrame(stats)

        stats['Reach'] = mbuf['reach']
        stats['name'] = mbuf['name']
        stats['type'] = mbuf['type']
        stats.to_csv(cat+'_elev_stats.csv')
        print(cat, 'finished')
    elif cat in ml_dict.keys():
        statl = []
        mld = ml_dict[cat]
        for lid in mld.keys():
            rl = mld[lid]
            def filt(x, r=rl):
                return (x in r)
            df_filt = mbuf['Reach'].apply(filt)
            mbf =mbuf[df_filt]

            stats = zonal_stats(list(mbf.geometry), lid, stats=stat)
            stats = pd.DataFrame(stats)
            stats['Reach'] = mbuf['reach']
            stats['name'] = mbuf['name']
            stats['type'] = mbuf['type']
            statl.append(stats)
        statsc = pd.concat(statl)
        statsc.to_csv(cat+'_elev_stats.csv')
        print(cat, 'finished')
        


