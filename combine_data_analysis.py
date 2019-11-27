"""

here I will bring the open data kit file and the gis analysis stuff together
"""
#%%
import pandas as pd 
import geopandas as gpd 
import os
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

#%%

# ok to deal with overlap, I am adding the name of the cathcment to the gps point, feck it

bris_geom = pd.read_csv('Bris_geom_data_LR.csv')
bris_geom['new_name'] = bris_geom.name.apply(lambda x: str(x)+'_Bris')
mary_geom = pd.read_csv('Mary_geom_data_LR.csv')
mary_geom['new_name'] = mary_geom.name.apply(lambda x: str(x)+'_Mary')

NPR_geom = pd.read_csv('NPR_geom_data_LR.csv')
NPR_geom['new_name'] = NPR_geom.name.apply(lambda x: str(x)+'_NPR')

kobble_geom = pd.read_csv('kobble_geom_data_LR.csv')
kobble_geom['new_name'] = kobble_geom.name.apply(lambda x: str(x)+'_kobble')


geom_comb = pd.concat([bris_geom, mary_geom, NPR_geom, kobble_geom])

geom_comb.new_name.unique().shape
len(geom_comb.name.unique())

doubles = list(geom_comb[geom_comb.duplicated(subset='new_name')].name.unique())
#%%
def inlist(x, doubles = doubles):
    return (x in doubles)
dble_filt = geom_comb.new_name.apply(inlist)

geom_comb[dble_filt]
#%%
#first problem is removing the recon
field_dat = pd.read_excel('working_dbh_burial_copy_from_odk.xlsx')
field_dat.Cat.unique()
#removes recon datasets
dayfilt = field_dat.Cat.apply(lambda x: x in ['NPR', 'kobble', 'Bris', 'Mary'])

field_dat = field_dat[dayfilt]

# adding the new names and stuff
field_dat['new_name'] = field_dat.apply(lambda x: str(x['GPS_point'])+'_'+x['Cat'], axis = 1)
field_dat['new_name'].unique().shape, geom_comb.new_name.unique().shape #some missing values
flddub = field_dat[field_dat.duplicated(subset='GPS_point')]['GPS_point'].unique()



fld_flt = field_dat['GPS_point'].apply(lambda x: inlist(x, doubles = flddub))
# note, it appears like some of these are intentional duplicates
field_dat[fld_flt].sort_values('GPS_point')