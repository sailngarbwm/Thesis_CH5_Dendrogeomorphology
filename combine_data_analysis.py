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

# combining the three datasets
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

#%% Loading elevation GIS data

bris_elev = gpd.read_file('bris_merged_buff_qgis.geojson')
bris_elev['new_name'] = bris_elev.name.apply(lambda x: str(x)+'_Bris')
bris_elev = pd.DataFrame(bris_elev.drop(columns = 'geometry'))


mary_elev = gpd.read_file('Mary_merged_buff_qgis.geojson')
mary_elev['new_name'] = mary_elev.name.apply(lambda x: str(x)+'_Mary')
mary_elev = pd.DataFrame(mary_elev.drop(columns = 'geometry'))


NPR_elev = gpd.read_file('NPR_merged_buff_qgis.geojson')
NPR_elev['new_name'] = NPR_elev.name.apply(lambda x: str(x)+'_NPR')
NPR_elev = pd.DataFrame(NPR_elev.drop(columns = 'geometry'))

kobble_elev =  gpd.read_file('kobble_merged_buf_QGIS.geojson')
kobble_elev['new_name'] = kobble_elev.name.apply(lambda x: str(x)+'_kobble')
kobble_elev = pd.DataFrame(kobble_elev.drop(columns = 'geometry'))

elev_stack = pd.concat([bris_elev, mary_elev, NPR_elev, kobble_elev])
#%%

types = list(kobble_elev.type.unique())
elev_statcols = [x for x in list(elev_stack.columns) if '_'==x[0]]

def unstack_pt_data(df,statcols,  types=types):
    """
    This function will unstack the point data by type 
    and line everything up by sample point
    """
    ct = 0

    
    for ty in types:
        stat_c_dict = {x:ty+x for x in statcols}
        if ct ==0:
           
            df_out = df[df['type']==ty].copy()
            df_out = df_out.rename(columns = stat_c_dict)

            ct =ct + 5
            print('ct', ct)
        elif ct>0:
            df_mer = df[df['type']==ty].copy()
            # print(ty, 'df size', df.shape, 'df_mer size', df_mer.shape)
            df_mer = df_mer.rename(columns = stat_c_dict).copy()
            attach_cols = list(stat_c_dict.values())+['new_name']

            df_out = df_out.merge(df_mer.loc[:,attach_cols],
                         on = 'new_name')
    return df_out

#%% unstack elevations and combine the gisdata

geom_comb.shape
df_elev_uns = unstack_pt_data(elev_stack, elev_statcols, types)

gis_dat_comb = geom_comb.merge(df_elev_uns, on ='new_name')

gis_dat_comb.head()
#%%

"""

do last couple cleans of field data  need to do geo units, and sediments
"""

field_dat.columns
field_dat.Geo_Unit.unique()

def geo_unit_cleaner(x):
    """
    logic to apply to geo unit column
    """

    if ('inset' in x.lower()) or ('i' in x.lower()) :
        return 'inset'
    elif 'bar' in x.lower():
        return 'bar'
    
    elif ('fp' in x.lower()) or ('floodplain' in x.lower()):
        return 'floodplain'
    elif 'mass failure' in x.lower():
        return 'mass failure'
    elif ('avulsion' in x.lower()) or ('old ch' in x.lower()):
        return 'avulsion'
    elif 'pool' in x.lower():
        return 'bed'
    elif 'stringer' in x.lower():
        return 'bar'
    elif 'abandoned channel' in x.lower():
        return 'avulsion'
    elif 'gravel channel' in x.lower():
        return 'bed'

    else:
        return 'NA'

field_dat['geo_unit_clean'] = field_dat['Geo_Unit'].apply(geo_unit_cleaner)

field_dat[field_dat['geo_unit_clean']=='NA'][['geo_unit_clean','Geo_Unit']]
    

#%% veg

field_dat.species.unique()

field_dat[field_dat['species'] == 'Casuarina dbh']


def clean_species(x):
    """
    now I will do a quick clean of species
    """
    x = x.lower()

    if 'casuarina' in x:
        return 'C. Cunninghamiana'
    elif 'acaci' in x: # note I mis identified these in the field
        return 'M. Bracteata'
    elif ('bottlebrush' in x) or ('bottle brush' in x) or ('viminalis' in x):
        return 'M. viminalis'
    elif 'gum' in x:
        return 'Eucalyptus'
    elif ('rainforest' in x) or ('rf' in x) or ('r forest' in x):
        return 'remnant rainforest species'
    else:
        return 'not veg'

field_dat['species_clean'] = field_dat['species'].apply(clean_species)

field_dat[['species_clean', 'species']]

#%% next we define the measurement type

def clean_meas_type(species, speciesclean):
    """
    get measurement type
    """

    if speciesclean != 'not veg':
        if ('dbh' in species.lower()) or ('inv' in species.lower()) or ('ct' in species.lower()):
            return 'dbh inventory'
        elif ('quad' in species.lower()) or  ('5x5' in species.lower()):
            return 'quadrat'
        else:
            return 'dbh burial'
    elif ('sed' in species.lower()) or ('soil' in species.lower()) or ('strat' in species.lower()):
        return 'sed strat column'
    else:
        return 'no meas type'


field_dat['meas_type'] = field_dat.apply(lambda x: clean_meas_type(x['species'], x['species_clean']), axis=1)

#%%
field_dat['species_clean'].unique()
field_dat[field_dat['meas_type']=='no meas type'][['meas_type', 'species']]
field_dat.columns

#%%
# next I need to check sed types

field_dat['structure category'].unique()

field_dat['smallest texture']

def ss_filter(x):
    '''
    going to try a real quick filter, where I just shave off the ses and turn medium int med
    '''
    x = str(x)
    xl = x.split(' ')
    # print('before', xl)
    new_l = []
    for w in xl:
        if w == 'medium':
            wout='med'
        elif len(w) >0:
            if w[-1] == 's':
                wout=w[:-1]
            else:
                wout = w
        else:
            wout = w
        new_l.append(wout)
    # print('after',new_l)
    out = ''
    for new in new_l:
        out+= new+' '
    
    return out[:-1]

field_dat['smallest texture clean'] = field_dat['smallest texture'].apply(ss_filter)
field_dat['biggest texture clean'] = field_dat['biggest texture'].apply(ss_filter)
field_dat['texture at base clean'] = field_dat['texture at base'].apply(ss_filter)

pd.DataFrame(field_dat['smallest texture clean'].unique())


pd.DataFrame(field_dat['biggest texture clean'].unique())

pd.DataFrame(field_dat['texture at base clean'].unique())

#%%
def ss_filter2(x):
    """
    this filter further simplfies everything
    """

    if ('silty' in x) and ('sand' in x):
        return 'loamy sand'
    elif 'sand' in x and 'loam' in x:
        return 'sand loam'
    elif (x=='gravel') or (x == 'sandy gravel'):
        return 'med gravel'
    else:
        return x


field_dat['smallest texture clean'] = field_dat['smallest texture clean'].apply(ss_filter2)
field_dat['biggest texture clean'] = field_dat['biggest texture clean'].apply(ss_filter2)
field_dat['texture at base clean'] = field_dat['texture at base clean'].apply(ss_filter2)

#%%

pd.DataFrame(field_dat['smallest texture clean'].unique())
pd.DataFrame(field_dat['biggest texture clean'].unique())
pd.DataFrame(field_dat['texture at base clean'].unique())

#%%combining stuff
field_dat.new_name

def new_name_fix(x):
    if x[0]=='0':
        return x[1:]
    else: return x
gis_dat_comb.new_name = gis_dat_comb.new_name.apply(new_name_fix)
elev_stack.new_name = elev_stack.new_name.apply(new_name_fix)

elev_stack.new_name
gis_dat_comb.new_name

"""
Now combine the data
"""

#%%
df_comb = field_dat.merge(gis_dat_comb, how='inner', on = 'new_name')

# df_comb = df_comb.merge(elev_stack,how= 'inner', on='new_name')


#%%

"""
begin analysis
"""

df_comb.describe()

# first  calculate root elevation

df_comb['sprout_elev'] = df_comb['sl_mean'] - df_comb['Root_depth']*.01

def rel_sprout_el(spr_el, thal_mean, lb_mean, rb_mean, lb_dist,rb_dist):

    if lb_dist > rb_dist:
        return (spr_el-thal_mean)/(lb_mean-thal_mean)
    else:
        return (spr_el-thal_mean)/(rb_mean-thal_mean)

rel_spr_el_ap = lambda x: rel_spr_el_ap(x['sprout_el'], x['thal_mean'])


df_comb['rel_sprout_el'] = 