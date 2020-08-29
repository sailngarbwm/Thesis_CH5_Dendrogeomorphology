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
sns.set_context('paper')

#%
#%%
import sys


# def make_contour(z, x='veg_age', y = 'rel_elev_int', site ='143010B',df_pl = df_pl,  **kwargs):
#     df_pl = df_pl[df_pl['site'] == site]
#     df_tbgew = df_pl[[z,x,y]].groupby([x,y]).mean()
#     df_tbgew = df_tbgew.reset_index()
#     triangles = tri.Triangulation(df_tbgew[x], df_tbgew[y])
#     contourer = plt.tricontour(triangles, df_tbgew[z], **kwargs)
#     plt.colorbar(contourer)
#     plt.ylim(top = 100)
#     plt.title('site')

def bootstrap(data, n=1000, func=np.mean, p=40):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    xbar_init = np.mean(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()
    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx],simulations[u_indx])
    return(ci)

def y_lineplot(df, x, y,liner = 'mean' , err_method = 'CI', low = 5, high = 95, ax=None,fb_params={} ,**kwargs):
    df_plt = df[[x,y]].groupby(y).describe(percentiles = [low/100, 0.5, high/100])
    if ax == None:
        fig, ax = plt.subplots()
    if err_method == 'CI':
        ax.fill_betweenx(df_plt.index.values, df_plt[x][str(low)+'%'].values,
                         df_plt[x][str(high)+'%'].values, alpha = 0.5, **fb_params)
    elif err_method=='std':
        ax.fill_betweenx(df_plt.index.values, df_plt[x][liner].values -df_plt[x]['std'].values , 
                         df_plt[x][liner].values+df_plt[x]['std'].values, alpha = 0.5, **fb_params)
    elif err_method=='bootstrap':
        bs = df[[x,y]].groupby(y).apply(lambda z: bootstrap(z[x].values, n=1000)(0.95))
        bs = pd.DataFrame([[i,a,b] for i, (a, b) in zip(bs.index, bs.values)], columns = ['y',str(low), str(high)] )
        ax.fill_betweenx(bs['y'], bs[str(low)], 
                         bs[str(high)], alpha = 0.5, **fb_params)




    if liner == 'median':
        ax.plot(df_plt[x]['50%'].values, df_plt.index.values, **kwargs)

    elif liner == 'mean':  
        ax.plot(df_plt[x][liner].values, df_plt.index.values, **kwargs)
            
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.legend()


    return ax.get_figure()

def plot_subsets_vertical(df, x,y, delimiter, plot_list=None, ax=None, err_meth = 'bootstrap'):
    if ax==None:
        figv3, ax = plt.subplots()
    if plot_list ==None:
        plot_list = df[delimiter].unique()
    for v in plot_list:
        
        df_t = df[df[delimiter] == v]
        if df_t.shape[0] >0:
            # print(df.shape)
            figv3 = y_lineplot(df_t, x, y, 'mean', err_method=err_meth, low = 25,high =  75, ax = ax, label = v )

    plt.legend()
    return figv3


#%% Adding some bits to bring in the reach data

bris_cat = gpd.read_file('bris_catchment_data.geojson')
bris_cat.head()
Mary_cat = gpd.read_file('Mary_catchment_data.geojson')
Mary_cat.head()
kobble_cat = gpd.read_file('kobble_catchment_data.geojson')
kobble_cat.head()

npr_cat = gpd.read_file('NPR_catchment_data.geojson')
npr_cat.head()

#%%
for df in [bris_cat,Mary_cat,kobble_cat, npr_cat]:
    renamer = {}
    for col in list(df.columns):
        if 'lid' in col:
            if 'mean' in col:
                renamer[col] = 'lid_el_mean'
            elif 'stdev' in col:
                renamer[col] = 'lid_el_stdev'
        elif 'acc' in col:
            renamer[col] = 'flow_acc_max'
    if renamer:
        df.rename(columns = renamer, inplace = True)
# renamer
# npr_cat    

df_cat_comb = pd.concat([bris_cat,Mary_cat,kobble_cat, npr_cat], join='inner')
# df_cat_comb

#%%
reachl = []
t_el1 = []
t_el2 = []
rlen = []
catcharea = []
for reach in list(df_cat_comb['Reach'].unique()):
    # print(reach)
    dft = df_cat_comb[df_cat_comb['Reach']==reach]
    reachl.append(reach)
    t_el1.append(dft['lid_el_mean'].max())
    t_el2.append(dft['lid_el_mean'].min())
    rlen.append(dft['linelength'].iloc[0])
    catcharea.append(dft['flow_acc_max'].max()*25*25/(1000**2))

df_cats = pd.DataFrame({'Reach':reachl, 'top_el':t_el1, 
                        'bot_el':t_el2,'reach_len':rlen, "catch_area":catcharea})

#%%

df_cats['slope'] = (df_cats['top_el']-df_cats['bot_el'])/df_cats['reach_len']



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
#%
def inlist(x, doubles = doubles):
    return (x in doubles)
dble_filt = geom_comb.new_name.apply(inlist)

geom_comb[dble_filt]
#%
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

#% Loading elevation GIS data

bris_elev = gpd.read_file('bris_merged_buff_qgis.geojson')
bris_elev['new_name'] = bris_elev.name.apply(lambda x: str(x)+'_Bris')
# bris_elev = pd.DataFrame(bris_elev.drop(columns = 'geometry'))
# print(bris_elev.head())

mary_elev = gpd.read_file('Mary_merged_buff_qgis.geojson')
mary_elev['new_name'] = mary_elev.name.apply(lambda x: str(x)+'_Mary')
# mary_elev = pd.DataFrame(mary_elev.drop(columns = 'geometry'))


NPR_elev = gpd.read_file('NPR_merged_buff_qgis.geojson')
NPR_elev['new_name'] = NPR_elev.name.apply(lambda x: str(x)+'_NPR')
# NPR_elev = pd.DataFrame(NPR_elev.drop(columns = 'geometry'))

kobble_elev =  gpd.read_file('kobble_merged_buf_QGIS.geojson')
kobble_elev['new_name'] = kobble_elev.name.apply(lambda x: str(x)+'_kobble')
def get_geom_cols(df):
    df['Center_point'] = df['geometry'].centroid
    #Extract lat and lon from the centerpoint
    df["x"] = df.Center_point.map(lambda p: p.x)
    df["y"] = df.Center_point.map(lambda p: p.y)

    return pd.DataFrame(df.drop(columns = 'geometry'))
kobble_elev = get_geom_cols(kobble_elev)
NPR_elev = get_geom_cols(NPR_elev)
mary_elev = get_geom_cols(mary_elev)
bris_elev = get_geom_cols(bris_elev)




elev_stack = pd.concat([bris_elev, mary_elev, NPR_elev, kobble_elev])
#%

types = list(kobble_elev.type.unique())
elev_statcols = [x for x in list(elev_stack.columns) if '_'==x[0]]
elev_statcols.append('x')
elev_statcols.append('y')


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

#% unstack elevations and combine the gisdata

geom_comb.shape
df_elev_uns = unstack_pt_data(elev_stack, elev_statcols, types)

gis_dat_comb = geom_comb.merge(df_elev_uns, on ='new_name')

gis_dat_comb.columns
#%

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
    

#% veg




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

#% next we define the measurement type

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

#%
field_dat['species_clean'].unique()
field_dat[field_dat['meas_type']=='no meas type'][['meas_type', 'species']]
field_dat.columns

#%
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

# pd.DataFrame(field_dat['smallest texture clean'].unique())


# pd.DataFrame(field_dat['biggest texture clean'].unique())

# pd.DataFrame(field_dat['texture at base clean'].unique())

#%
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
    elif 'cobble' in x:
        return 'cobble'
    else:
        return x


field_dat['smallest texture clean'] = field_dat['smallest texture clean'].apply(ss_filter2)
field_dat['biggest texture clean'] = field_dat['biggest texture clean'].apply(ss_filter2)
field_dat['texture at base clean'] = field_dat['texture at base clean'].apply(ss_filter2)

#%

pd.DataFrame(field_dat['smallest texture clean'].unique())
pd.DataFrame(field_dat['biggest texture clean'].unique())
pd.DataFrame(field_dat['texture at base clean'].unique())

#% combining stuff
field_dat.new_name

def new_name_fix(x):
    if x[:2]=='00':
        return x[2:]
    elif x[0]=='0':
        return x[1:]
    else: return x

    
gis_dat_comb.new_name = gis_dat_comb.new_name.apply(new_name_fix)
elev_stack.new_name = elev_stack.new_name.apply(new_name_fix)

elev_stack.new_name
gis_dat_comb.new_name

"""
Now combine the data
"""
print(gis_dat_comb.head())
#%
df_comb = field_dat.merge(gis_dat_comb, how='inner', on = 'new_name')
# df_comb.Reach_y
# df_comb = df_comb.merge(elev_stack,how= 'inner', on='new_name')
#%%
df_comb = df_comb.merge(df_cats, how='left', left_on='Reach_y',
                         right_on = 'Reach')

df_comb.drop(columns=['Center_point'], inplace=True)

df_comb_geo = gpd.GeoDataFrame(
    df_comb, geometry=gpd.points_from_xy(x=df_comb['slx'], y=df_comb['sly']))
df_comb_geo.rs = kobble_cat.crs
print(df_comb_geo.dtypes)
df_comb_geo.to_file("df_combined_points.geojson", driver='GeoJSON')
del df_comb_geo
#%%

"""
begin analysis first run of calculations
"""

df_comb.describe()

# first  calculate root elevation

df_comb['sprout_elev'] = df_comb['sl_mean'] - df_comb['Root_depth']*.01

def rel_sprout_el(spr_el, thal_mean, lb_mean, rb_mean, lb_dist,rb_dist, biggest = False):
    """
    calculate the sprouting elevation from the nearest bankline
    """
    if biggest ==False:
        if lb_dist > rb_dist:
            return (spr_el-thal_mean)/(lb_mean-thal_mean)
        else:
            return (spr_el-thal_mean)/(rb_mean-thal_mean)
    else:
        b = np.max([lb_mean, rb_mean])
        return (spr_el-thal_mean)/(b-thal_mean)


rel_spr_el_ap = lambda x: rel_sprout_el(x['sprout_elev'], x['thal_mean'],
                                        x['LB_mean'], x['RB_mean'],
                                        x['lb_dis'], x['rb_dis'])

rel_spr_el_ap2 = lambda x: rel_sprout_el(x['sprout_elev'], x['thal_mean'],
                                        x['LB_mean'], x['RB_mean'],
                                        x['lb_dis'], x['rb_dis'],biggest=True)

rel_spr_el_ap3 = lambda x: rel_sprout_el(x['sprout_elev'], x['thal_mean'],
                                        x['LB_max'], x['RB_max'],
                                        x['lb_dis'], x['rb_dis'],biggest=True)


df_comb['rel_sprout_el'] = df_comb.apply(rel_spr_el_ap, axis = 1)
df_comb['rel_sprout_el_b'] = df_comb.apply(rel_spr_el_ap2, axis = 1)
df_comb['rel_sprout_el_b_max'] = df_comb.apply(rel_spr_el_ap3, axis = 1)



#%%
def rel_spr_el_err(spr_el, thal_mean, lb_mean, rb_mean,
                    lb_dist,rb_dist, spr_er, thal_er,rb_er, lb_er):
    
    if lb_dist > rb_dist:
        ba_mean = rb_mean
        ba_er = rb_er
    else:
        ba_mean = lb_mean
        ba_er = lb_er

        top = spr_el - thal_mean
        bot = ba_mean-thal_mean

        t_er = np.sqrt(thal_er**2+spr_er**2)
        b_er  = np.sqrt(thal_er**2+ba_er**2)
        ans = (spr_el-thal_mean)/(ba_mean-thal_mean)

        err = ans*np.sqrt((t_er/top)**2+(b_er/bot)**2)
        return err

rel_spr_el_er_ap = lambda x: rel_spr_el_err(x['sprout_elev'], x['thal_mean'],
                                        x['LB_mean'], x['RB_mean'],
                                        x['lb_dis'], x['rb_dis'],
                                        x['sl_stdev'], x['thal_stdev'],
                                        x['RB_stdev'], x['LB_stdev'])
  
df_comb['rel_sprout_el_er'] = df_comb.apply(rel_spr_el_er_ap, axis = 1)

#%%
df_comb.species.unique()

#%%
print(df_comb['rel_sprout_el'].describe(percentiles = [.05,.1,.25,.50,.75,.9]).to_latex())
# df_comb['rel_sprout_el_er'].describe()


# df_comb.columns
#%%
# df_bury['sprout_depth_reached'].unique()
df_bury = df_comb[df_comb['meas_type'] == 'dbh burial']

df_bury[df_bury['structure category'] == ' ']['Notes']

df_bury = df_bury[df_bury['structure category']!=' ']

# df_bury = df_bury[df_bury[]]

df_bury = df_bury[df_bury['structure category']!=' ']

df_bury = df_bury[df_bury['rel_sprout_el'] >=0]
df_bury.loc[df_bury['Root_depth']<=0, 'structure category'] = 'no deposition'


print('after first filter df_bury shape', df_bury.shape)
#%%
with open('buried_positions_data.tex', 'w') as file:
    file.write(df_bury['rel_sprout_el'].describe(percentiles = [.05,.1,.25,.50,.75,.9, .95]).to_latex())
    file.close()
#%%

cat_colors = sns.xkcd_palette(['greyish']) +sns.color_palette('Blues', 5)[2:]+ sns.color_palette(['orange'])

# sns.palplot(cat_colors)

cat_list = ['no deposition', 'floodplain', 'inset floodplain', 'bar', 'niche construction']

cat_dict1 = {k:v for k, v in zip(cat_list, cat_colors)}
                          
df_bury.loc[df_bury['structure category'] == 'reverse', 'structure category']= 'interbedded'
# df_bury = df_bury[]
#%% rename structre/ associative categorie

def rename_sed_cats(struc, smallest):
    if struc == 'interbedded':
        return 'inset floodplain'
    elif struc == 'fining upward':
        return 'floodplain'
    elif struc == 'massive':
        if 'loam' in smallest:
            return 'floodplain'
        elif 'gravel' in smallest:
            return 'bar'
        elif 'coarse' in smallest:
            return 'inset floodplain'
        else:
            return 'inset floodplain'
    elif struc == 'distinct change':
        return 'niche construction'
    else:
        return struc

df_bury['deposition type'] = df_bury.apply(lambda x: rename_sed_cats(x['structure category'], x['smallest texture clean']), axis=1)



#%%


sns.scatterplot(x ='Root_depth', 
                y = 'rel_sprout_el', hue = 'deposition type', 
                data = df_bury)

#%%


sns.scatterplot(x ='Root_depth', 
                y = 'rel_sprout_el_b', hue = 'deposition type', 
                data = df_bury)

#%% make the palette for  our categories


sns.set_context('paper')
fig, ax = plt.subplots()
sns.countplot(x='deposition type', data = df_bury,
              order = cat_list,palette=cat_dict1, ax= ax)
ticks, labels = plt.xticks()
# plt.xticks(ticks, labels = ['no deposition','bar', 'inset floodplain', 'floodplain', 'distinct change' ])
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
plt.xlabel('deposition type')
plt.ylabel('# of observations')

plt.tick_params(axis='x', labelrotation = 90)
fig.savefig('struct_type.svg')


#%%
fig, ax = plt.subplots()
sns.barplot(x='deposition type',y='rel_sprout_el', data = df_bury, estimator = lambda x: len(x) / len(df_bury) * 100,
              order = cat_list,palette=cat_dict1, ax= ax)
ticks, labels = plt.xticks()
# plt.xticks(ticks, labels = ['no deposition','bar', 'inset floodplain', 'floodplain', 'distinct change' ])
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
plt.xlabel('deposition type')
plt.ylabel('# of observations')

plt.tick_params(axis='x', labelrotation = 90)
fig.savefig('struct_type.svg')

#%%
# import veg_an
# help(veg_an)
# from veg_an.plot_helpers import 
sns.boxplot(x='deposition type', y = 'rel_sprout_el_b', data =  df_bury,
            order = cat_list, palette = cat_dict1)
plt.tick_params(axis='x', labelrotation = 90)
plt.ylabel('sprouting position \n $(\\frac{ht \\; above Thalweg}{Bank\\;ht})$')
plt.xlabel('sedimentary style')

#%%
# df_bury.columns
# df_bury[df_bury['structure category'] == dep_order[-1]]
dep_order = list(df_bury['structure category'].unique())
# type(dep_order[-1])
sns.boxplot(x='structure category', y = 'Root_depth', data =  df_bury,
            order = dep_order[:-1])
plt.tick_params(axis='x', labelrotation = 90)

#%%
# fig, ax = plt.subplots()
# for st in list(df_bury['structure category'].unique())[:-1]:
#     df = df_bury[df_bury['structure category']==st]
#     sns.kdeplot(df['rel_sprout_el'], shade=True, vertical=True, ax=ax, color = cat_dict1[st])

# fig.savefig('kdeforpres.svg')
# os.getcwd()
#%%

# df_bury.shape
"""
Bring in some age control
"""

from veg_an import dbh_an
from scipy import stats

df_lmods = dbh_an.df_lmods

df_lmods

def pred_age(dbh, dflm_row):
    s = dflm_row['slope']
    i = dflm_row['intercept']
    N = dflm_row['N']
    sy = dflm_row['sy']
    sx = dflm_row['sx']
    sy = dflm_row['sy']
    Xbar = dflm_row['Xbar'] 

    t = stats.t.ppf(0.95, N-2)
    #print(t, Xbar, sx, sy,N)
    pred_age = s*dbh
    pred_f  = lambda x: t*sy*np.sqrt(1+1/N+(dbh-Xbar)**2/((N-1)*sx))
    p_int = pred_f(dbh)
    return (pred_age, p_int)

#cas first
#%%
df_bury['age'] = 0
df_bury['age_pi'] = 0

caser = df_bury['species_clean'] == 'C. Cunninghamiana'
meler = df_bury['species_clean'] == 'M. Bracteata'

#%%
cas_p = lambda x: pred_age(x['DBH'], df_lmods.loc['Cas_dbh_age'])
cas_agetup= df_bury[caser].apply(cas_p, axis=1)

cas_age = [c[0] for c in cas_agetup]
cas_pi = [c[1] for c in cas_agetup]

df_bury.loc[caser, 'age'] = cas_age

df_bury.loc[caser, 'age_pi'] = cas_pi

df_bury.plot.scatter('DBH', 'age') #just to check

#%% redo it for Mel

cas_p = lambda x: pred_age(x['DBH'], df_lmods.loc['mel_dbh_age'])
cas_agetup= df_bury[meler].apply(cas_p, axis=1)

cas_age = [c[0] for c in cas_agetup]
cas_pi = [c[1] for c in cas_agetup]

df_bury.loc[meler, 'age'] = cas_age

df_bury.loc[meler, 'age_pi'] = cas_pi


df_bury.plot.scatter('DBH', 'age')


#%%

df_bury['d_rate'] = df_bury['Root_depth']/df_bury['age']

df_bury['d_rate_err'] = np.sqrt((df_bury['age_pi']/df_bury['age'])**2)*np.absolute(df_bury['d_rate'])


#%%
df_comb.columns



#%%
# line plot for completion seminar
fig, ax = plt.subplots()

ax.errorbar(df_bury['Root_depth'], df_bury['rel_sprout_el_b'], 
            yerr = df_bury['rel_sprout_el_er'],
            alpha = 0.4, fmt='none', c='grey')

# sns.lineplot(data = df_bury, x = 'Root_depth', y  = 'rel_sprout_el_b', hue = 'structure category',ax = ax)
plot_subsets_vertical(df_bury, 'Root_depth', 'rel_sprout_el_b', 'structure category', plot_list=None, ax=ax, err_meth = 'bootstrap')
plt.ylabel('establishment position \n $(\\frac{ht \\; above\\: Thalweg}{Bank\\;ht})$')
# plt.xlabel('deposition rate $(\\frac{cm}{yr})$')
ax.set_xlabel('stem burial depth (cm)')
ax.set_xlim(0,250)
ax.set_ylim(0,3)

os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
fig.savefig('root_depth_vs_ht_line.pdf', bbox_inches='tight')


df_comb['rel_sprout_el'].describe()
df_comb['rel_sprout_el_er'].describe()

fig, ax = plt.subplots()

ax.errorbar(df_bury['d_rate'], df_bury['rel_sprout_el_b'], 
            yerr = df_bury['rel_sprout_el_er'], xerr = df_bury['d_rate_err'],
            alpha = 0.4, fmt='none', c='grey')

sns.scatterplot(data = df_bury, x = 'd_rate', y  = 'rel_sprout_el_b', hue = 'structure category',ax = ax)
plt.ylabel('sprouting position \n $(\\frac{ht \\; above\\: Thalweg}{Bank\\;ht})$')
plt.xlabel('deposition rate $(\\frac{cm}{yr})$')
ax.set_xlabel(' Depos')
ax.set_xlim(0,12)
ax.set_ylim(0,3)



#%%

sns.boxplot(data = df_bury, x = 'deposition type',
            y = 'Root_depth', order = cat_list, palette = cat_dict1 )

ticks, labels = plt.xticks()
# plt.xticks(ticks, labels = ['no deposition','bar', 'inset floodplain', 'floodplain', 'distinct change' ])
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
plt.xlabel('deposition type')
plt.ylabel('sprouting depth (cm)')
plt.ylim(0,200)

plt.tick_params(axis='x', labelrotation = 90)

#%%
# df_buryd = df_bury.dropna(subset = ['d_rate'])
sns.boxplot(data = df_bury, x = 'deposition type',
            y = 'd_rate', order = cat_list, palette = cat_dict1 )

ticks, labels = plt.xticks()
# plt.xticks(ticks, labels = ['no deposition','bar', 'inset floodplain', 'floodplain', 'distinct change' ])
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
plt.xlabel('deposition type')
plt.ylabel('deposition rate $(\\frac{cm}{yr})$')
plt.ylim(0,4)

plt.tick_params(axis='x', labelrotation = 90)

#%%


sns.boxplot(data = df_bury, x = 'deposition type',
            y = 'age', order = cat_list, palette = cat_dict1 )

ticks, labels = plt.xticks()
# plt.xticks(ticks, labels = ['no deposition','bar', 'inset floodplain', 'floodplain', 'distinct change' ])
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
plt.xlabel('deposition type')
plt.ylabel('inferred tree age (yr)')
# plt.ylim(0,4)

plt.tick_params(axis='x', labelrotation = 90)

#%%
# df_bury.columns
sns.scatterplot(data = df_bury, x = 'catch_area',y = 'd_rate' ,hue='Cat')
plt.xscale('log')
# df_bury[df_bury['catch_area']==df_bury['catch_area'].min()].Reach
# df_bury['cat']

# 29734*25*25/(10**6)

#%%
# df_bury.columns
sns.scatterplot(data = df_bury, x = 'slope',y = 'd_rate' ,hue='Cat')
plt.xscale('log')
df_bury[df_bury['catch_area']==df_bury['catch_area'].min()].Reach
# df_bury['cat']

#%%
sns.scatterplot(x='age', y = 'rel_sprout_el', hue= 'Cat', data= df_bury)
#%%

df_bury['Root_depth'].hist()

# df_bury.Reach_y


df_bury.plot.scatter('age', 'Root_depth')


#%%
sns.boxplot(x='structure category', y = 'Root_depth', data =  df_bury,
            order = dep_order)
plt.tick_params(axis='x', labelrotation = 90)

# plt.xlabel()
#%% ok lets do a little bit with the sed size


os.chdir(r'C:\Users\jgarber\Jup\PhD_jup\GIS_scripts_dendro')
# sed_sizes = pd.read_excel('ss_order.xlsx')

# sed_order = list(sed_sizes['texture'].values)

# df_comb.columns

# ss_cols = ['smallest texture clean', 'biggest texture clean',
# 'texture at base clean']

# def clean_ss5(x):
#     if x[-1] == ' ':
#         return x[:-1]
#     else: return x
# for ss_row in ss_cols:
#     df_bury = df_bury[df_bury[ss_row]!='nan']
#     df_bury[ss_row] = df_bury[ss_row].apply(clean_ss5)

#     df_bury[ss_row] = df_bury[ss_row].astype('category')

#     sed_order2 = [x for x in sed_order if x in list(df_bury[ss_row].unique())]

#     df_bury[ss_row] = df_bury[ss_row].cat.reorder_categories(sed_order2, ordered = True)

# list(df_bury[ss_row].unique()), sed_order2
# ss_pal = sns.color_palette('coolwarm', len(sed_order))
# sns.palplot(ss_pal)

# pal_dict = {k:v for k, v in zip(sed_order, ss_pal)}
# print('df_bury shape after ssrow cleaning', df_bury.shape)
# #%%

# fig, ax = plt.subplots()

# ax.errorbar(df_bury['d_rate'], df_bury['rel_sprout_el'], 
#             yerr = df_bury['rel_sprout_el_er'], xerr = df_bury['d_rate_err'],
#             alpha = 0.4, fmt='none', c='grey')

# sns.scatterplot(data = df_bury, x = 'd_rate', y  = 'rel_sprout_el', 
#                 hue = 'smallest texture clean',palette = pal_dict,ax = ax)

# ax.set_xlim(0,12)
# ax.set_ylim(0,3)

# #%% for 

# # from veg_an.sed_trans import make_shields_diagram
# df_noz =df_bury[df_bury['Root_depth']>0]
# smaltex = list(df_noz['smallest texture clean'].unique())

# smt_d = {k:v for k, v in pal_dict.items() if k in smaltex}
# sns.countplot(data = df_bury[df_bury['Root_depth']>0], x = 'deposition type',
#             hue = 'smallest texture clean',palette = pal_dict)
# # fig = make_shields_diagram(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures\brownlie.svg')

# # fig

# #%%

# sns.boxplot()


#%%

fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.bar(range(1, 5), range(1, 5), color='red', edgecolor='black', hatch="/")
ax1.bar(range(1, 5), [6] * 4, bottom=range(1, 5),
        color='blue', edgecolor='black', hatch='//')
ax1.set_xticks([1.5, 2.5, 3.5, 4.5])

bars = ax2.bar(range(1, 5), range(1, 5), color='yellow', ecolor='black') + \
    ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5),
            color='green', ecolor='black')
ax2.set_xticks([1.5, 2.5, 3.5, 4.5])

patterns = ('-.-', '+', 'x', '\\', '*', 'o', '-.--', '.')
for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)



#%% 
df_bury.columns
df_bury[df_bury['structure category'] == 'massive'][['smallest texture clean', 'biggest texture clean','texture at base clean']]


#%%

#categorize by catchment area

bins = [0,50, 10**2, 2*(10**2), 5*(10**2), 10**3, 10**5]
labels = [str(bins[i-1]) +' to ' + str(bins[i]) for i in range(1,len(bins))]
# labels.append('>'+str(bins[-1]))
df_bury['ca_cut'] = pd.cut(df_bury['catch_area'],bins = bins, labels = labels )



ax = sns.countplot(x="ca_cut", hue="deposition type", data=df_bury)
df_bury = df_bury[df_bury['deposition type'] != 'unknown']
#%%
print('cat_dict1', cat_dict1, df_bury.groupby('deposition type').meas_type.count(), df_bury.shape)
sns.barplot(x='ca_cut',y='rel_sprout_el',hue='deposition type', data = df_bury, estimator = lambda x: len(x) / len(df_bury) * 100,
              order = cat_list,palette=cat_dict1)#, ax= ax)


#%%

g = sns.catplot(x="deposition type",# y="rel_sprout_el",
                hue="deposition type", col="ca_cut",
                data=df_bury, kind="count", #estimator = lambda x: len(x) / len(df_bury) * 100,
              order = cat_list,palette=cat_dict1,
                height=4, aspect=.7)


#%%

# df = sns.load_dataset("tips")
def stacked_bar(df, grouper, dval, p):
    # p = [v for k, v in palette]
    p = list(p.values())
    props = df.groupby(grouper)[dval].value_counts(normalize=True).unstack()
    props.plot(kind='bar', stacked='True', color = p)

stacked_bar(df_bury, 'ca_cut', 'deposition type', cat_dict1)

sns.boxplot(data = df_bury,x = 'ca_cut', y = 'd_rate')#, hue='deposition type', palette = cat_dict1)



#%%

"""

making plots for the chapter
"""

from veg_an.plot_helpers import xtick_rotater, create_wrap, add_subls, figure_legend

help(create_wrap)
sns.set_context('paper')
df_comb['rel_sprout_el'].describe()
df_comb['rel_sprout_el_er'].describe()

fig, (ax,ax2) = plt.subplots(ncols=2, sharey=True)

ax.errorbar(df_bury['Root_depth'], df_bury['rel_sprout_el_b'], 
            yerr = df_bury['rel_sprout_el_er'],
            alpha = 0.4, fmt='none', c='grey')

sns.scatterplot(data = df_bury, x = 'Root_depth', y  = 'rel_sprout_el_b', hue = 'deposition type',ax = ax,palette = cat_dict1)
plt.ylabel('establishment position \n $(\\frac{ht \\; above\\: Thalweg}{Bank\\;ht})$')
# plt.xlabel('deposition rate $(\\frac{cm}{yr})$')
ax.set_xlabel('stem burial depth (cm)')
ax.set_xlim(0,250)
ax.set_ylim(0,2)


sns.boxplot(data = df_bury, x = 'deposition type',
                y = 'rel_sprout_el_b_max', order = cat_list, palette = cat_dict1, ax=ax2)

ax2.tick_params(axis='x', labelrotation=90)
ax2.set_ylim(0,2)

add_subls([ax,ax2])

fig = figure_legend(fig)

os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
fig.savefig('root_depth_vs_ht.pdf', bbox_inches='tight')
#%%


#%%
def bar_and_boxplot():

    sns.set_context('paper')
    fig, [ax1, ax2,ax3,ax4, ax5] = plt.subplots(nrows=5, ncols=1, sharex=True, figsize = (7,9))
    sns.countplot(x='deposition type', data = df_bury,
                order = cat_list,palette=cat_dict1, ax= ax1)
    # ticks, labels = plt.xticks()
    # plt.xticks(ticks, labels = ['no deposition','bar', 'inset floodplain', 'floodplain', 'distinct change' ])
    # os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
    # plt.xlabel('deposition type')
    ax1.set_xlabel("")
    create_wrap(ax1, '# of observations')
    ax1.tick_params(axis='x', bottom = False, labelbottom=False)


    # plt.tick_params(axis='x', labelrotation = 90)


    
    # plt.tick_params(axis='x', labelrotation = 90)

    sns.boxplot(data = df_bury, x = 'deposition type',
                y = 'rel_sprout_el_b_max', order = cat_list, palette = cat_dict1, ax=ax2)
    ax2.set_xlabel("")
    
    create_wrap(ax2, 'normalized establishment elevation $(Z_{norm})$')
    ax2.tick_params(axis='x', bottom = False, labelbottom=False)



    sns.boxplot(data = df_bury, x = 'deposition type',
                y = 'age', order = cat_list, palette = cat_dict1, ax=ax3)
    ax3.set_xlabel("")
    
    create_wrap(ax3,'tree age (yr)')
    ax3.tick_params(axis='x', bottom = False, labelbottom=False)


    sns.boxplot(data = df_bury, x = 'deposition type',
                y = 'Root_depth', order = cat_list, palette = cat_dict1, ax=ax4)
    ax4.set_xlabel("")
    
    create_wrap(ax4,'burial depth (cm)')
    ax4.tick_params(axis='x', bottom = False, labelbottom=False)
    ax4.set_ylim(0,200)



    # ticks, labels = plt.xticks()
    # plt.xticks(ticks, labels = ['no deposition','bar', 'inset floodplain', 'floodplain', 'distinct change' ])
    # os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
    # plt.xlabel('deposition type')

    sns.boxplot(data = df_bury, x = 'deposition type',
                y = 'd_rate', order = cat_list, palette = cat_dict1, ax=ax5 )

    # ticks, labels = plt.xticks()
    # plt.xticks(ticks, labels = ['no deposition','bar', 'inset floodplain', 'floodplain', 'distinct change' ])
    # os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\completion seminar figures')
    # plt.xlabel('deposition type')
    plt.xlabel("")
    
    create_wrap(ax5,'deposition rate $(\\frac{cm}{yr})$')
    ax5.set_xlabel('deposition type')

    ax5.set_ylim(0,4)

    
    # ax5.set_xticklabels(cat_list, labelrotation = 45, horizontalalignment='right')
    xtick_rotater(ax5)

    plt.tight_layout(h_pad=0)

    return fig

fig_out = bar_and_boxplot()

add_subls(fig_out.axes, coords= (0.93,0.80))
# help(add_subls)
fig_out
# os.getcwd()
#%%
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\dendro text')

fig_out.savefig('multifig.pdf')

#%%
fig, ax = plt.subplots(figsize = (7,9))

ax.errorbar(df_bury['d_rate'], df_bury['rel_sprout_el'], 
            yerr = df_bury['rel_sprout_el_er'], xerr = df_bury['d_rate_err'],
            alpha = 0.4, fmt='none', c='grey')

sns.scatterplot(data = df_bury, x = 'd_rate', y  = 'rel_sprout_el',
                hue = 'deposition type', style = 'deposition type',
                hue_order  = cat_list, palette = cat_dict1,ax = ax)

create_wrap(ax, 'normalized establishment elevation $(Z_{norm})$')
ax.set_xlabel('deposition rate $(\\frac{cm}{yr})$')

ax.set_xlim(0,12)
ax.set_ylim(0,3)

fig.savefig('drate_pos.pdf')


#%%
print(df_bury.groupby('deposition type')['Root_depth'].count())
# df_buried = df_bury[df_bury['deposition type'] != 'no deposition']
fig, (ax, ax2) = plt.subplots(figsize = (7,4.5), ncols=2, sharey=True)

ax.errorbar(df_bury['Root_depth'], df_bury['rel_sprout_el'], 
            yerr = df_bury['rel_sprout_el_er'],
            alpha = 0.4, fmt='none', c='grey')

sns.scatterplot(data = df_bury, x = 'Root_depth', y  = 'rel_sprout_el',
                hue = 'deposition type', style = 'deposition type',
                hue_order  = cat_list, palette = cat_dict1,ax = ax)

create_wrap(ax, 'normalized establishment elevation $(Z_{norm})$')
ax.set_xlabel('burial depth $(cm)$')

ax.set_xlim(-20,150)
ax.set_ylim(0,2.5)

ax2.errorbar(df_bury['d_rate'], df_bury['rel_sprout_el'], 
            yerr = df_bury['rel_sprout_el_er'], xerr = df_bury['d_rate_err'],
            alpha = 0.4, fmt='none', c='grey')

sns.scatterplot(data = df_bury, x = 'd_rate', y  = 'rel_sprout_el',
                hue = 'deposition type', style = 'deposition type',
                hue_order  = cat_list, palette = cat_dict1,ax = ax2)


ax2.set_xlabel('deposition rate $(\\frac{cm}{yr})$')

ax2.set_xlim(-1,10)
ax2.set_ylim(0,2.5)
ax2.tick_params('y', left=False)

add_subls([ax,ax2],coords= (0.87,0.05))
fig = figure_legend(fig, loc = (.73,.69) )
plt.tight_layout(w_pad=0)
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\dendro text')
fig.savefig('rootdepth_andeprate_pos.pdf')


# get the age_dbh relationship plot ready
# help(figure_legend)

from veg_an import dbh_an

fig = dbh_an.fig2

fig.set_size_inches(7,9)
sns.set_context('paper')
# fig

for ax in fig.axes:
    ax.set_title('')
    ax.set_ylabel('minimum age (yr)')


add_subls(fig.axes, coords= (0.85,0.05))
fig.axes[1].set_ylabel('')
fig.axes[1].tick_params(axis='y', left=False, labelleft = False)

fig = figure_legend(fig, loc = (.65,.25))

plt.tight_layout(w_pad=0)
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\dendro text')
fig.savefig('age_dbh.pdf')


#%%
os.chdir(r'C:\Users\jgarber\Documents\PhD Thesis Research\dendro text')

name_change = {'slope':'m', 'intercept':'b', 'p':'p-value', 
                'R^2':'$R^2$', 'se':'s_{e}'}
t_out = df_lmods.loc[['Cas_dbh_age','mel_dbh_age' ],['slope','intercept', 'se', 'p', 'R^2', 'N']]
for cols in list(t_out.columns):
    try:
        t_out[cols] = pd.to_numeric(t_out[cols])
    except:
        continue
oout_str = t_out.rename(columns = name_change).to_latex(float_format="%.2f",escape = False)

with open('lm_tab.tex', 'wt') as filer:
    filer.write(oout_str)
    filer.close()

#%% Calculating uncertainties
df_bury.columns
df_bury['perc_age_er'] = np.absolute(df_bury['age_pi']/ df_bury['age'])

df_bury['spr_perc_er'] = np.absolute(df_bury['rel_sprout_el_er']/df_bury['rel_sprout_el_b'])

df_bury['d_rate_perc_er'] = np.absolute(df_bury['d_rate_err']/df_bury['d_rate'])

df_bury = df_bury.replace([np.inf, -np.inf], np.nan)


df_bury[['age_pi','perc_age_er','rel_sprout_el_er', 'spr_perc_er','d_rate_err', 'd_rate_perc_er']].describe()


df_bury[df_bury['deposition type']=='niche construction']

# NPR_elev.columns

# mary_elev.columns
# kobble_elev.columns
# bris_elev.columns

#%%

df_quads = dbh_an.return_quad_table()

df_quads[df_quads['use_age']>55]

#%%
df_b = df_bury[df_bury['age']>0]
err_tab = df_b[['age_pi','perc_age_er','rel_sprout_el_er', 'spr_perc_er','d_rate_err', 'd_rate_perc_er']].describe().to_latex(float_format="%.2f")
print(err_tab)

#%%
with open('er_tab.tex', 'wt') as filer:
    filer.write(err_tab)
    filer.close()

os.getcwd()


#%%
df_bury.columns

df_cats.columns
sns.scatterplot(data=df_bury, x='catch_area', 
                y = 'rel_sprout_el_b', hue = 'deposition type',
                hue_order  = cat_list, palette = cat_dict1)


df_bury[df_bury['deposition type']== 'no deposition']['Reach_x']


#%%

sns.scatterplot(data = df_bury, x = 'Root_depth', y  = 'rel_sprout_el',
                hue = 'deposition type', style = 'deposition type',
                hue_order  = cat_list, palette = cat_dict1)
