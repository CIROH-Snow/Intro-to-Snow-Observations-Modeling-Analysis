#Standardized Snow Water Equivalent Evaluation Tool (SWEET)
#created by Dr. Ryan C. Johnson as part of the Cooperative Institute for Research to Operations in Hydrology (CIROH)
# SWEET supported by the University of Utah
# 10-19-2023

#Load packages
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import sklearn
from sklearn.metrics import mean_squared_error
import hydroeval as he
from pickle import dump
import pickle 
from tqdm import tqdm
import geopandas as gpd
import folium
from folium import features
from folium.plugins import StripePattern
import branca.colormap as cm
import vincent
from vincent import AxisProperties, PropertySet, ValueRef, Axis
import hvplot.pandas
import holoviews as hv
from holoviews import dim, opts, streams
from bokeh.models import HoverTool
import matplotlib.colors as mcolors
import os
import json
import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import contextily as cx
from mpl_toolkits.axes_grid1 import make_axes_locatable

#A function to load model predictions
def load_Predictions(Region_list):
    #Regions = ['N_Sierras','S_Sierras_Low', 'S_Sierras_High']
    RegionTest = {}
    for Region in Region_list:
        RegionTest[Region] = pd.read_hdf("./Predictions/Testing/Predictions.h5", key = Region)
        
        #convert predictions/observations to SI units, aka, cm
        RegionTest[Region]['y_test'] = RegionTest[Region]['y_test']*2.54
        RegionTest[Region]['y_pred'] = RegionTest[Region]['y_pred']*2.54
        RegionTest[Region]['y_pred'] = RegionTest[Region]['y_pred']*2.54
        
        #get SWE obs columns to convert to si
        obscols = [match for match in RegionTest[Region] if 'SWE_' in match]
        for col in obscols:
            RegionTest[Region][col] = RegionTest[Region][col]*2.54
        
        
    return RegionTest


#Function to convert predictions into parity plot plus evaluation metrics
def parityplot(EvalDF, savfig, region, watershed, date):   

    Title = f"SWEMLv2.0 Model Performance {date} \n {watershed} River Basin, {region}"
    figname = f"./Figures/{region}_{watershed}_parity_{date}.png"
    
    #Plot the results in a parity plot
    sns.set(style='ticks')
    SWEmax = max(EvalDF['y_test'])

    sns.relplot(data=EvalDF, x='y_test', y='y_pred', hue = 'Elevation_m', aspect=1.61)
    plt.plot([0,SWEmax], [0,SWEmax], color = 'red', linestyle = '--')
    plt.xlabel('Observed SWE (cm)')
    plt.ylabel('Predicted SWE (cm)')
    plt.title(Title)

    if savfig==True:
        plt.savefig(figname, dpi =600, bbox_inches='tight')

    plt.show()

    #Run model evaluate functions
    #Regional
    Performance = pd.DataFrame()
    y_test = EvalDF['y_test']
    y_pred = EvalDF['y_pred']

    
    r2 = sklearn.metrics.r2_score(y_test, y_pred)
    rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False)
    kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test)
    pbias = he.evaluator(he.pbias, y_pred, y_test)

    r2_fSCA = sklearn.metrics.r2_score(y_test, y_pred)
    rmse_fSCA = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False)
    kge_fSCA, r_fSCA, alpha_fSCA, beta_fSCA = he.evaluator(he.kge, y_pred, y_test)
    pbias_fSCA = he.evaluator(he.pbias, y_pred, y_test)
    
    error_data = np.array([ round(r2,2),  
                            round(rmse,2), 
                            round(kge[0],2),
                            round(pbias[0],2),
                            round(r2_fSCA,2),
                            round(rmse_fSCA,2),
                            round(kge_fSCA[0],2),
                            round(pbias_fSCA[0],2)])
    
    error = pd.DataFrame(data = error_data.reshape(-1, len(error_data)), 
                            columns = ['R2',
                                    'RMSE',
                                    'KGE', 
                                    'PBias', 
                                    'R2_fSCA',
                                    'RMSE_fSCA',
                                    'KGE_fSCA', 
                                    'PBias_fSCA',
                                    ])    
    return error
    
    
    
#Plot the error/prediction compared to different variables
def Model_Vs(EvalDF,metric,model_output,savfig, region, watershed, date):   

    figname = f"./Figures/{region}_{watershed}_{metric}_{model_output}_{date}.png"
        
    #Calculate error
    EvalDF['error'] = EvalDF['y_test'] - EvalDF['y_pred']
    EvalDF['Perc_error'] = ((EvalDF['y_test'] - EvalDF['y_pred'])/EvalDF['y_test'])*100
    EvalDF['Perc_error'] = EvalDF['Perc_error'].fillna(0)
    #change error > 100 to 100
    EvalDF.loc[EvalDF['Perc_error'] >100, 'Perc_error'] = 100
    EvalDF.loc[EvalDF['Perc_error'] < -100, 'Perc_error'] = -100
    
    if model_output == 'Prediction':
        Y = 'y_pred'
        ylabel ='SWE (cm)'
    
    if model_output == 'Error':
        Y = 'error'
        ylabel ='SWE (cm)'
        
    if model_output == 'Percent_Error':
        Y = 'Perc_error'
        ylabel = 'SWE Error'
        
    if metric == 'northness':
        xlabel = 'Northness'
        
    if metric == 'Elevation_m':
        xlabel = 'Elevation (m)'
    if metric == 'WYWeek':
        xlabel = 'Water Year Week (From Oct 1st)'
    if metric == 'prev_SWE':
        xlabel = 'Previous SWE Estimate'
    if metric == 'Lat':
        xlabel = 'Latitude'
    if metric == 'Aspect_Deg':
        xlabel = 'Aspect Degree'
    if metric == 'Slope_Deg':
        xlabel = 'Slope (%)'
    if metric == 'season_precip_cm':
        xlabel = 'Precipitation (cm)'
    if metric == 'sturm_value':
        xlabel = 'Sturm Snow Classification'

    Title = f"{model_output} by {xlabel} {date} \n {watershed} River Basin, {region}"
    
    sns.set(style='ticks')
    sns.relplot(data=EvalDF, x=metric, y=Y, aspect=1.61)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(Title, fontsize = 20)
    if savfig==True:
            plt.savefig(figname, dpi =600, bbox_inches='tight')

    plt.show()



def SpatialAnalysis(files, basinname, output_res, markersize, cmap, var,variant, swethres, plttitle, pltfig=True, savfig=True):
    
    for file in files:
        df = pd.read_parquet(f"files/ASO/{basinname}/{output_res}M_SWE_parquet/{file}")

        #convert m to in to be consistent with SNOTEL
        columns = df.columns
        if var == 'swe_m':
            df['swe_in'] = df['swe_m'] * 39.3701
            #select obs > 0.5"
            df = df[df['swe_m'] > swethres]
        if var == 'swe_in':
            df['swe_in'] = df['swe_m'] * 39.3701
            #select obs > 0.5"
            df = df[df['swe_in'] > swethres]
        if var == 'median_SWE_m':
            #select obs > 0.5"
            df = df[df['median_SWE_m'] > swethres]

        if var == 'median_SWE_in':
            #select obs > 0.5"
            df = df[df['median_SWE_in'] > swethres]

        #Make a date
        Y = file[-16:-12]
        M = file[-12:-10]
        D = file[-10:-8]

        figpath = f"Figures/ASO/{basinname}/{output_res}M"
        if not os.path.exists(figpath):
            os.makedirs(figpath, exist_ok=True)
        figname = f"{figpath}/{Y}-{M}-{D}.png"

        #Convert to a geopandas DF
        Pred_Geo = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.cen_lon, df.cen_lat), crs=4326)

        Pred_Geo = Pred_Geo.to_crs(epsg=3857)


        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
        if var == 'swe_in':
            Pred_Geo.plot(column=var,
                    ax = ax,
                    legend=True,
                    markersize=markersize,
                    marker = 's',
                    legend_kwds={"label": 'Snow Water Equivalent (in)', "orientation": "vertical"},
                    cmap = cmap,
                    )
        if var == 'swe_m':
            Pred_Geo.plot(column=var,
                    ax = ax,
                    legend=True,
                    markersize=markersize,
                    marker = 's',
                    legend_kwds={"label": 'Snow Water Equivalent (m)', "orientation": "vertical"},
                    cmap = cmap,
                    )
            
        if var == 'median_SWE_in':
            Pred_Geo.plot(column=var,
                    ax = ax,
                    legend=True,
                    markersize=markersize,
                    marker = 's',
                    legend_kwds={"label": 'Median Snow Water Equivalent (in)', "orientation": "vertical"},
                    cmap = cmap,
                    )
            
        if var == 'median_SWE_m':
            Pred_Geo.plot(column=var,
                    ax = ax,
                    legend=True,
                    markersize=markersize,
                    marker = 's',
                    legend_kwds={"label": 'Median Snow Water Equivalent (m)', "orientation": "vertical"},
                    cmap = cmap,
                    )

        if var == 'SWE_diff_m':
            # Create a colormap centered at 0

            vmin = Pred_Geo['SWE_diff_m'].min()
            vcen = 0
            vmax = Pred_Geo['SWE_diff_m'].max()

            if vmin>=vcen:
                vmin = -abs(vmax)
            if vmax<= vcen:
                vmax = abs(vmin)

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcen, vmax=vmax)

            Pred_Geo.plot(column=var,
                    ax = ax,
                    legend=True,
                    markersize=markersize,
                    marker = 's',
                    legend_kwds={"label": 'Snow Water Equivalent Difference (m)', "orientation": "vertical"},
                    cmap = cmap,
                    norm = norm
                    )
        if var == 'SWE_diff_in':
            vmin = Pred_Geo['SWE_diff_in'].min()
            vcen = 0
            vmax = Pred_Geo['SWE_diff_in'].max()

            if vmin>=vcen:
                vmin = -abs(vmax)
            if vmax<= vcen:
                vmax = abs(vmin)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcen, vmax=vmax)
            # Create a colormap centered at 0
            #cmap = plt.get_cmap('coolwarm')
            Pred_Geo.plot(column=var,
                    ax = ax,
                    legend=True,
                    markersize=markersize,
                    marker = 's',
                    legend_kwds={"label": 'Snow Water Equivalent Difference (in)', "orientation": "vertical"},
                    cmap = cmap,
                    norm = norm
                    )


        ax.set_xlim(-1.335e7, -1.325e7)
        ax.set_ylim(4.54e6, 4.61e6)
        cx.add_basemap(ax, source="https://server.arcgisonline.com/ArcGIS/rest/services/"+variant+"/MapServer/tile/{z}/{y}/{x}")   #cx.providers.OpenStreetMap.Mapnik)
        ax.set_axis_off()
        # ax.text(-1.345e7, 4.64e6, f"SWE estimate: {date}", fontsize =14)

        plt.title(f"SWE for {basinname} on {Y}-{M}-{D}")

        if var == 'median_SWE_m' or 'median_SWE_in':
            plt.title(plttitle)

        if var == 'SWE_diff_in' or 'SWE_diff_m':
            plt.title(plttitle)

        if savfig == True:
            plt.savefig(figname, dpi =600, bbox_inches='tight')
        F =  plt

        if pltfig==True:
            plt.show()
        
        return df
    
    
    
#create geopandas dataframes to map predictions and obs
def createGeoSpatial(EvalDF):
    

    #Convert the Prediction Geospatial dataframe into a geopandas dataframe
    Pred_Geo = gpd.GeoDataFrame(EvalDF, geometry = gpd.points_from_xy(Pred_Geo.Long, Pred_Geo.Lat))

    Pcols = ['Long','Lat','elevation_m','slope_deg','aspect', 'geometry']
    Obscols = ['Long','Lat','elevation_m','slope_deg','aspect', 'geometry']
    Pred_Geo= Pred_Geo[Pcols].reset_index()
    Snotel_Geo = Snotel_Geo[Obscols].reset_index()
    
    #add to respective dataframe
    GeoPred = pd.concat([GeoPred, Pred_Geo])
    GeoObs = pd.concat([GeoObs, Snotel_Geo])
    
    #Select sites used for prediction
    GeoObs = GeoObs.set_index('station_id').T[Sites].T.reset_index()
    
    return GeoPred, GeoObs






#need to put the predictions, obs, error in a time series format
def ts_pred_obs_err(EvalDF):
    print('Processing Dataframe into timeseries format: predictions, observations, error.')
    x = EvalDF.copy()
    x.index = x.index.set_names(['cell_id'])

    #predictions
    x_pred = x.copy()
    cols = ['Date','y_pred']
    x_pred = x_pred[cols].reset_index().set_index('Date').sort_index()
    x_pred = df_transpose(x_pred, 'y_pred')

    #observations
    x_obs = x.copy()
    cols = ['Date','y_test']
    x_obs = x_obs[cols].reset_index().set_index('Date').sort_index()
    x_obs = df_transpose(x_obs, 'y_test')

    #error
    x_err = x.copy()
    cols = ['Date','error']
    x_err = x_err[cols].reset_index().set_index('Date').sort_index()
    x_err = df_transpose(x_err, 'error')
    
    return x_pred, x_obs, x_err


def map_data_prep(RegionTest):
    #Get regions
    Regions = list(RegionTest.keys())
    
    #put y_pred, y_pred, y_test, Region into one DF for parity plot
    EvalDF = pd.DataFrame()
    cols = ['Date', 'Lat', 'Long', 'elevation_m', 'y_test', 'y_pred', 'y_pred', 'Region']
    for Region in Regions:
        df = RegionTest[Region][cols]
        EvalDF = pd.concat([EvalDF, df])
        
    #Calculate error
    EvalDF['error'] = EvalDF['y_test'] - EvalDF['y_pred']
    EvalDF['error_fSCA'] = EvalDF['y_test'] - EvalDF['y_pred']
    
    return EvalDF

def df_transpose(df, obs):
    #get index
    date_idx = df.index.unique()
    #get columns names, aka sites
    sites = df['cell_id'].values
    #make dataframe
    DF =pd.DataFrame(index = date_idx)
    for site in tqdm(sites):
        s = pd.DataFrame(df[df['cell_id'] == site][obs])
        DF = DF.join(s)
        DF = DF.rename(columns ={obs: site})
    DF = DF.loc[:,~DF.columns.duplicated()].copy()
    return DF


#Map locations and scoring of sites
#def Map_Plot_Eval(self, freq, df, size):
def Map_Plot_Eval(RegionTest, yaxis, error_metric, Region_list):   
    
    #Make sure dates are in datetime formate
    for key in RegionTest.keys():
        RegionTest[key]['Date'] = pd.to_datetime(RegionTest[key]['Date'])
    
    #correctly configure dataframes for plotting
    pred, obs, err = ts_pred_obs_err(map_data_prep(RegionTest))
    
    #Get SNOTEL sites used as features
    #load RFE optimized features
    Sites = InSitu_Locations(RegionTest)

    #Get the geometric DF for prediction locations and in situ obs
    GeoDF, Snotel = createGeoSpatial(Sites, Region_list)

    print('Plotting monitoring station locations')
    cols =  ['cell_id', 'Lat', 'Long', 'geometry']

    df_map = GeoDF[cols].copy()

    #Get Centroid of watershed
    centeroid = df_map.dissolve().centroid

    # Create a Map instance
   # m = folium.Map(location=[centeroid.y[0], centeroid.x[0]], tiles = 'Stamen Terrain', zoom_start=8, 
    #               control_scale=True)
    m = folium.Map(location=[centeroid.y[0], centeroid.x[0]], zoom_start=8, control_scale=True)
    #add legend to map
    if error_metric == 'KGE':
        colormap = cm.StepColormap(colors = ['darkred', 'r', 'orange', 'g'], vmin = 0, vmax = 1, index = [0,0.4,0.6,0.85,1])
        colormap.caption = 'Model Performance (KGE)'
        
    elif error_metric == 'cm':
        colormap = cm.StepColormap(colors = ['g', 'orange', 'r', 'darkred'], vmin = 0, vmax = 20, index = [0,6,12,25,50])
        colormap.caption = 'Model Error (cm)'
        
    elif error_metric == '%':
        colormap = cm.StepColormap(colors = ['g', 'orange', 'r', 'darkred'], vmin = 0, vmax = 50, index = [0,10,20,30,50])
        colormap.caption = 'Model Error (%)'
        
    
    m.add_child(colormap)

    ax = AxisProperties(
    labels=PropertySet(
        angle=ValueRef(value=300),
        align=ValueRef(value='right')
            )
        )

    for i in obs.columns:


        #get site information
        site = i
        Obs_site = 'Observations'#_' + site
        Pred_site = 'Predictions'#_' + site
        Err_site = 'Errors'#_' + site


        #get modeled, observed, and error information for each site
        df = pd.DataFrame(obs[site])
        df = df.rename(columns = {site: Obs_site})
        df[Pred_site] = pd.DataFrame(pred[site])
        df[Err_site] = pd.DataFrame(err[site])
        
        #drop na values
        df.dropna(inplace = True)

        if error_metric == 'KGE':
            #set the color of marker by model performance
            kge, r, alpha, beta = he.evaluator(he.kge, df[Pred_site].astype('float32'), df[Obs_site].astype('float32'))

            if kge[0] > 0.85:
                color = 'green'

            elif kge[0] > 0.6:
                color = 'orange'

            elif kge[0] > 0.40:
                color = 'red'

            else:
                color = 'darkred'
                
        #error in absolute value and inches       
        elif error_metric == 'cm':
            error = np.abs(np.mean(df[Obs_site] - df[Pred_site]))
            if error < 6:
                color = 'green'

            elif error < 12:
                color = 'orange'

            elif error <25:
                color = 'red'

            else:
                color = 'darkred'
        
        #mean percentage error
        elif error_metric == '%':
            #make all predictions and observations below 1", 1" to remove prediction biases, it does not matter if there
            #is 0.5" or 0.9" of SWE but the percentage error here will be huge and overpowering
            df[df[Obs_site]<1] = 1
            df[df[Pred_site]<1] = 1
            
            error = np.mean(np.abs(df[Obs_site] - df[Pred_site])/df[Obs_site])*100
            if error < 10:
                color = 'green'

            elif error < 20:
                color = 'orange'

            elif error <30:
                color = 'red'

            else:
                color = 'darkred'
                

        title_size = 14
        
        #display(df)

        #create graph and convert to json
        graph = vincent.Scatter(df, height=300, width=500)
        graph.axis_titles(x='Datetime', y=yaxis)
        graph.legend(title= 'Legend')
        graph.colors(brew='Paired')
        graph.x_axis_properties(title_size=title_size, title_offset=35,
                      label_angle=300, label_align='right', color=None)
        graph.y_axis_properties(title_size=title_size, title_offset=-30,
                      label_angle=None, label_align='right', color=None)

        data = json.loads(graph.to_json())

        #Add marker with point to map, https://fontawesome.com/v4/cheatsheet  - needs to be v4.6 or less
        lat_long = df_map[df_map['cell_id'] == i]
        lat = lat_long['Lat'].values[0]
        long = lat_long['Long'].values[0]

        mk = features.Marker([lat, long], icon=folium.Icon(color=color, icon = ' fa-ge', prefix = 'fa'))
        p = folium.Popup()
        v = features.Vega(data, width="100%", height="100%")

        mk.add_child(p)
        p.add_child(v)
        m.add_child(mk)
        
        
    # add SNOTEL marker one by one on the map
    for i in range(0,len(Snotel)):
        

        folium.Marker(
          location=[Snotel.iloc[i]['Lat'], Snotel.iloc[i]['Long']],
            icon=folium.Icon(color='blue', icon = 'fa-area-chart', prefix = 'fa'),
            tooltip =  str(Snotel.iloc[i]['station_id']),
          popup= str(Snotel.iloc[i]['elevation_m']) + "m",
       ).add_to(m)

    display(m)
    



 #make plot of water at different elevation bands and aspects
def barplot(EvalDF, incols, outcols, output_res, ncol, Title, save, figname):
    #col = [cols[0], cols[1], 'Elevation_m']

    df = EvalDF.copy()
    if incols[0] == 'median_SWE_m':
        Elevation_range = df['Elevation_m'].max()-df['Elevation_m'].min()
        Etier = Elevation_range/3
        low = int(round(df['Elevation_m'].min()+Etier,0))
        mid =  int(round(df['Elevation_m'].min()+(2*Etier),0))

        lowdf = df[df['Elevation_m']<low]
        middf = df[(df['Elevation_m']>low) & (df['Elevation_m']<mid)]
        highdf = df[df['Elevation_m']>mid]
        

    if incols[0] == 'median_SWE_in':
        df['Elevation_ft'] = df['Elevation_m']*3.28084
        Elevation_range = df['Elevation_ft'].max()-df['Elevation_ft'].min()
        Etier = Elevation_range/3
        low = int(round(df['Elevation_ft'].min()+Etier,0))
        mid =  int(round(df['Elevation_ft'].min()+(2*Etier),0))

        lowdf = df[df['Elevation_ft']<low]
        middf = df[(df['Elevation_ft']>low) & (df['Elevation_ft']<mid)]
        highdf = df[df['Elevation_ft']>mid]
        output_res = output_res*3.28084

    lowobs = len(lowdf)
    midobs = len(middf)
    highobs = len(highdf)

    lowdf = pd.DataFrame(lowdf.mean()).T
    middf = pd.DataFrame(middf.mean()).T
    highdf = pd.DataFrame(highdf.mean()).T

    #get total watervolumes
    lowdf['Median_Volume'] = lowdf[incols[0]]*lowobs*output_res*output_res
    middf['Median_Volume'] = middf[incols[0]]*midobs*output_res*output_res
    highdf['Median_Volume'] = highdf[incols[0]]*highobs*output_res*output_res

    lowdf['Observed_Volume'] = lowdf[incols[1]]*lowobs*output_res*output_res
    middf['Observed_Volume'] = middf[incols[1]]*midobs*output_res*output_res
    highdf['Observed_Volume'] = highdf[incols[1]]*highobs*output_res*output_res

    tierdf = pd.concat([lowdf, middf, highdf])
    
    tierdf['Volume_Difference'] = tierdf['Median_Volume']-tierdf['Observed_Volume']

    #tierdf =round(tierdf, 0)
    if incols[0] == 'median_SWE_m':
        tierdf['Elevation_Range'] = [f"Below {low}m", f"{low}m - {mid}m", f"Above {mid}m"] #set elevation ranges in meters
    if incols[0] == 'median_SWE_in':
        tierdf['Elevation_Range'] = [f"Below {low}ft", f"{low}ft - {mid}ft", f"Above {mid}ft"] #set elevation ranges in feet

    if incols[0] == 'median_SWE_in' and outcols[1] == 'Median_Volume':
        tierdf['Median_Volume'] = (tierdf['Median_Volume']/12)*2.29569e-5 #convert to acre-ft
        tierdf['Observed_Volume'] = (tierdf['Observed_Volume']/12)*2.29569e-5 #convert to acre-ft
        tierdf['Volume_Difference'] = (tierdf['Volume_Difference']/12)*2.29569e-5 #convert to acre-ft
        ylab = f'Frozen Water Volume (AcFt)'

    if incols[0] == 'median_SWE_m' and outcols[1] == 'Median_Volume':
        scaler = 10000    
        ylab = f'Frozen Water Volume (x{scaler} $m^3$)'

    if outcols[0] == 'median_SWE_in':
        ylab = f'Average SWE (in)'
    if outcols[0] == 'median_SWE_m':
        ylab = f'Average SWE (m)'
 

    tierdf.set_index('Elevation_Range', inplace=True)

    df = tierdf[outcols]

    if incols[0] == 'median_SWE_m' and outcols[1] == 'Median_Volume':
        df = df/scaler

    #f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax = df.plot.bar(rot=0)
    ax.set_ylabel(ylab)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.265), ncol =ncol,fancybox=True)
    plt.title(Title)

    if save ==True:
        plt.savefig(figname, dpi=600, bbox_inches="tight")

    return df