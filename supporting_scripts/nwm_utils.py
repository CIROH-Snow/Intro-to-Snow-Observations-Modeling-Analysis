import os
import sys
import pytz
import time
import urllib3
import datetime
import numpy as np
import pandas as pd
import pyproj
import folium
import hvplot.pandas
import holoviews as hv
import hvplot.xarray
from holoviews import opts
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xyzservices.providers as xyz
from scipy.stats import pearsonr, spearmanr
pd.options.mode.chained_assignment = None

import geoviews as gv
import geoviews.tile_sources as gts
gv.extension('bokeh')

def getSNOTELData(SiteName, SiteID, StateAbb, StartDate, EndDate, OutputFolder):
	url1 = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/'
	url2 = f'{SiteID}:{StateAbb}:SNTL%7Cid=%22%22%7Cname/'
	url3 = f'{StartDate},{EndDate}/'
	url4 = 'WTEQ::value?fitToScreen=false'
	url = url1+url2+url3+url4
    
	dl_start_time = time.time()
	
	http = urllib3.PoolManager()
	response = http.request('GET', url)
	data = response.data.decode('utf-8')
	i=0
	for line in data.split("\n"):
		if line.startswith("#"):
			i=i+1
	data = data.split("\n")[i:]
	
	df = pd.DataFrame.from_dict(data)
	df = df[0].str.split(',', expand=True)
	df.rename(columns={0:df[0][0], 
					   1:df[1][0]}, inplace=True)
	df.drop(0, inplace=True)
	df.dropna(inplace=True)
	df.reset_index(inplace=True, drop=True)
	df["Date"] = pd.to_datetime(df["Date"])
	df.rename(columns={df.columns[1]:'Snow Water Equivalent (m) Start of Day Values'}, inplace=True)
	df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: pd.to_numeric(x) * 0.0254)  # convert in to m
	df['Water_Year'] = pd.to_datetime(df['Date']).map(lambda x: x.year+1 if x.month>9 else x.year)
	
	df.to_csv(f'./{OutputFolder}/df_{SiteID}_{StateAbb}_SNTL.csv', index=False)

	dl_elapsed = time.time() - dl_start_time
	print(f'✅ Retrieved data for {SiteName}, {SiteID} in {dl_elapsed:.2f} seconds\n')

def getCCSSData(SiteName, SiteID, StartDate, EndDate, OutputFolder):
    StateAbb = 'Ca'
    url1 = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/start_of_period/' 
    url2 = f'{SiteID}:CA:MSNT%257Cid=%2522%2522%257Cname/'
    url3 = f'{StartDate},{EndDate}/'
    url4 = 'WTEQ::value?fitToScreen=false'
    url = url1+url2+url3+url4

    dl_start_time = time.time()

    http = urllib3.PoolManager()
    response = http.request('GET', url)
    data = response.data.decode('utf-8')
    i=0
    for line in data.split("\n"):
        if line.startswith("#"):
            i=i+1
    data = data.split("\n")[i:]

    df = pd.DataFrame.from_dict(data)
    print(df.columns)
    df = df[0].str.split(',', expand=True)
    df.rename(columns={0:df[0][0], 
                        1:df[1][0]}, inplace=True)
    df.drop(0, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.rename(columns={df.columns[1]:'Snow Water Equivalent (m) Start of Day Values'}, inplace=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: pd.to_numeric(x) * 0.0254)  # convert in to m
    df['Water_Year'] = pd.to_datetime(df['Date']).map(lambda x: x.year+1 if x.month>9 else x.year)

    df.to_csv(f'./{OutputFolder}/df_{SiteID}_{StateAbb}_CCSS.csv', index=False)

    dl_elapsed = time.time() - dl_start_time
    print(f'✅ Retrieved data for {SiteName}, {SiteID} in {dl_elapsed:.2f} seconds\n')

def convert_latlon_to_yx(lat, lon, input_crs, ds, output_crs):
    """
    This function takes latitude and longitude values along with
    input and output coordinate reference system (CRS) and 
    uses Python's pyproj package to convert the provided values 
    (as single float values, not arrays) to the corresponding y and x 
    coordinates in the output CRS.
    
    Parameters:
    lat: The latitude value
    lon: The longitude value
    input_crs: The input coordinate reference system ('EPSG:4326')
    output_crs: The output coordinate reference system
    
    Returns:
    y, x: a tuple of the transformed coordinates in the specified output
    """
    # Create a transformer
    transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)

    # Perform the transformation
    x, y = transformer.transform(lon, lat)

    return y, x 

def convert_utc_to_local(state, df):
    state_abbreviations = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
    }

    state_timezones = {
    'AL': 'US/Central', 'AK': 'US/Alaska', 'AZ': 'US/Mountain', 'AR': 'US/Central',
    'CA': 'US/Pacific', 'CO': 'US/Mountain', 'CT': 'US/Eastern', 'DE': 'US/Eastern',
    'FL': 'US/Eastern', 'GA': 'US/Eastern', 'HI': 'US/Hawaii', 'ID': 'US/Mountain',
    'IL': 'US/Central', 'IN': 'US/Eastern', 'IA': 'US/Central', 'KS': 'US/Central',
    'KY': 'US/Eastern', 'LA': 'US/Central', 'ME': 'US/Eastern', 'MD': 'US/Eastern',
    'MA': 'US/Eastern', 'MI': 'US/Eastern', 'MN': 'US/Central', 'MS': 'US/Central',
    'MO': 'US/Central', 'MT': 'US/Mountain', 'NE': 'US/Central', 'NV': 'US/Pacific',
    'NH': 'US/Eastern', 'NJ': 'US/Eastern', 'NM': 'US/Mountain', 'NY': 'US/Eastern',
    'NC': 'US/Eastern', 'ND': 'US/Central', 'OH': 'US/Eastern', 'OK': 'US/Central',
    'OR': 'US/Pacific', 'PA': 'US/Eastern', 'RI': 'US/Eastern', 'SC': 'US/Eastern',
    'SD': 'US/Central', 'TN': 'US/Central', 'TX': 'US/Central', 'UT': 'US/Mountain',
    'VT': 'US/Eastern', 'VA': 'US/Eastern', 'WA': 'US/Pacific', 'WV': 'US/Eastern',
    'WI': 'US/Central', 'WY': 'US/Mountain'
    }    

    if len(state) == 2:
        state_abbr = state
    else:
        state_abbr = state_abbreviations.get(state, "State not found")

    # Extract the state abbreviation from the filename
    # state_abbr = os.path.basename(filename).split('_')[2]  
    timezone = state_timezones.get(state_abbr)

    if timezone:
        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Convert to local time zone
        local_tz = pytz.timezone(timezone)
        df['Date_Local'] = df['Date'].dt.tz_convert(local_tz)

         # Save the timezone-aware Date_Local column
        df['Date_Local'] = df['Date_Local'].astype(str)
        df['Date_Local'] = df['Date_Local'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))
        df['Date_Local'] = df['Date_Local'].apply(lambda x: x.replace(tzinfo=None))

    else:
        print(f"Timezone for state abbreviation {state_abbr} not found.")
        
    return df

def combine(obs_files, mod_files, StartDate, EndDate):

    # Create a dictionary to store dataframes
    dataframes = {}
    
    # Read SNOTEL files
    for file in obs_files:
        location = os.path.basename(file).split('_')[1]  # Extract location from filename
        network = os.path.basename(file).split('_')[-1].split('.')[0] # Extract network from filename
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date']).dt.date  # .dt.date is required because times are not excatly the same between SNOTEL and NWM
        dataframes[f'{network}_{location}'] = df.set_index('Date')
    
    # Read NWM files
    for file in mod_files:
        location = os.path.basename(file).split('_')[1]  # Extract location from filename
        df = pd.read_csv(file)
        df['Date_Local'] = pd.to_datetime(df['Date_Local']).dt.date  # .dt.date is required because times are not excatly the same between SNOTEL and NWM
        dataframes[f'NWM_{location}'] = df.set_index('Date_Local')
    
    # Merge dataframes on Date
    combined_df = pd.DataFrame(index=pd.date_range(start=StartDate, end=EndDate))  
    for key, df in dataframes.items():
        if 'SNTL' in key:
            combined_df[f'{key}_swe_m'] = df['Snow Water Equivalent (m) Start of Day Values']
        if 'CCSS' in key:
            combined_df[f'{key}_swe_m'] = df['Snow Water Equivalent (m) Start of Day Values']
        elif 'NWM' in key:
            combined_df[f'{key}_swe_m'] = df['NWM_SWE_meters']

    return combined_df

def report_max_dates_and_values(df, col_obs, col_mod):
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime")

    # Find max values and associated dates
    max_obs = df[col_obs].max()
    date_obs = df[col_obs].idxmax()

    max_mod = df[col_mod].max()
    date_mod = df[col_mod].idxmax()

    # Create a summary table as a DataFrame (nice for Jupyter display)
    summary_table = pd.DataFrame({
        'Data Source': [col_obs, col_mod],
        'Peak SWE (m)': [max_obs, max_mod],
        'Date of Maximum': [date_obs.date(), date_mod.date()]
    })

    return summary_table

def compute_melt_period(swe_series, min_zero_days=10):
    
    # Find peak date and maximum SWE
    peak_date = swe_series.idxmax()
    peak_swe = swe_series.max()

    # Subset data to only include days after peak date
    after_peak = swe_series.loc[peak_date:]

    # Find first date where SWE becomes zero and stays zero for at least `min_zero_days`
    zero_streak = 0
    melt_end_date = None

    for date, value in after_peak.items():
        if value == 0:
            zero_streak += 1
        else:
            zero_streak = 0

        if zero_streak >= min_zero_days:
            melt_end_date = date
            break

    if melt_end_date is None:
        raise ValueError("Could not find a period of at least 10 consecutive zero SWE days after the peak.")

    # Calculate melt period length (days between peak and melt completion)
    melt_period_days = (melt_end_date - peak_date).days

    # Calculate melt rate (m/day)
    melt_rate = peak_swe / melt_period_days

    # Return results in a dictionary
    return {
        'peak_date': peak_date,
        'peak_swe_m': peak_swe,
        'melt_end_date': melt_end_date,
        'melt_period_days': melt_period_days,
        'melt_rate_m/d': melt_rate
    }

def prep_nwm_swe_dataframe(ds, state):
    df = ds.to_dataframe()
    df.drop(columns=['crs'], inplace=True)
    df.reset_index(inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    df.rename(columns={df.columns[0]:'Date', df.columns[1]:'NWM_SWE_meters'}, inplace=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: pd.to_numeric(x)/1000)  # convert mm to m   
    df_local = convert_utc_to_local(state, df)   
    df_local.index = pd.to_datetime(df_local['Date_Local'])
    df_local = df_local.groupby(pd.Grouper(freq='D')).first()
    df_local = df_local.reset_index(drop=True)  
    #df_local.to_csv(output_path, index=False)

    return df_local

def compute_spatial_agg_from_obs(folder_path, agg):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        raise ValueError("No CSV files found in the specified folder.")

    # Read all files into a list of DataFrames
    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, parse_dates=['Date'])
        dfs.append(df)

    # Concatenate all files into a single DataFrame
    combined_df = pd.concat(dfs)

    # Group by Date and Water_Year, compute mean SWE
    averaged_df = combined_df.groupby(['Date', 'Water_Year'], as_index=False).agg({
        'Snow Water Equivalent (m) Start of Day Values': agg
    })

    # Save to output CSV
    # averaged_df.to_csv(output_file, index=False)
    return averaged_df

    print(f"Averaged CSV saved to: {output_file}")

def plot_sites_within_domain(gdf_sites, domain_gdf, zoom_start=10):
    """
    Create and return a folium map showing observation sites within a given watershed boundary.

    Parameters:
    - gdf_sites: GeoDataFrame containing site locations.
    - domain_gdf: GeoDataFrame containing the watershed boundary.
    - zoom_start: Initial zoom level for the map (default=10).

    Returns:
    - folium.Map object ready to display.
    """

    # Calculate center of the domain's bounding box
    minx, miny, maxx, maxy = domain_gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    # Convert to GeoJSON (ensuring date fields are strings if necessary)
    geojson_sites = gdf_sites.astype(dict(beginDate=str, endDate=str)).to_json()
    geojson_domain = domain_gdf.to_json()

    # Create folium map centered on the domain
    m = folium.Map([center_lat, center_lon], zoom_start=zoom_start)

    # Add site markers
    for _, row in gdf_sites.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Site Name: {row.get('Site Name', row['name'])}<br>Site Code: {row.get('Site Code', row.code)}",
            icon=folium.Icon(color="green"),
            tooltip=row.get('Site Name', row['name'])
        ).add_to(m)

    # Add watershed boundary as GeoJSON overlay
    folium.GeoJson(geojson_domain, name='Watershed Boundary', style_function=lambda x: {"color": "lightcyan", "fillOpacity": 0.3}).add_to(m)

    # Add Esri Imagery layer
    esri_tiles = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/Tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Esri Imagery"
    )
    esri_tiles.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m

def compute_stats(df, ts1, ts2):
    df = df[[f'{ts1}', f'{ts2}']]
    df.dropna(inplace=True)  # Both Pearson and Spearman correlations cannot handle NaN values, so make sure to drop nan values before any calculatoin.
    
    # Compute statistics for each time series
    stats = {
        'Mean': [df[f'{ts1}'].mean(), df[f'{ts2}'].mean()],
        'Median': [df[f'{ts1}'].median(), df[f'{ts2}'].median()],
        'Standard Deviation': [df[f'{ts1}'].std(), df[f'{ts2}'].std()],
        'Variance': [df[f'{ts1}'].var(), df[f'{ts2}'].var()],
        'Min': [df[f'{ts1}'].min(), df[f'{ts2}'].min()],
        'Max': [df[f'{ts1}'].max(), df[f'{ts2}'].max()]
    }

    # Calculate correlation coefficients
    pearson_corr, _ = pearsonr(df[f'{ts1}'], df[f'{ts2}'])
    spearman_corr, _ = spearmanr(df[f'{ts1}'], df[f'{ts2}'])

    # Compute Bias (mean error)
    bias = df[ts2].mean() - df[ts1].mean()

    # Compute Nash-Sutcliffe Efficiency (NSE)
    obs_mean = df[ts1].mean()
    numerator = np.sum((df[ts2] - df[ts1])**2)
    denominator = np.sum((df[ts1] - obs_mean)**2)
    nse = 1 - (numerator / denominator)

    # Compute Kling-Gupta Efficiency (KGE)
    r = pearson_corr
    alpha = df[ts2].std() / df[ts1].std()
    beta = df[ts2].mean() / df[ts1].mean()
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # Create a DataFrame for the statistics
    stats_table = pd.DataFrame(stats, index=['observed', 'modeled'])

    # Add Pearson and Spearman correlations as additional rows
    stats_table.loc[''] = [''] * len(stats_table.columns)  # Blank row for formatting
    stats_table.loc['Pearson Correlation'] = [pearson_corr, '', '', '', '', '']
    stats_table.loc['Spearman Correlation'] = [spearman_corr, '', '', '', '', '']
    stats_table.loc['Bias (Modeled - Observed)'] = [bias, '', '', '', '', '']
    stats_table.loc['Nash-Sutcliffe Efficiency (NSE)'] = [nse, '', '', '', '', '']
    stats_table.loc['Kling-Gupta Efficiency (KGE)'] = [kge, '', '', '', '', '']

    return stats_table

def comparison_plots(df, ts1, ts2):
    '''
    ts1: observed
    ts2: modeled
    '''
    # Timeseries plot (Overlay)
    observed_plot = df.hvplot.line(
        y=ts1,
        ylabel='Snow Water Equivalent (m)',
        label='Observed SWE',
        color='blue',
        line_width=2,
        width=400,
        height=300,
    )

    modeled_plot = df.hvplot.line(
    y=ts2,
    ylabel='Snow Water Equivalent (m)',
    label='Modeled SWE',
    color='orchid',
    line_width=2,
    width=400,
    height=300,
    )

    # Overlay (combines both lines into a single visual object)
    timeseries_plot = (observed_plot * modeled_plot).opts(
        title='Snow Water Equivalent Comparison',
        legend_position='top',
    )

    # Scatter plot
    scatter_plot = df.hvplot.scatter(
        x=ts1,
        y=ts2,
        xlabel='Observed SWE (m)',
        ylabel='Modeled SWE (m)',
        color='black',
        width=400,
        height=300,
        size=15,
        hover_cols=['index']  # This will add the date (index) to hover tooltip
    )

    # Add 1:1 line (perfect match line)
    swe_max = max(df[ts1].max(), df[ts2].max())
    one_to_one_line = hv.Curve(([0, swe_max], [0, swe_max])).opts(
        color='gray',
        line_dash='solid',
        line_width=1,
    ).relabel('1:1 Line')  # This is the correct way to set a label for a Curve
    
    # Combine scatter plot and 1:1 line into an Overlay
    scatter_with_line = (scatter_plot * one_to_one_line).opts(
        legend_position='top'
    )
    
    # Explicitly convert Overlay to Layout
    timeseries_plot = hv.Layout([timeseries_plot])
    
    # Combine both into a 1-row, 2-column layout
    layout = hv.Layout([timeseries_plot, scatter_with_line]).opts(shared_axes=False)

    return layout

def plot_custom_scatter(df, site_code, highlight_months=None):
    
    if highlight_months is None:
        highlight_months = [10]  # Default if none provided

    # Define color column based on the provided months
    df = df.copy()  # Just in case you want to avoid modifying the original dataframe
    df['color'] = df['month'].apply(lambda x: 'teal' if x in highlight_months else 'tomato')

    # Create the scatter plot
    scatter_plot = df.hvplot.scatter(
        x=f'CCSS_{site_code}_swe_m',
        y=f'NWM_{site_code}_swe_m',
        xlabel='Observed SWE (m)',
        ylabel='Modeled SWE (m)',
        title='Observed vs. Modeled SWE',
        size=15,
        width=400,
        height=300,
        hover_cols=['index', 'month'],
        color='color'
    )

    # Add 1:1 line (perfect match line)
    swe_max = max(df[f'CCSS_{site_code}_swe_m'].max(), df[f'NWM_{site_code}_swe_m'].max())
    one_to_one_line = hv.Curve(([0, swe_max], [0, swe_max])).opts(
        color='gray',
        line_dash='dashed',
        line_width=1,
    ).relabel('1:1 Line')

    # Combine scatter plot and 1:1 line into an Overlay
    scatter_with_line = (scatter_plot * one_to_one_line).opts(
        legend_position='top'
    )

    return scatter_with_line

def plot_grid_vector_data(ds_clip, data_var, time_index, shp, sites):
    hv.extension('bokeh')
    da = ds_clip[data_var]

    # Select one timestep
    if isinstance(time_index, int):
        da = da.isel(time=time_index)
    else:
        da = da.sel(time=time_index)

    # Create an interactive map plot
    clipped = da.rio.reproject("EPSG:4326")
    clipped = clipped.rename({'x': 'longitude', 'y': 'latitude'})
    hvplot_map = clipped.hvplot(
        x='longitude',
        y='latitude', 
        geo=True,
        project=True,
        tiles=gts.ESRI,
        cmap='kbc',
        alpha=0.6,
        frame_height=400,
        title=f"Snow Water Equivalent, at {pd.to_datetime(time_index).strftime('%Y-%m-%d %H:%M')}",
        clim=(0, 300)
    )
    
    shp = shp.to_crs("EPSG:4326").reset_index(drop=True)
    sites = sites.to_crs("EPSG:4326").reset_index(drop=True)

    # Plot the shapefile outline
    shp_plot = shp.hvplot(
    geo=True, project=True,
    color='none', line_width=2
    )

    # Plot sites (scatter)
    points_plot = sites.hvplot.points(
    x='longitude', y='latitude',
    geo=True, project=True,
    color='red', size=100, hover_cols=['name']
    )

    # Combine the two by overlaying
    combined_map = (hvplot_map * shp_plot * points_plot).opts(framewise=True)
    
    return combined_map

def plot_grid_vector_monthly_data(ds_clip, data_var, shp, sites):
    hv.extension('bokeh')

    # Create an interactive map plot
    clipped = ds_clip[data_var].rio.reproject("EPSG:4326")
    clipped = clipped.rename({'x': 'longitude', 'y': 'latitude'})
    
    # Plot the shapefile outline
    shp_plot = shp.hvplot(
    geo=True, project=True,
    color='none', line_width=2
    )

    # Plot sites (scatter)
    points_plot = sites.hvplot.points(
    x='longitude', y='latitude',
    geo=True, project=True,
    color='red', size=100, hover_cols=['name']
    )

    # Split into individual plots (list of plots)
    plots = []
    for t in clipped.time.values:
        base_plot = clipped.sel(time=t).hvplot(
            x='longitude', y='latitude',
            geo=True, project=True,
            tiles=gts.ESRI,
            title=f'SWE (mm) on {pd.to_datetime(t).strftime("%Y-%m-%d")}',
            frame_height=200, frame_width=300
        )
        # Overlay shapefile and points on top of SWE map
        combined_plot = base_plot * shp_plot * points_plot
        plots.append(combined_plot)
        
    layout = hv.Layout(plots).cols(3)
    
    return layout