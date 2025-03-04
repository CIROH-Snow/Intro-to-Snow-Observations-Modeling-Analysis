import pandas as pd
import os


def processSNOTEL(site, stateab):
    print(site)

    sitedf = pd.read_csv(f"files/SNOTEL/df_{site}_{stateab}_SNTL.csv")

    WYs = sitedf['Water_Year'].unique()

    WYsitedf = pd.DataFrame()

    for WY in WYs:
        cols =['M', 'D', 'Snow Water Equivalent (m) Start of Day Values']

        #get water year of interest
        wydf = sitedf[sitedf['Water_Year']==WY]
        wydf['M'] = pd.to_datetime(sitedf['Date']).dt.month
        wydf['D'] = pd.to_datetime(sitedf['Date']).dt.day

        #change NaN to 0, most NaN values are from low to 0 SWE measurements
        wydf['Snow Water Equivalent (m) Start of Day Values'] = wydf['Snow Water Equivalent (m) Start of Day Values'].fillna(0)
        wydf = wydf[cols]
        wydf.rename(columns = {'Snow Water Equivalent (m) Start of Day Values':f"{WY}_SWE_m"}, inplace=True)
        wydf.reset_index(inplace=True, drop=True)
        WYsitedf[f"{WY}_SWE_in"] = wydf[f"{WY}_SWE_m"]*39.3701 #converting m to inches (standard for snotel)

        if len(wydf) == 365:
            try:
                WYsitedf.insert(0,'M',wydf['M'])
                WYsitedf.insert(1,'D',wydf['D'])
            except:
                pass
    #WYsitedf.fillna(0)

    #remove July, August, September
    months = [8,9]
    WYsitedf = WYsitedf[~WYsitedf['M'].isin(months)]

    #remove M/D to calculate row min, mean, median, max tiers
    coldrop = ['M', 'D']
    df = WYsitedf.drop(columns = coldrop)
    df['min'] = df.min(axis=1)
    df['Q10'] = df.quantile(0.10, axis=1)
    df['Q25'] = df.quantile(0.25, axis=1)
    df['mean'] = df.mean(axis=1)
    df['median'] = df.median(axis=1)
    df['Q75'] = df.quantile(0.75, axis=1)
    df['Q90'] = df.quantile(0.90, axis=1)
    df['max'] = df.max(axis=1)

    #add back in M/d for plotting
    df.insert(0,'M',WYsitedf['M'])
    df.insert(1,'D',WYsitedf['D'])

    # Convert to datetime format
    df['date'] = pd.to_datetime(dict(year = 2023, month = df['M'], day = df['D'])) 

    # Format the date
    df['M-D'] = df['date'].dt.strftime('%m-%d')
    df.set_index('M-D', inplace=True)

    return df


def Spatial_median_SWE_df(output_res, basinname, begdate, enddate, filename, decround,  save = True):

    #Get all file names for ASO images in Tuolumne river basin
    files = [f for f in os.listdir(f"files/ASO/{basinname}/{output_res}M_SWE_parquet/") if os.path.isfile(os.path.join(f"files/ASO/{basinname}/{output_res}M_SWE_parquet/", f))]

    SWE_tempDF = pd.DataFrame()
    datefiles = [f for f in files if int(f[-11:-8]) >= begdate and int(f[-11:-8]) <= enddate]

    #Load and combine all files into a single dataframe
    for file in datefiles:
        df = pd.read_parquet(f"files/ASO/{basinname}/{output_res}M_SWE_parquet/{file}")
        df['Y'] = int(file[-16:-12])
        df['M'] = int(file[-12:-10])
        df['D'] = int(file[-10:-8])
        #convert m to in to be consistent with SNOTEL
        df['swe_in'] = df['swe_m'] * 39.3701
        locations = []
        for index, row in df.iterrows():
            location = f"{basinname}_{output_res}M_{round(row['cen_lat'],decround)}_{round(row['cen_lon'],decround)}"
            locations.append(location)
        df['location'] = locations
        SWE_tempDF = pd.concat([SWE_tempDF, df])

        #get the median SWE for each location
    locations = SWE_tempDF['location'].unique()

    for location in locations:
        locationDF = SWE_tempDF[SWE_tempDF['location'] == location]
        if len(locationDF) > 1:
            SWE_tempDF.loc[SWE_tempDF['location'] == location, 'median_SWE_m'] = locationDF['swe_m'].median()
        #else:
         #    SWE_tempDF.loc[SWE_tempDF['location'] == location, 'median_SWE_m'] = locationDF['swe_m']
        # if len(locationDF) > 3:
        #     display(locationDF)

    SWE_tempDF['median_SWE_in'] = SWE_tempDF['median_SWE_m'] * 39.3701

    #round lat and lon to desired decimal places
    SWE_tempDF['cen_lat'] = SWE_tempDF['cen_lat'].round(decround)
    SWE_tempDF['cen_lon'] = SWE_tempDF['cen_lon'].round(decround)
    #make a median SWE DF
    cols = ['cen_lat', 'cen_lon', 'location', 'median_SWE_m', 'median_SWE_in']
    MedianSWE_df = SWE_tempDF.drop_duplicates(subset=['location'], keep='first').copy()
    MedianSWE_df = MedianSWE_df[cols]

    #make a median SWE DF
    cols = ['cen_lat', 'cen_lon', 'location', 'median_SWE_m', 'median_SWE_in']
    MedianSWE_df = SWE_tempDF.drop_duplicates(subset=['location'], keep='first').copy()
    MedianSWE_df = MedianSWE_df[cols]
    totalobs = len(MedianSWE_df)

    #drop rows with less than 2 obs
    MedianSWE_df = MedianSWE_df.dropna(subset=['median_SWE_m'])
    print(f"Number of locations with median SWE: {len(MedianSWE_df)}, dropped {totalobs - len(MedianSWE_df)} locations because of only 1 observation")
    
    if save == True:
        filepath = f"files/ASO/{basinname}/{output_res}M_SWE_parquet/"
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
        MedianSWE_df.to_parquet(f"{filepath}/{filename}")

    return MedianSWE_df

def SWE_diff(basinname, output_res, medianSWEfile, WYSWEfile, decround, swedifffilename, save =True):
    #load the median SWE data
    MedianSWE_df = pd.read_parquet(f"files/ASO/{basinname}/{output_res}M_SWE_parquet/{medianSWEfile}")
    MedianSWE_df.set_index('location', inplace = True)

    #load year of interest
    yeardf = pd.read_parquet(f"files/ASO/{basinname}/{output_res}M_SWE_parquet/{WYSWEfile}")

    #need to round only to 2 decimal places for the location
    locations = []
    for index, row in yeardf.iterrows():
            location = f"{basinname}_{output_res}M_{round(row['cen_lat'],decround)}_{round(row['cen_lon'],decround)}"
            locations.append(location)
    yeardf['location'] = locations
    yeardf = yeardf.drop_duplicates(subset=['location'], keep='first').copy()
    #yeardf.rename(columns = {'cell_id': 'location'}, inplace = True)
    yeardf.set_index('location', inplace = True)
    dropcols = ['cen_lat', 'cen_lon', 'cell_id']
    yeardf = yeardf.drop(columns = dropcols)
    yeardf['swe_in'] = yeardf['swe_m'] * 39.3701

    #Add median SWE to the year of interest
    df = pd.concat([yeardf, MedianSWE_df], axis = 1)

    #drop rows without a swe observation
    df = df.dropna(subset = ['swe_in'])
    df = df.dropna(subset = ['median_SWE_in'])

    #Calculate the difference between the median SWE and the year of interest
    df['SWE_diff_in'] = df['swe_in'] - df['median_SWE_in']
    df['SWE_diff_m'] = df['SWE_diff_in'] / 39.3701
    Perc = []
    for index, row in df.iterrows():
            
            perc = (row['SWE_diff_in']/row['median_SWE_in'])*100
            if perc == float('inf'):
                perc = (row['SWE_diff_in']/1)*100
            Perc.append(perc)
    Perc = [round(i,0) for i in Perc]

    df['SWE_perc_norm'] = Perc

    #limit the percent difference to 500%
    df['SWE_perc_norm'][df['SWE_perc_norm']>500] = 500
    df['SWE_perc_norm'][df['SWE_perc_norm']<-500] = -500

    if save == True:
        df.to_parquet(f"files/ASO/{basinname}/{output_res}M_SWE_parquet/{swedifffilename}")

    return df