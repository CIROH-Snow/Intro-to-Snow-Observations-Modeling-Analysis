import pandas as pd



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