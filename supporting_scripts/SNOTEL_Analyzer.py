import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd



def SNOTELPlots(sitedict, gdf_in_bbox, WY, watershed, AOI, DOI,plot = True):

    title = f'Snow Outlook for {watershed} Basin \n {AOI} for WY {WY}'



    fig, axs = plt.subplots(2,2, figsize = (8,8))
    fig.suptitle(title)
    opacity = 0.25
    WYOI = f"{WY}_SWE_in"

    axs = axs.ravel()
    for i, key in enumerate(sitedict.keys()):
        df = sitedict[key]

        axs[i].set_title(f"SNOTEL Site: {gdf_in_bbox['name'][gdf_in_bbox['code']==key].item()}")
        #check dataframe for respective water year
        if f"{WY}_SWE_in" in df.columns:

            #key swe lines on SNOTEL plot
            axs[i].plot(df['max'], color = 'slateblue', label = 'Max')
            axs[i].plot(df['median'], color = 'green', label = 'Median')
            axs[i].plot(df['min'], color = 'red', label = 'Min')

            #Fill between Quantiles
            axs[i].fill_between(df.index, df['max'], df['Q90'], color = 'slateblue', alpha = opacity, label = 'Q90')
            axs[i].fill_between(df.index, df['Q90'], df['Q75'], color = 'cyan', alpha = opacity, label = 'Q75')
            axs[i].fill_between(df.index, df['Q75'], df['Q25'], color = 'green', alpha = opacity)
            axs[i].fill_between(df.index, df['Q25'], df['Q10'], color = 'yellow', alpha = opacity, label = 'Q25')
            axs[i].fill_between(df.index, df['Q10'], df['min'], color = 'red', alpha = opacity, label = 'Q10')

            #Plotting year of interest
            axs[i].plot(df[WYOI], color = 'black', label = f"WY {WY}")

              # Plot vertical line at a specific date
            axs[i].axvline(DOI, color='black', linestyle='--')


            axs[i].xaxis.set_major_locator(ticker.MaxNLocator(4))
            axs[i].tick_params(labelrotation=45)
            handles, labels = axs[i].get_legend_handles_labels()

            # Add text box in the upper left portion of the subplot
            
            mpeak = max(df['median'])
            mpeakday = f"{WY}-{df.index[df['median']==mpeak][0]}"
            WYpeak = max(df[WYOI])
            WYpeakday = f"{WY}-{df.index[df[WYOI]==WYpeak][0]}"
            doivalue = df.loc[DOI, WYOI] if DOI in df.index else None
            doimed = df.loc[DOI, 'median'] if DOI in df.index else None
            PSWEDiff_day = (pd.to_datetime(WYpeakday)-pd.to_datetime(mpeakday)).days
            medpercPeak = round(doivalue/mpeak *100, 0)
            medperc = round(doivalue/doimed *100, 0)

            textstr = f"DOI: {WY}-{DOI} \n % of median - {medperc}%  \n % of median peak - {medpercPeak}% \n Peak SWE Date: {WYpeakday}  \n Days from Median Peak - {PSWEDiff_day}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            axs[i].text(0.05, 0.95, textstr, transform=axs[i].transAxes, fontsize=6,
                    verticalalignment='top', bbox=props)


        else:
            axs[i].annotate('No Data', xy=(0.45, 0.45), xytext=(0.45, 0.45))

         # Set axis labels
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel('SWE (inches)')

 
            
    fig.legend(handles, labels,loc='lower center',ncol=8, bbox_to_anchor=(.5, -.05))
    plt.tight_layout()

    if plot == True:
        if not os.path.exists('Figures'):
            os.makedirs('Figures')
        fig.savefig(f"Figures/{watershed}_{WY}_snotelanalysis.png",  dpi = 600, bbox_inches='tight')




def catchmentSNOTELAnalysis(sitedict, WY, watershed, AOI, DOI,plot = True):
    WYOI = f"{WY}_SWE_in"
    sites = sitedict.keys()
    basindf =  pd.DataFrame()
    columns =['min', 'Q10', 'Q25', 'mean', 'median','Q75', 'Q90', 'max', f"{WY}_SWE_in"]
    Basin_df = pd.DataFrame()

    sitedict = {key: df for key, df in sitedict.items() if WYOI in df.columns}
    sites = sitedict.keys()

    for column in columns:
        # Extract the column values from each DataFrame and concatenate them
        all_values = pd.concat([df[column] for df in sitedict.values()], axis=1)

        # Calculate the mean
        mean_value = all_values.mean(axis=1)
        Basin_df[column] = mean_value
    title = f'Snow Outlook for {watershed} Basin \n {AOI} for WY {WY}'
    df = Basin_df.copy()


    fig, axs = plt.subplots(1,1, figsize = (8,8))
    fig.suptitle(title)
    opacity = 0.25
   

    #key swe lines on SNOTEL plot
    axs.plot(df['max'], color = 'slateblue', label = 'Max')
    axs.plot(df['median'], color = 'green', label = 'Median')
    axs.plot(df['min'], color = 'red', label = 'Min')

    #Fill between Quantiles
    axs.fill_between(df.index, df['max'], df['Q90'], color = 'slateblue', alpha = opacity, label = 'Q90')
    axs.fill_between(df.index, df['Q90'], df['Q75'], color = 'cyan', alpha = opacity, label = 'Q75')
    axs.fill_between(df.index, df['Q75'], df['Q25'], color = 'green', alpha = opacity)
    axs.fill_between(df.index, df['Q25'], df['Q10'], color = 'yellow', alpha = opacity, label = 'Q25')
    axs.fill_between(df.index, df['Q10'], df['min'], color = 'red', alpha = opacity, label = 'Q10')

    #Plotting year of interest
    axs.plot(df[WYOI], color = 'black', label = f"WY {WY}")

        # Plot vertical line at a specific date
    axs.axvline(DOI, color='black', linestyle='--')


    axs.xaxis.set_major_locator(ticker.MaxNLocator(4))
    axs.tick_params(labelrotation=45)
    handles, labels = axs.get_legend_handles_labels()

    # Add text box in the upper left portion of the subplot
    
    mpeak = max(df['median'])
    mpeakday = f"{WY}-{df.index[df['median']==mpeak][0]}"
    WYpeak = max(df[WYOI])
    WYpeakday = f"{WY}-{df.index[df[WYOI]==WYpeak][0]}"
    doivalue = df.loc[DOI, WYOI] if DOI in df.index else None
    doimed = df.loc[DOI, 'median'] if DOI in df.index else None
    PSWEDiff_day = (pd.to_datetime(WYpeakday)-pd.to_datetime(mpeakday)).days
    medpercPeak = round(doivalue/mpeak *100, 0)
    medperc = round(doivalue/doimed *100, 0)

    textstr = f"DOI: {WY}-{DOI} \n % of median - {medperc}%  \n % of median peak - {medpercPeak}% \n Peak SWE Date: {WYpeakday}  \n Days from Median Peak - {PSWEDiff_day} \n Statistics based on {len(sites)} SNOTEL sites"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

        # Set axis labels
    axs.set_xlabel('Date')
    axs.set_ylabel('SWE (inches)')


            
    fig.legend(handles, labels,loc='upper right',ncol=1, bbox_to_anchor=(.97, .9))
    plt.tight_layout()

    if plot == True:
        if not os.path.exists('Figures'):
            os.makedirs('Figures')
        fig.savefig(f"Figures/{watershed}_{WY}_Basinsnotelanalysis.png",  dpi = 600, bbox_inches='tight')
