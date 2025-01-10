import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd



def SNOTELPlots(sitedict, gdf_in_bbox, WY, watershed, AOI, DOI,plot = True):

    title = f'Snow Outlook for {watershed} Basin \n {AOI} for WY {WY}'



    fig, axs = plt.subplots(2,2, figsize = (8,8))
    fig.suptitle(title)
    opacity = 0.25


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
            axs[i].plot(df[f"{WY}_SWE_in"], color = 'black', label = f"WY {WY}")

              # Plot vertical line at a specific date
            axs[i].axvline(DOI, color='black', linestyle='--')


            axs[i].xaxis.set_major_locator(ticker.MaxNLocator(4))
            axs[i].tick_params(labelrotation=45)
            handles, labels = axs[i].get_legend_handles_labels()

            # Add text box in the upper left portion of the subplot
            mpeak = max(df['median'])
            doivalue = df.loc[DOI, f"{WY}_SWE_in"] if DOI in df.index else None
            doimed = df.loc[DOI, 'median'] if DOI in df.index else None


            medpercPeak = round(doivalue/mpeak *100, 0)
            medperc = round(doivalue/doimed *100, 0)

            # medpeak = 
            # dpeak = 
            # percentile = 
            textstr = f"DOI: {WY}-{DOI} \n % of median - {medperc}%  \n % of median peak - {medpercPeak}% "# \n Days from Median Peak - {dpeak} \n Percentile - {percentile}"
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            axs[i].text(0.05, 0.95, textstr, transform=axs[i].transAxes, fontsize=8,
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

