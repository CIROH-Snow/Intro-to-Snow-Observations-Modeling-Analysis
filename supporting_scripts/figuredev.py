import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def SNOTELPlots(sitedict, gdf_in_bbox, WY, watershed, plot = True):

    title = f'Snow Outlook for Tuolumne River Basin \n Above Hetch Hetchy Reservoir for WY {WY}'



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
            axs[i].plot(df[f"{WY}_SWE_in"], color = 'black')

            axs[i].xaxis.set_major_locator(ticker.MaxNLocator(4))
            axs[i].tick_params(labelrotation=45)
            handles, labels = axs[i].get_legend_handles_labels()
        else:
            axs[i].annotate('No Data', xy=(0.45, 0.45), xytext=(0.45, 0.45))
            
    fig.legend(handles, labels,loc='lower center',ncol=7, bbox_to_anchor=(.5, -.05))
    plt.tight_layout()

    if plot == True:
        fig.savefig(f"Figures/{watershed}_{WY}_snotelanalysis.png",  dpi = 600, bbox_inches='tight')

