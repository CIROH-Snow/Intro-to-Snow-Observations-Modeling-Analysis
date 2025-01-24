# Import packages
# Dataframe Packages
import numpy as np
from numpy import gradient, rad2deg, arctan2
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Vector Packages
import geopandas as gpd
import shapely
from shapely import wkt
from shapely.geometry import Point, Polygon
from pyproj import CRS, Transformer

# Raster Packages
import rioxarray as rxr
import rasterio
from rasterio.mask import mask
from rioxarray.merge import merge_arrays
import rasterstats as rs
from osgeo import gdal
from osgeo import gdalconst

# Data Access Packages
import earthaccess as ea
import pickle
import pystac_client
#import richdem as rd
import planetary_computer
from planetary_computer import sign

# General Packages
import os
import re
import shutil
import math
from datetime import datetime
import glob
from pprint import pprint
from typing import Union
from pathlib import Path
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
import time
import requests
import concurrent.futures as cf
# import fiona
import re
# import s3fs

#need to mamba install gdal, earthaccess 
#pip install pystac_client, richdem, planetary_computer, dask, distributed, retrying

#connecting to AWS
import warnings; warnings.filterwarnings("ignore")
import pickle as pk


#Processing using gdal
def process_single_location(args):
    location, lat, lon, DEMs, tiles = args
    
    #maybe thorugh a try/except here, look up how to find copernicus data --Problem is this! and finding the nearest location below
    try:
        tile_id = f"Copernicus_DSM_COG_30_N{str(math.floor(lat))}_00_W{str(math.ceil(abs(lon)))}_00_DEM"
        index_id = DEMs.loc[tile_id]['sliceID']

        signed_asset = planetary_computer.sign(tiles[int(index_id)].assets["data"])

        elevation = rxr.open_rasterio(signed_asset.href)
        
        slope = elevation.copy()
        aspect = elevation.copy()

        transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)
        xx, yy = transformer.transform(lon, lat)

        tilearray = np.around(elevation.values[0]).astype(int)
        geo = (math.floor(float(lon)), 90, 0.0, math.ceil(float(lat)), 0.0, -90)

        driver = gdal.GetDriverByName('MEM')
        temp_ds = driver.Create('', tilearray.shape[1], tilearray.shape[0], 1, gdalconst.GDT_Float32)

        temp_ds.GetRasterBand(1).WriteArray(tilearray)

        tilearray_np = temp_ds.GetRasterBand(1).ReadAsArray()
        grad_y, grad_x = gradient(tilearray_np)

        # Calculate slope and aspect
        slope_arr = np.sqrt(grad_x**2 + grad_y**2)
        aspect_arr = rad2deg(arctan2(-grad_y, grad_x)) % 360 
        
        slope.values[0] = slope_arr
        aspect.values[0] = aspect_arr

        elev = round(elevation.sel(x=xx, y=yy, method="nearest").values[0])
        slop = round(slope.sel(x=xx, y=yy, method="nearest").values[0])
        asp = round(aspect.sel(x=xx, y=yy, method="nearest").values[0])
    except:
        elev, slop, asp = np.nan, np.nan, np.nan
        print(f"{location} does not have copernicus DEM data, manual input")

    return location, elev, slop, asp


def extract_terrain_data_threaded(metadata_df, basinname, output_res):
    global elevation_cache 
    elevation_cache = {} 
    metadata_df.reset_index(inplace=True)
    #convert to geodataframe
    metadata_df = gpd.GeoDataFrame(metadata_df, geometry=gpd.points_from_xy(metadata_df.cen_lon, metadata_df.cen_lat))
    metadata_df.crs = "EPSG:4326"

    print('Calculating dataframe bounding box')
    bounding_box = metadata_df.geometry.total_bounds
    #get the max and mins to make sure we get all geos
    min_x, min_y, max_x, max_y = math.floor(bounding_box[0])-1, math.floor(bounding_box[1])-1, math.ceil(bounding_box[2])+1, math.ceil(bounding_box[3])+1
    print(min_x, min_y, max_x, max_y)
    client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            ignore_conformance=True,
        )

    search = client.search(
                    collections=["cop-dem-glo-90"],
                    intersects = {
                            "type": "Polygon",
                            "coordinates": [[
                            [min_x, min_y],
                            [max_x, min_y],
                            [max_x, max_y],
                            [min_x, max_y],
                            [min_x, min_y]  
                        ]]})

    tiles = list(search.items())

    DEMs = []

    print("Retrieving Copernicus 90m DEM tiles")
    for i in tqdm_notebook(range(0, len(tiles))):
        row = [i, tiles[i].id]
        DEMs.append(row)
    DEMs = pd.DataFrame(columns = ['sliceID', 'tileID'], data = DEMs)
    DEMs = DEMs.set_index(DEMs['tileID'])
    del DEMs['tileID']
    print(f"There are {len(DEMs)} tiles in the watershed")


    print("Determining Grid Cell Spatial Features")

    
    results = []
    with cf.ThreadPoolExecutor(max_workers=None) as executor:
        jobs = {executor.submit(process_single_location, (metadata_df.iloc[i]['location'], metadata_df.iloc[i]['cen_lat'], metadata_df.iloc[i]['cen_lon'], DEMs, tiles)): 
                i for i in tqdm_notebook(range(len(metadata_df)))}
        
        print(f"Job complete for getting geospatial metadata, putting into dataframe")
        for job in tqdm_notebook(cf.as_completed(jobs)):
            results.append(job.result())
   
    # for i in tqdm_notebook(range(len(metadata_df))):
    #     location, elev, slop, asp = process_single_location((metadata_df.iloc[i]['location'], metadata_df.iloc[i]['cen_lat'], metadata_df.iloc[i]['cen_lon'], DEMs, tiles))
    #     site = [location, elev, slop, asp]
    #     results.append(site)

            

    meta = pd.DataFrame(results, columns=['location', 'Elevation_m', 'Slope_Deg', 'Aspect_Deg'])
    meta.set_index('location', inplace=True)
    metadata_df.set_index('location', inplace=True)
    metadata_df = pd.concat([metadata_df, meta], axis = 1)

    #save watershed dataframe
    dfpath = f"files/ASO/{basinname}"
    print(f"Saving  dataframe in {dfpath}")
    
    # Save the DataFrame as a parquet file
    #Convert DataFrame to Apache Arrow Table, drop the geometry column to play nice with parquet files
    metadata_df.pop('geometry')
    table = pa.Table.from_pandas(metadata_df)
    # Parquet with Brotli compression
    pq.write_table(table, f"{dfpath}/{output_res}_metadata.parquet", compression='BROTLI')
        
    return metadata_df#, DEMs, tiles