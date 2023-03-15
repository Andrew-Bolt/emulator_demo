from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import raster_geometry_mask
from sklearn.model_selection import train_test_split
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import ntpath
import os
import glob
import json
from datetime import datetime
import re

from skimage import measure
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely
import random
import math

def rowcolheightwidth_from_crspoint(src_image, tgt_image):
    
    """ finds the row and column in the source image 
     that correspond to the bounds of the target image) """
    
    with rasterio.open(tgt_image):
        
        tgt_left   = tgt.bounds.left
        tgt_right  = tgt.bounds.right
        tgt_top    = tgt.bounds.top
        tgt_bottom = tgt.bounds.bottom
        height = tgt.height
        width = tgt.width
        
    
    src_left, src_top = rasterio.transform.TransformMethodsMixin(tgt_left, tgt_top)
    src_right, src_bottom = raterio.transform.TransformMethodsNixin(tgt_right, tgt_bottom)
    
    # return the area of interest 
    
    return (src_left, src_top, src_right, src_bottom, src_height, src_width)

    
    
def transform_crs(src_image, tgt_image, proc_path, resampling=Resampling.nearest, num_threads=4, no_data=65526):
    """
    Takes an image and transforms it into a target coordinate system (crs). 
    Resampling can be specified and a mapping function applied for categorical data.

    Takes a source image (src) and transforms the CRS and size to conform with some target image (tgt).
    The destination image is stored within "./tmp" with the same name as the source image.

    """

    if resampling == "bilinear":
        resampling = Resampling.bilinear
    else:
        resampling = Resampling.nearest

    with rasterio.open(tgt_image) as tgt:

        tgt_crs    = tgt.crs
        tgt_res    = tgt.res
        #tgt_height = tgt.height # ignore comparisons target's height and width
        #tgt_width  = tgt.width
        #tgt_left   = tgt.bounds.left
        #tgt_bottom = tgt.bounds.bottom
        #tgt_right  = tgt.bounds.right
        #tgt_top    = tgt.bounds.top


    with rasterio.open(src_image) as src:
        
        src_height = src.height
        src_width  = src.width
        src_left   = src.bounds.left
        src_bottom = src.bounds.bottom
        src_right  = src.bounds.right
        src_top    = src.bounds.top
        
        transform = rasterio.transform.from_bounds(
            src_left, src_bottom, src_right, src_top, src_width, src_height)
        ## copy out entire image

        # update kwargs
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': tgt_crs,
            'transform': transform,
            'width': src_width,
            'height': src_height,
            'resolution' : tgt_res 
        })


        with rasterio.open(proc_path + os.path.basename(src_image), 'w', **kwargs) as dst:

            # apply the reprojection
            
            reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_nodata = no_data,
                    dst_nodata = 0,
                    src_transform=transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=tgt_crs,
                    dst_resolution = tgt_crs, # ensure same resolution
                    resampling=resampling,
                    num_threads=num_threads)
    

def transform_crs_old(src_image, tgt_image, proc_path, resampling=Resampling.nearest, num_threads=4, no_data=65526):
    """
    Takes an image and transforms it into a target coordinate system (crs). 
    Resampling can be specified and a mapping function applied for categorical data.

    Takes a source image (src) and transforms the CRS and size to conform with some target image (tgt).
    The destination image is stored within "./tmp" with the same name as the source image.

    """

    if resampling == "bilinear":
        resampling = Resampling.bilinear
    else:
        resampling = Resampling.nearest

    with rasterio.open(tgt_image) as tgt:

        tgt_crs    = tgt.crs
        tgt_res    = tgt.res
        tgt_height = tgt.height
        tgt_width  = tgt.width
        tgt_left   = tgt.bounds.left
        tgt_bottom = tgt.bounds.bottom
        tgt_right  = tgt.bounds.right
        tgt_top    = tgt.bounds.top


    with rasterio.open(src_image) as src:

        transform = rasterio.transform.from_bounds(
            tgt_left, tgt_bottom, tgt_right, tgt_top, tgt_width, tgt_height)
        ##print((tgt_left, tgt_bottom, tgt_right, tgt_top, tgt_width, tgt_height))

        # update kwargs
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': tgt_crs,
            'transform': transform,
            'width': tgt_width,
            'height': tgt_height,
            'resolution' : tgt_res 
        })


        with rasterio.open(proc_path + os.path.basename(src_image), 'w', **kwargs) as dst:

            # apply the reprojection
            
            reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_nodata = no_data,
                    dst_nodata = 0,
                    src_transform=transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=tgt_crs,
                    resampling=resampling,
                    num_threads=num_threads)
            
            
def rescale_heights(hm_image):
    """
    Scales the pixel values of a source image by the images resolution.
    Recentres the average height to zero
    Returns the max and min height
    """

    with rasterio.open(hm_image) as src:

        height_array = src.read(1)
        #height_array[height_array > 10000] = 0 # zero N/A values
        
        
        height_array = height_array / src.res[0] # express height in pixel values
        height_array = height_array - np.mean(height_array) # 0-average the heights
        
        max_height = np.max(height_array)
        min_height = np.min(height_array)
        
        profile = src.profile
        profile.update(
        dtype=rasterio.float64)
        

    # now need to save this array data in a destination TIF file  
    with rasterio.Env():

        with rasterio.open(hm_image, 'w', **profile) as dst:

            #dst.write(height_array.astype(profile['dtype']), 1) 
            dst.write(height_array, 1)
            
    return max_height, min_height


            
def rescale_heights_old(hm_image):
    """
    Scales the pixel values of a source image by the images resolution.
    Recentres the average height to zero
    Returns the max and min height
    """

    with rasterio.open(hm_image) as src:

        height_array = src.read(1)
        #height_array[height_array > 10000] = 0 # zero N/A values
        
        
        height_array = height_array / src.res[0] # express height in pixel values
        height_array = height_array - np.mean(height_array) # 0-average the heights
        
        max_height = np.max(height_array)
        min_height = np.min(height_array)
        
        profile = src.profile
        profile.update(
        dtype=rasterio.float64)
        

    # now need to save this array data in a destination TIF file  
    with rasterio.Env():

        with rasterio.open(hm_image, 'w', **profile) as dst:

            #dst.write(height_array.astype(profile['dtype']), 1) 
            dst.write(height_array, 1)
            
    return max_height, min_height




def zeroed_heightmap(src_image, tgt_name):
    """scales the pixel values of a source image by the images resolution"""

    with rasterio.open(src_image) as src:

        height_array = src.read(1) 
        height_array = height_array * 0 # all values zeroed
        profile = src.profile

    # now need to save this array data in a destination TIF file  
    with rasterio.Env():

        with rasterio.open(tgt_name, 'w', **profile) as dst:

            dst.write(height_array.astype(profile['dtype']), 1) 





def transform_classes(src_image, class_map, path, mask = None):
    """ applys masking and land model classification transformations"""
    num_bands = 6


    # open fuel class map
    with rasterio.open(src_image) as src:

        # apply the class map function to the images data array
        image_array = class_map(src.read(1))

        # apply burn mask
        if mask is not None:    
            burn_mask = gpd.read_file(mask)
            burn_mask = burn_mask.to_crs(src.crs)
            # creates a masking array
            burn_mask = raster_geometry_mask(src, burn_mask.geometry, crop=False) 

            # apply the burn mask to the data
            image_array[~burn_mask[0]] = 0

        # label encode the image array
        class_labels = []
        for i in range(num_bands):
            class_labels.append(image_array == i)

        # convert list to array
        land_class_array = np.array(class_labels) # (channels, width, height)
        land_class_array = np.moveaxis(land_class_array, 0, -1) # (width, height, channels)

        profile = src.profile

    # now need to save this array data in a destination TIF file  
    with rasterio.Env():

        profile.update(
            dtype=rasterio.uint8,
            nodata=0, # ensure the 'nodata' value is valid for uint8
            count=num_bands)

        with rasterio.open(path + os.path.basename(src_image), 'w', **profile) as dst:

            for i in range(num_bands):
                dst.write(land_class_array[:,:,i].astype(rasterio.uint8), i+1) 


def transform_time_series(data_csv, proc_path, interval, pixel_length):

    """
    Takes time series data, a start time, an end time, and an 
    interval period is a datetime string eg. '60S'
    Returns time series resampled to the interval period
    Wind speed is scaled to pixel resolution of arrival time map and interval time
    """

    #with rasterio.open(src_folder + arrival_time) as src:
    #    # calculate the size of a pixel in meters
    #    pixel_length = src.res[0]

    time_series = pd.read_csv(data_csv)
    time_series = time_series[24:] # trials last from 12:00 to 23:30 
    time_series.index = pd.to_datetime(time_series.DateTime) # make the index a datetime object (allows for resampling)

        
    time_series = time_series[[
                       'W_SPD',
                       'W_DIR',
                       'D_TMP',
                       'TMP',
                       'RH',
                       'DF',
                       'Curing']]
                      
    
    ##time_series = time_series.loc[date+start:date+end]

    interval_numeric = float(interval[:-1])
    # wind direction in degrees
    wind_dir = time_series['W_DIR'].resample(interval).interpolate(method='linear')
        # wind speed as pixels per interval
    wind_speed = time_series['W_SPD'].resample(interval).interpolate(method='pchip')\
                        * (1 / 1) * (interval_numeric / pixel_length)
    air_temp = time_series['TMP'].resample(interval).interpolate(method='pchip')
        # relative humidity as [0,1]
    rel_humid = time_series['RH'].resample(interval).interpolate(method='pchip')\
                        / 100
    dew_temp = time_series['D_TMP'].resample(interval).interpolate(method='pchip')
    drought_factor = time_series['DF'].resample(interval).interpolate(method='linear')
    curing = time_series['Curing'].resample(interval).interpolate(method='linear')
    
    
    # create new 'time_series' dataframe with interpolated data
    time_series = wind_dir.to_frame('WD')
    time_series['WS'] = wind_speed
    time_series['AT'] = air_temp
    time_series['RH'] = rel_humid
    time_series['DT'] = dew_temp
    time_series['CG'] = curing / 100 
    time_series['DF'] = drought_factor / 100

    # convert wind speed and direction to vector components
        # remember wind direction is the direction the wind is coming FROM
    time_series['VX'] = time_series['WS'] * np.cos(np.deg2rad(270-time_series['WD'])) 
    time_series['VY'] = -time_series['WS'] * np.sin(np.deg2rad(270-time_series['WD']))

    #return time_series
    #savepath = cwd+'/tmp/' + ntpath.basename(data_csv)

    time_series.to_csv(proc_path + os.path.basename(data_csv), header=True, index=True, index_label=False)
    
    return time_series
    
    
def transform_arrivaltime_map(at_raw, proc_path, interval, num_intervals):    
    """
    scales arrival time map by number of intervals
    write an output image where each channel corresponds to a single interval of propagation
    Each channel has values lying between 0 and 1.
    0: burnt at start of sim
    0-1: burnt during the interval
    1: not burnt during the interval
    """

    # scale arrival time map by number of intervals
    # then add a channel for each integer of arrival

    with rasterio.open(at_raw) as src:

        img = src.read(1) / interval
        profile = src.profile

        img = np.where(img <0, num_intervals, img)
        
        img_int = num_intervals - img

    # create arrival time chloropleths for each interval
    with rasterio.Env():

        profile.update(
            nodata=0.0,
            dtype='float32',
            count=num_intervals) # number of bands

        with rasterio.open(proc_path + os.path.basename(at_raw), 'w', **profile) as dst:

            area_final = []
            for band in range(num_intervals):


                ## old version below
                #img_int = 1 + band - img # 1 start value, 0 final value
                #img_int = np.clip(img_int, 0, 1) # clips values that are out of range
                
                img_clip = np.clip(img_int-num_intervals+band+1, 0, band+1)
                
                dst.write(img_clip, band+1)
                
                #area_final.append(sum(sum(img_int < 1))) # total burned pixels at end
                area_final.append(sum(sum(img_clip > 0)))
                
                
    # create arrival time burn masks (to train @ start of each interval)

    with rasterio.Env():
        
        profile.update(
            nodata=0.0,
            dtype='float32', 
            count=num_intervals) # number of bands
        
        with rasterio.open(proc_path + 'start_' + os.path.basename(at_raw), 'w', **profile) as dst:
            
            
            area_start = []
            for band in range(num_intervals):
                
                if band == 0:
                    # when band == 0, fire is at initial location, and has not 'moved'
                    img_clip = np.clip(img_int-num_intervals+band+1, 0, band+1)
                    img_clip = np.where(img_clip > 0.99, img_clip, 0) 
                
                    dst.write(img_clip, band+1)

                    #area_final.append(sum(sum(img_int < 1))) # total burned pixels at end
                    area_start.append(sum(sum(img_clip > 0)))
                    
                else:
                
                    img_clip = np.clip(img_int+band+0-num_intervals, 0, band+1)

                    dst.write(img_clip, band+1)

                    #area_start.append(sum(sum(img_int < 1))) # total burned pixels at start
                    area_start.append(sum(sum(img_clip > 0)))
                
    return area_start, area_final
    
       
    
    
# define the class mapping function
def land_to_fuel_class(x):
    """convert land class values to model classes"""
    x = np.floor(x / 10)
    if x == 0:
        return 0 # burn mask
    elif x > 60:
        return 0 # water
    elif x < 20 or x == 22 or x == 31 or x == 41:
        return 2 # forest
    elif x > 50:
        return 3 # urban
    elif x > 30 or x == 21:
        return 1 # grassland
    else:
        return 0 # default value
    
def class_mapping(x):
    
    # nan values replaced by 0 (water / unburnable)
    if x > 100:
        return 0
    else:
        return x
class_map = np.vectorize(class_mapping)  
        
                
def preprocess_data(raw_dir, proc_dir, heightmap_dir, landclass_dir, weather_dir, data_pkl):
    """
    Performs all data preprocessing
    raw_dir: the directory containing trial data. Each trial should be in a seperate subfolder
    proc_dir: the directory containing processed data. Each trial is in a corresponding subfolder
    data_csv: a csv file recoding informatino about each trial.
    """

    
    ## CREATE MASTER.CSV FILE TO TRACK DATA
    
    cwd = os.getcwd() # current directory
    if not(os.path.exists(cwd + '/' + data_pkl)):
        print('creating ' + data_pkl)

        columns = ['sample_name', 
                   'arrival_time_raw', 
                   'intensity_raw', 
                   'heightmap_raw', 
                   'landclass_raw', 
                   'burnmask', 
                   'weather_raw',
                   'stats', 
                   'arrival_time_proc', 
                   'intensity_prod',
                   'heightmap_proc', 
                   'landclass_proc', 
                   'weather_proc',
                   'sample_class', 
                   'num_intervals', 
                   'date', 'duration', 
                   'location', 
                   'area_start', 
                   'area_final', 
                   'resolution', 
                   'interval', 
                   'interval_num',
                   'max_height', 
                   'min_height', 
                   'max_temp', 
                   'min_temp', 
                   'max_wind', 
                   'min_wind', 
                   'max_RH', 
                   'min_RH', 
                   'wind', 
                   'wind_x',
                   #'wind_x_start',
                   #'wind_x_final',
                   'wind_y', 
                   #'wind_y_start',
                   #'wind_y_final',
                   'temp',
                   #'temp_start',
                   #'temp_final',
                   'RH',
                   #'RH_start',
                   #'RH_final',
                   'DF',
                   'CG',
                   'weather', # an array containing weather and climate values
                   'weather_format'] 

        
        
        dataframe = pd.DataFrame(columns=columns)

        dataframe.to_pickle(data_pkl)#, index=False) # create new csv
    else:
        dataframe = pd.read_pickle(data_pkl) # read csv
    
    
    ## APPEND NEW RAW TRIAL DATA TO MASTER.CSV
    
    #dataframe = pd.read_csv(data_csv)
    dataframe = pd.read_pickle(data_pkl)
    
    # loop over each subfolder - ie trial
    ##for trial in os.listdir(raw_dir):
    # loop over subfolders in the 'raw' folder
    # arrivaltime, landclass, and weather files are required for each trial
    # heightmaps are assumed flat if not supplied
    # burn masks are optional

    
    list_of_arrival_files=list()
    for (dirpath, dirnames, filenames) in os.walk(raw_dir):
    
        # get a list of just the arrival images
        list_of_arrival_files += [os.path.join(dirpath, file) for file in filenames if re.search('_Arrival', file)]
    
    # for each fire
    for trial in list_of_arrival_files:
    
        #cdir = raw_dir + trial

        # try to load in data

        try:
            #arrivaltime_file = os.path.basename(glob.glob(cdir+'/*arrival.tif*')[0])
            arrivaltime_file = trial
            
        except:
            print(os.path.basename(trial) + ' skipped. Missing or incompatible arrivaltime file.')
            continue # skip 

        try: 
            # use string replace to get an intensity filepath
            intensity_file = trial.replace('Arrival', 'Intensity')
        except:
            print(os.path.basename(trial) + ' skipped. Missing or incompatible intensity file.')
            continue # skip
            
            
        try:
            #landclass_file = os.path.basename(glob.glob(cdir+'/*land*.tif*')[0])
            landclass_file = glob.glob(landclass_dir + '/*Fuel*.tif*')[0]
        except:
            print(os.path.basename(trial) + ' skipped. Missing or incompatible landclass file.')
            continue # skip
            
        try:
            #weather_file = os.path.basename(glob.glob(cdir+'/*.csv')[0])
            
            # need to read weather from the weather file name from FireLocation MetaData
            trial_dir = os.path.dirname(arrivaltime_file) # get the directory from the full filepath
            
            metadata_file = glob.glob(trial_dir + '/' + '*Metadata*')[0]
            
            metadata = pd.read_csv(metadata_file) # read the metadata linking trial to weather file
            
            trial_id = os.path.basename(arrivaltime_file)
            
            ensemble_id = re.findall(r"\d+" ,re.findall(r"\d+.tif", trial_id)[0])[0] # get the digit
            
            # get the WeatherFilename from metadata which matches ensemble id
            
            weather_file = metadata.loc[metadata['EnsembleID'] == int(ensemble_id), 'WeatherFilename'].to_numpy()[0]
            #weather_file = 'dummy'
            
            
        except:
            print(os.path.basename(trial) + ' skipped. Missing or incompatible timeseries file.')
            continue # skip

        #try:
        #    #stats_file = os.path.basename(glob.glob(cdir+'/*.json')[0])
        #    stats_file = os.path.basename(glob.glob(
        #except:
        #    stats_file = None
        stats_file = None

        try:
            #heightmap_file = os.path.basename(glob.glob(cdir+'/*topog*.tif*')[0])
            heightmap_file = glob.glob(heightmap_dir + '/*DEM*.tif*')[0]           
        except:
            heightmap_file = None
            print('Heightmap Skipped')

        try:
            #burnmask_file = os.path.basename(glob.glob(cdir+'/*burn*.shp')[0])
            burnmask_file = glob.glob(landclass_dir + '/*burnmask*.shp')[0]
        except:
            burnmask_file = None


        sample_name = os.path.basename(trial)
        sample_name = os.path.splitext(sample_name)[0]
        sample_name = sample_name.replace('Arrival_', '')

        # check if entry already exists
        if sample_name not in list(dataframe.sample_name):
            # create new row
            
            d = {'sample_name':sample_name, 
                 'arrival_time_raw':arrivaltime_file, 
                 'intensity_raw':intensity_file,
                 'heightmap_raw':heightmap_file,
                 'landclass_raw':landclass_file, 
                 'burnmask':burnmask_file, 
                 'weather_raw':weather_file, 
                 'stats':stats_file}

            dataframe = dataframe.append(d, ignore_index=True)
            
            # create trial folder inside /proc/ folder
            try:
                os.mkdir(proc_dir + sample_name)
            except: #row already exists
                continue
       
    if True: # implement reprojection of entire heightmap, landclass map
        
        
        at_raw = dataframe.loc[0].arrival_time_raw # get an arrival image with crs
        
        # transform heightmap
        
        heightmap_list = dataframe.heightmap_raw.unique()
        
        proc_path = proc_dir + '/' # global resources
        
        for hm_raw in heightmap_list:
            

            # set crs and bounds to match arrival time image
            transform_crs(hm_raw, at_raw, proc_path, resampling='bilinear')

            # rescale_heights
            hm_proc = proc_path + os.path.basename(hm_raw)
            max_height, min_height = rescale_heights(hm_proc) # height rescaling unessesary when gradients used

            # update entry in dataframe
            dataframe.loc['heightmap_global_proc'] = hm_proc

            #dataframe.loc[ind, 'max_height'] = max_height
            #dataframe.loc[ind, 'min_height'] = min_height  
            
        landclass_list = dataframe.landclass_raw.unique()    
            
        for lc_raw in landclass_list:
            
            # set crs and bounds to match arrival time image
            transform_crs(lc_raw, at_raw, proc_path, resampling='nearest', no_data=254)    

            lc_proc = proc_path + os.path.basename(lc_raw)    

            # transform classes
            transform_classes(lc_proc, class_map, proc_path)    

            # update entry in dataframe
            dataframe.loc['landclass_global_proc'] = lc_proc
     
    print('finished global preprocessing')
    
    if True: # temporary gate to stop processing (when false)

        ## PROCESS DATA 
        print('processing trials')
        for ind, trial in zip(dataframe.index, dataframe.sample_name):

            # file paths
            #raw_path = raw_dir + trial + '/'
            proc_path = proc_dir + trial + '/'

            # file names
            at_raw = dataframe.loc[ind].arrival_time_raw
            hm_raw = dataframe.loc[ind].heightmap_raw
            lc_raw = dataframe.loc[ind].landclass_raw
            burn_mask = dataframe.loc[ind].burnmask
            ts_raw = dataframe.loc[ind].weather_raw
            stats_file = dataframe.loc[ind].stats

            at_proc = dataframe.loc[ind].arrival_time_proc
            hm_proc = dataframe.loc[ind].heightmap_proc
            lc_proc = dataframe.loc[ind].landclass_proc
            ts_proc = dataframe.loc[ind].weather_proc



            # temporarily open required files within the trial folder
            with rasterio.open(at_raw) as at_raw_image:

                # append resolution to dataframe
                resolution = at_raw_image.res[0]
                dataframe.loc[ind, 'resolution'] = resolution

                # determine temporal resolution from spatial resolution
                # 1 minute = 1 meter 
                # 30 min = 30 meters etc.
                interval_s = float(resolution*60)
                interval = str(int(resolution*60))+'S' # seconds
                dataframe.loc[ind, 'interval'] = interval



            # process heightmap data
            if hm_raw != None:
                
                if os.path.basename(hm_raw) != os.path.basename(str(hm_proc)):

                    # set crs and bounds to match arrival time image
                    transform_crs_old(hm_raw, at_raw, proc_path, resampling='bilinear')

                    # rescale_heights
                    hm_proc = proc_path + os.path.basename(hm_raw)
                    max_height, min_height = rescale_heights_old(hm_proc)
                    
                    # update entry in dataframe
                    dataframe.loc[ind, 'heightmap_proc'] = hm_proc

                    dataframe.loc[ind, 'max_height'] = max_height
                    dataframe.loc[ind, 'min_height'] = min_height
                    
                    # location is the location of the trial within the global heightmap
                    ##proc_path = proc_path = proc_dir + '/' # global resources
                    ##hm_proc = proc_path + os.path.basename(hm_raw)
                    hm_global = proc_dir + '/' + os.path.basename(hm_raw)
                    
                    #location = rowcolheightwidth_from_crspoint(hm_global, at_proc)
                    
                    #dataframe.loc[ind, 'hm_location'] = location
                
                #else:
                    #do nothing

            else:
                # create a dummy heightmap with all 0 elevation
                # use arrival time map as a template for bounds, size and resolution
                hm_proc = proc_path + 'zeroed_heightmap.tif'
                
                zeroed_heightmap(at_raw, hm_proc)

                # update entry in dataframe
                dataframe.loc[ind, 'heightmap_proc'] = hm_proc

                max_height, min_height = 0

                dataframe.loc[ind, 'max_height'] = max_height
                dataframe.loc[ind, 'min_height'] = min_height



            # process landclass data
            if os.path.basename(lc_raw) != os.path.basename(str(lc_proc)):
                # class mapping not required for SA data.
                ##class_map = np.vectorize(land_to_fuel_class)

                
                # set crs and bounds to match arrival time image
                transform_crs_old(lc_raw, at_raw, proc_path, resampling='nearest', no_data=254)    

                lc_proc = proc_path + os.path.basename(lc_raw)    

                # transform classes
                transform_classes(lc_proc, class_map, proc_path, mask=burn_mask)    

 
                # update entry in dataframe
                dataframe.loc[ind, 'landclass_proc'] = lc_proc
                
                  

            # process weather (timeseries) data
            if os.path.basename(ts_raw) != os.path.basename(str(ts_proc)):

                #data = json.load(json_file)

                #start_time = data['start_datetime']
                #end_time = data['end_datetime']
                      

                # set windspeed to pixels per interval, extract Vx, Vy, RH and Temp
                ##weather = transform_time_series(ts_raw, proc_path, interval, resolution)
                weather = transform_time_series(weather_dir + '/' + ts_raw, proc_path, interval, resolution) 
                
                start = weather.index[0]
                fire_date = start.date()
                end = weather.index[-1]
                duration = end-start
                
                #start = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
                #end   = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S')
                #duration = end-start

                # update entries in dataframe
                dataframe.loc[ind, 'weather_proc'] = ts_raw
                dataframe.loc[ind, 'date'] = fire_date

                dataframe.loc[ind, 'duration'] = duration
                num_intervals = int(np.round(duration.seconds / float(dataframe.loc[ind, 'interval'][:-1])))
                dataframe.loc[ind, 'num_intervals'] = num_intervals
                dataframe.loc[ind, 'interval_num'] = list(range(0, num_intervals))

                dataframe.loc[ind, 'max_wind'] = weather['WS'].max()
                dataframe.loc[ind, 'min_wind'] = 0 # zero the minimum wind speed
                dataframe.loc[ind, 'max_temp'] = weather['AT'].max()
                dataframe.loc[ind, 'min_temp'] = weather['AT'].min()
                dataframe.loc[ind, 'max_RH'] = weather['RH'].max()
                dataframe.loc[ind, 'min_RH'] = weather['RH'].min()
                
                # store complete windspeed, temperature and humidity values
                dataframe.loc[ind, 'wind'] = weather['WS'].values
                dataframe.loc[ind, 'wind_x'] = weather['VX'].values
                dataframe.loc[ind, 'wind_y'] = weather['VY'].values
                dataframe.loc[ind, 'temp'] = weather['AT'].values
                dataframe.loc[ind, 'RH'] = weather['RH'].values
                ##dataframe.loc[ind, 'DT'] = weather['DT'].values # DT not included, since it is a function of T and RH
                dataframe.loc[ind, 'DF'] = weather['DF'].values[0]  # drought factor (0 - 1)
                dataframe.loc[ind, 'CG'] = weather['CG'].values[0]  # curing (0 - 1) 
                
                wind_x_start = weather['VX'].values[:-1]
                wind_x_final = weather['VX'].values[1:]
                wind_y_start = weather['VY'].values[:-1]
                wind_y_final = weather['VY'].values[1:]
                temp_start = weather['AT'].values[:-1]
                temp_final = weather['AT'].values[1:]
                RH_start = weather['RH'].values[:-1]
                RH_final = weather['RH'].values[1:]
                DF = weather['DF'].values[1:] # truncate to equal intervals
                CG = weather['CG'].values[1:] # as above
             
                
                
                # create an array of weather values for a fire, then append to dataframe
                weather_array = np.stack((wind_x_start, wind_x_final, wind_y_start, wind_y_final, temp_start, temp_final, RH_start, RH_final, DF, CG))
                dataframe.loc[ind, 'weather'] = weather_array
                dataframe.loc[ind, 'weather_format'] = 'wind_x_start, wind_x_final, wind_y_start, wind_y_final, temp_start, temp_final, RH_start, RH_final, Drought Factor, Curing'
                              

            # process arrival time data
            if os.path.basename(at_raw) != os.path.basename(str(at_proc)):

                # apply transformations
                area_start, area_final = transform_arrivaltime_map(at_raw, proc_path, interval_s, num_intervals)


                at_proc = proc_path + os.path.basename(at_raw)   

                # update dataframe
                dataframe.loc[ind, 'arrival_time_proc'] = at_proc
                dataframe.loc[ind, 'area_start'] = area_start
                dataframe.loc[ind, 'area_final'] = area_final
            
        dataframe['g_max_wind'] = dataframe['max_wind'].max()
        dataframe['g_min_wind'] = dataframe['min_wind'].min()
        dataframe['g_max_temp'] = dataframe['max_temp'].max()
        dataframe['g_min_temp'] = dataframe['min_temp'].min()
        dataframe['g_max_RH'] = dataframe['max_RH'].max()
        dataframe['g_min_RH'] = dataframe['min_RH'].min()   
        dataframe['g_max_DF'] = dataframe['DF'].max()
        dataframe['g_min_DF'] = dataframe['DF'].min()
        dataframe['g_max_CG'] = dataframe['CG'].max()
        dataframe['g_min_CG'] = dataframe['CG'].min()
            
        # old approach
        if False:
            # ensure that values are in an array    
            dataframe['wind_x_start'] = np.array([entry[:-1] for entry in dataframe['wind_x']])
            dataframe['wind_x_final'] = np.array([entry[1:] for entry in dataframe['wind_x']])
            dataframe['wind_y_start'] = np.array([entry[:-1] for entry in dataframe['wind_y']])
            dataframe['wind_y_final'] = np.array([entry[1:] for entry in dataframe['wind_y']])
            dataframe['temp_start'] = np.array([entry[:-1] for entry in dataframe['temp']])
            dataframe['temp_final'] = np.array([entry[1:] for entry in dataframe['temp']])
            dataframe['RH_start'] = np.array([entry[:-1] for entry in dataframe['RH']])
            dataframe['RH_final'] = np.array([entry[1:] for entry in dataframe['RH']])
            
            
            
            
            
            
            # save updated dataframe after each trial is processed
            #dataframe.to_csv('master.csv', index=False)
        ##print('Processed ' + str(ind) + '/' + str(len(dataframe.index)))
    dataframe.to_pickle(data_pkl)
    print('all samples have been processed')
   
    return None



def create_trial_dataframe(master_dataframe, trials_pkl):
    
    """
    Takes the file location master_dataframe and returns a csv file with the name trials_pkl.
    This csv file contains a unique row for each time interval in each sample
    """
    
    # read in master.csv and convert string lists to object lists
    #dataframe = pd.read_csv(master_dataframe, converters={'area_final': eval, 'area_start': eval})
    dataframe = pd.read_pickle(master_dataframe)
    
    # create a row for each interval within a sample
    trial_data = dataframe.loc[dataframe.index.repeat(dataframe['num_intervals'])].reset_index(drop=True)

    # number each interval within each sample
    # each interval_num within a sample represents a single trial
    trial_data['interval_num'] = trial_data.groupby(['sample_name']).cumcount()
    
    trial_data['interval_num'] = [entry for entry in trial_data['interval_num']] # insert 'interval_num' into a list
    
    # get the correct areas based on interval
    trial_data['area_start'] = [trial_data.loc[trial, 'area_start'][trial_data.loc[trial].interval_num] for trial in trial_data.index]
    trial_data['area_final'] = [trial_data.loc[trial, 'area_final'][trial_data.loc[trial].interval_num] for trial in trial_data.index]
    
    # produce an array of weather values
    # subsumes curing and drought factor terms
    trial_data['weather'] = [trial_data.loc[trial, 'weather'][:, trial_data.loc[trial].interval_num] for trial in trial_data.index]
    
    
     # add curing term
    #trial_data['CG'] = [trial_data.loc[trial, 'CG'][trial_data.loc[trial].interval_num] for trial in trial_data.index]
    
    # add drought factor term
    #trial_data['DF'] = [trial_data.loc[trial, 'DF'][trial_data.loc[trial].interval_num] for trial in trial_data.index]
    
    
    if False:
        # modified approach 
        
        # add start and end windspeeds for each interval
        trial_data['wind_start'] = [trial_data.loc[trial, 'wind'][trial_data.loc[trial].interval_num] for trial in trial_data.index]
        trial_data['wind_final'] = [trial_data.loc[trial, 'wind'][trial_data.loc[trial].interval_num+1] for trial in trial_data.index]   
        trial_data.drop(['max_wind','min_wind','wind'], axis=1, inplace=True)
    
        trial_data['wind_x_start'] = [trial_data.loc[trial, 'wind_x'][trial_data.loc[trial].interval_num] for trial in trial_data.index]
        trial_data['wind_x_final'] = [trial_data.loc[trial, 'wind_x'][trial_data.loc[trial].interval_num+1] for trial in trial_data.index]
        trial_data['wind_y_start'] = [trial_data.loc[trial, 'wind_y'][trial_data.loc[trial].interval_num] for trial in trial_data.index] 
        trial_data['wind_y_final'] = [trial_data.loc[trial, 'wind_y'][trial_data.loc[trial].interval_num+1] for trial in trial_data.index] 
        trial_data.drop(['wind_x', 'wind_y'], axis=1, inplace=True)

        # add start and end temperatures for each interval
        trial_data['temp_start'] = [trial_data.loc[trial, 'temp'][trial_data.loc[trial].interval_num] for trial in trial_data.index]
        trial_data['temp_final'] = [trial_data.loc[trial, 'temp'][trial_data.loc[trial].interval_num+1] for trial in trial_data.index]   
        trial_data.drop(['max_temp','min_temp','temp'], axis=1, inplace=True)

        # add start and end relative humidities for each interval
        trial_data['RH_start'] = [trial_data.loc[trial, 'RH'][trial_data.loc[trial].interval_num] for trial in trial_data.index]
        trial_data['RH_final'] = [trial_data.loc[trial, 'RH'][trial_data.loc[trial].interval_num+1] for trial in trial_data.index]   
        trial_data.drop(['max_RH','min_RH','RH'], axis=1, inplace=True)
        

        # make sure that all values are in LIST format
        # this is to ensure formatting is compatible with master.pkl / dataframe
        trial_data['wind_x_start'] = [[value] for value in trial_data['wind_x_start']] 
        trial_data['wind_x_final'] = [[value] for value in trial_data['wind_x_final']]
        trial_data['wind_y_start'] = [[value] for value in trial_data['wind_y_start']]
        trial_data['wind_y_final'] = [[value] for value in trial_data['wind_y_final']]
        trial_data['temp_start'] = [[value] for value in trial_data['temp_start']]
        trial_data['temp_final'] = [[value] for value in trial_data['temp_final']]
        trial_data['RH_start'] = [[value] for value in trial_data['RH_start']]
        trial_data['RH_final'] = [[value] for value in trial_data['RH_final']]
        
        # make interval_num a list
        trial_data['interval_num'] = [[value] for value in trial_data['interval_num']]
    
    

      
    # save trial data
    trial_data.to_pickle(trials_pkl)
    
    print('all trials have been sorted')
    
def create_train_test_split(trials_pkl, test_size=0.2, random_state=42):
    """
    creates a 'train' column in the trials dataframe. 
    Using 'sample_class' for data stratification a train/test split is assigned.
    """
    
    trial_data = pd.read_pickle(trials_pkl)
    
    # unknnown sample classes are assigned -1 (for stratification of split)
    trial_data['sample_class'] = trial_data['sample_class'].fillna(-1)

    # calculate the train test split, stratify by sample class
    train_ind, test_ind = train_test_split(trial_data.index, test_size=0.2, random_state=random_state, stratify=trial_data.sample_class)
    
    # assign 1 where trial is used for training. 0 corresponds to the test set.
    trial_data.loc[test_ind, 'train_or_test'] = 'test'
    trial_data.loc[train_ind, 'train_or_test'] = 'train'
    
    # save the trial data
    trial_data.to_pickle(trials_pkl)
    print('train/test split assigned')
    

    
    
def point_in_triangle(A,B,C):
    """
    inputs: A, B, C, point arrays
    returns a point within the triangle ABC 
    """

    r1 = random.uniform(0,1)
    r2 = random.uniform(0,1)

    point = (1-math.sqrt(r1))*A + (math.sqrt(r1)*(1-r2))*B + (math.sqrt(r1)*r2)*C

    return point


def sample_points_in_polygon(polygon, N=1000):
    """
    Triangulates a polygon and samples points within it without rejection
    """

    # triangulate the polygon
    triangles = shapely.ops.triangulate(polygon)
    
    # remove triangles outside the original polygon
    triangles = [triangle for triangle in triangles if polygon.contains(triangle.centroid)]
    
    
    
    areas = [triangle.area for triangle in triangles]
    
    # sample triangles within polygon, weighted by area
    indices = np.random.choice(np.array(range(0, len(triangles))), size=N, replace=True, p= areas / np.sum(areas))

    # return a list of sampled triangle polygons
    sampled_triangles = [triangles[ind] for ind in indices]

    # return list of triangle vertices
    vertices = [np.reshape(np.asarray(sample.exterior.xy)[:,0:3], 6) for sample in sampled_triangles]

    # return set of randomly sampled points
    points = [point_in_triangle(pnts[[0,3]], pnts[[1,4]], pnts[[2,5]]) for pnts in vertices]

    return points 
    
    
    
def create_trial_sample_points(trials_pkl, N=1000):
    """
    Cycles through each arrival time image and finds N points that lie in the region that changes in the burn. This version simply looks for pixels with intermediate values. ie 0 < x < 1
    """
    
    trial_data = pd.read_pickle(trials_pkl)
    points_list = []
    
    for trial_num in trial_data.index: # loop over all samples
        
        at_path = trial_data.loc[trial_num].arrival_time_proc
        interval = trial_data.loc[trial_num].interval_num
        
        # check whether sample points already exist
        try:
            num_points_in_trial = len(trial_data['sample_points'][trial_num])
        except:
            num_points_in_trial = -1
            
        if num_points_in_trial == N:
            continue # skips over rest of this iteration
        
            
        
        with rasterio.open(at_path) as src:
            
            img = src.read(int(interval) + 1)
            
            delta = np.argwhere( (img > 0.01) & (img < 0.99) ) # get locations of pixels that are burnt during this interval
    
        # sample N points in delta 
    
        num_points = len(delta)
        if num_points > N:
            # choose without replacement
            subset = np.random.choice(num_points, N, replace=False)
        else:
            # choose with replacement
            subset = np.random.choice(num_points, N, replace=True)
            
        points_list.append(delta[subset,:]) # return a sample of changed points
        
        
    trial_data['sample_points'] = points_list

    # save the trial data
    trial_data.to_pickle(trials_pkl)                  
    print('sample points generated for all trials')

    
    
def create_trial_sample_points_backup(trials_pkl, N=1000):
    """
    Cycles through each arrival time image and finds N points that lie within the region that changes during the burn.
    This can be used to seed pseudo-random cropping of a sample during ML training such that the region will be intersting 
    (ie there will be at least some pixels that are burned over the course of the interval).
    """
    
    
    trial_data = pd.read_pickle(trials_pkl)
    
    points_list = []
    
    cwd = os.getcwd()
    proc_path = cwd+'/proc/'
    
    for trial_num in trial_data.index: # loop over all samples
        
        # open the samples arrival time image
        folder = trial_data.loc[trial_num]['sample_name'] + '/'
        at_path = proc_path+folder+trial_data.loc[trial_num]['arrival_time_proc']
        interval = trial_data.loc[trial_num]['interval_num']
        
        
        with rasterio.open(at_path) as src:

            img = src.read(int(interval)+1)

            initial_contour = measure.find_contours(img, .999) # prior burn perimeter
            final_contour = measure.find_contours(img, .001) # final burn perimeter
        
        
        # MultiPolygon(Polygon(initial_contour[i]) for i in range(len(inner_contour)))
        
        initial_poly = MultiPolygon(Polygon(initial_contour[i]) for i in range(len(initial_contour)))
        final_poly = MultiPolygon(Polygon(final_contour[i]) for i in range(len(final_contour)))
        
        #display(initial_poly)
        #display(final_poly)
        
        #initial_poly = initial_poly[0] # return just simple polygons?
        #final_poly = final_poly[0]
        
        
        # creates a polygon containing pixels burned during the samples interval
        change_in_area_polygon = final_poly.symmetric_difference(initial_poly)
        

        # sample points within this polygon
        points = sample_points_in_polygon(change_in_area_polygon.buffer(0), N)
        
        # save the list of points to the dataframe
        #trial_data.loc[trial_num]['sample_points'] = points
        
        points_list.append(points)
                       
            
    trial_data['sample_points'] = points_list
            
 
    # save the trial data
    trial_data.to_pickle(trials_pkl)                  
    print('sample points generated for all trials')