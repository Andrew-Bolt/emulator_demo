import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import random
import time
import sys
RNN = tf.keras.layers.RNN


import matplotlib.pyplot as plt

import os
cwd = os.getcwd()

import rasterio
    
IMAGE_WIDTH = None
IMAGE_HEIGHT = None
slices = None # number of time slices, None for unknown

discard_pad = 32
offset_fraction = 0.05 #amount of jitter to apply to crops


# define how to load in images
def load_image(image, band=None, pad=None, window=None, zeros_pad=False, crop=False):
    """
    used to load in tiff file images. Images are in the format (width, height, band). Bands can refer to feature layers, in the case of landclass maps, or bands can refer to intervals in the case of arrival maps. Training commonly uses a local subset of the image, so a window may be defined such that only the relevent portion of the image is loaded.
    """
    try:
        band = int(band)
    except:
        band = None
    
    
    with rasterio.open(image) as src:
        if crop:
            return_image = src.read(band, window=window)    
            
            if zeros_pad:
                return_image = np.pad(return_image, pad_width=pad, mode='constant', constant_values=0)
            else:

                
                return_image = np.pad(return_image, pad_width=pad, mode='reflect')
                
                
        else:
            return_image = src.read(band)

        return return_image                                       
    
 
# define how to load in a trial (batch)
def load_trial(trial_number, dataframe, window_size=128, crop=False, transform = [0,0,0,0]):
    """
    takes a trial number and the trial dataframe
    returns: weather, heightmap, landclass, arrival_int, arrival_final
    """
    
    padding = None
    padding_c = None
    window = None

    ## CALCULATE WHETHER TO TRANSFORM A SPECIFIC SAMPLE
    fliplr_fraction, flipud_fraction, transp_fraction, rotate_fraction = transform
    
    # fraction of samples to rotate from 0 to 45 degrees
    # rotate not implemented right now
    rotate = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) < rotate_fraction
    rotate = 0
    
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    
    # GET DATA LOCATION FOR A SAMPLE
    #folder = dataframe.loc[trial_number]['sample_name'] + '/'
    #proc_path = cwd+'/proc/'
    
    at_path = root + dataframe.loc[trial_number]['arrival_time_proc']
    ##at_path_init = os.path.dirname(dataframe.loc[trial_number]['arrival_time_proc']) + '/start_' +                     os.path.basename(dataframe.loc[trial_number]['arrival_time_proc'])
    hm_path = root + dataframe.loc[trial_number]['heightmap_proc']
    lc_path = root + dataframe.loc[trial_number]['landclass_proc']
    weather = dataframe.loc[trial_number, ['weather']][0]
    
    
    # NUMBER OF INTERVALS THE TRIAL IS RUN OVER
    # (for training this is a single interval so that targeted cropping is possible)
    interval_num = dataframe.loc[trial_number]['interval_num']
    

    
    ## LOAD IN DATA
    

    # extract points on fire perimeter for training
    sample_points = dataframe.loc[trial_number,'sample_points']
    n = np.random.randint(low=0, high=len(sample_points))
    point = sample_points[n]
    # apply jitter
    jitter = np.random.uniform(low=-offset_fraction*window_size, high=offset_fraction*window_size, size=(2))
    point = point + jitter



    if rotate:
        glimpse_size = int(window_size * np.math.sqrt(2)) # cut larger area to account for rotation
    else:
        glimpse_size = window_size

    # CREATE BOUNDING BOXES TO GLIMPSE IMAGE DATA
    left = int(point[1] - glimpse_size / 2)
    top = int(point[0] - glimpse_size / 2)

    x_left = left
    x_right = left + glimpse_size
    y_top = top
    y_bottom = top + glimpse_size
    # ensure values fall within image width, height limits.
    with rasterio.open(at_path) as src:
        height, width = src.shape

    # pad extends out the image in case the glimpse area exceeds its bounds
    left_pad, x_left     = max(0 - x_left, 0), max(x_left, 0)
    right_pad, x_right   = max(x_right - width, 0), min(x_right, width)
    top_pad, y_top       = max(0 - y_top, 0), max(y_top, 0)
    bottom_pad, y_bottom = max(y_bottom - height, 0), min(y_bottom, height)

    window = rasterio.windows.Window.from_slices((y_top, y_bottom), (x_left, x_right))
    padding = ((top_pad, bottom_pad), (left_pad, right_pad)) # padding for images without channels
    padding_c = ((0,0),(top_pad, bottom_pad), (left_pad, right_pad)) # padding for images with channels




    arrival_init = load_image(at_path, interval_num+1, pad=padding, window=window, zeros_pad=True, crop=True)
    heightmap = load_image(hm_path, 1, pad=padding, window=window, crop=True)
    landclass = load_image(lc_path, None, pad=padding_c, window=window, crop=True)
    arrival_final = load_image(at_path, interval_num+2, pad=padding, window=window, zeros_pad=True, crop=True)

    # make unburnt pixels have value -1
    ##arrival_init = tf.where(arrival_init <= 0.0, -1.0, arrival_init)
    ##arrival_final = tf.where(arrival_final <= 0.0, -1.0, arrival_final)
    
    
    
    
    
    # format landclass so each class falls in the last dimension
    landclass   = tf.transpose(landclass,perm=[1,2,0]) # move classes 
    
    if len(np.shape(arrival_init)) == 3: 
           # when there are more than one arrival channel
        arrival_init = tf.transpose(arrival_init,perm=[1,2,0])

    # Transformations to data will be applied in "transform_trial"
    # this is performed in batches, and is more efficient

    if rotate:
        # need to apply rotation transformation, followed by crop
        pass 


    # PREPROCESS DATA

    # expand dimensions of image data  (incorporate batching)
    #arrival_init = tf.expand_dims(arrival_init, 0)
    #heightmap = tf.expand_dims(heightmap, 0)
    #landclass = tf.expand_dims(landclass, 0)
    #landclass = np.moveaxis(landclass, 0, -1) # move channels to end
    #arrival_final = tf.expand_dims(arrival_final, 0)
    # expand dimensions of weather data (format for multiple intervals, ie timeseries)
    weather = np.expand_dims(weather, 1)

    # Truncate arrival maps
    #arrival_init = tf.where(arrival_init > 1.0, 1.0, arrival_init)
    #arrival_final = tf.where(arrival_final > 1.0, 1.0, arrival_final)
    arrival_init = tf.clip_by_value(arrival_init, 0, 1)
    arrival_final = tf.clip_by_value(arrival_final, 0, 1)
    

    # NORMALIZE WEATHER VALUES
    max_vals = dataframe.loc[0, ['g_max_wind', 'g_max_wind', 'g_max_wind', 'g_max_wind', 'g_max_temp', 'g_max_temp',\
'g_max_RH', 'g_max_RH', 'g_max_DF', 'g_max_CG']]
    max_vals = np.array(max_vals)
    min_vals = dataframe.loc[0, ['g_min_wind', 'g_min_wind', 'g_min_wind', 'g_min_wind', 'g_min_temp', 'g_min_temp', \
'g_min_RH', 'g_min_RH', 'g_min_DF', 'g_min_CG']]
    min_vals = np.array(min_vals)

    #weather = weather_scaled

    # unstack weather
    wind_x_init  = weather[0,:] / max_vals[0]
    wind_x_final = weather[1,:] / max_vals[1]
    wind_y_init  = weather[2,:] / max_vals[2]
    wind_y_final = weather[3,:] / max_vals[3]
    temp_init    = (weather[4,:] - min_vals[4]) / (max_vals[4] - min_vals[4])
    temp_final   = (weather[5,:] - min_vals[5]) / (max_vals[5] - min_vals[5])
    RH_init      = (weather[6,:] - min_vals[6]) / (max_vals[6] - min_vals[6])
    RH_final     = (weather[7,:] - min_vals[7]) / (max_vals[7] - min_vals[7])
    DF           = (weather[8,:] - min_vals[8]) / (max_vals[8] - min_vals[8])
    CG           = (weather[9,:] - min_vals[9]) / (max_vals[9] - min_vals[9])

    weather = np.stack((wind_x_init, wind_x_final, wind_y_init, wind_y_final, temp_init, temp_final, RH_init, RH_final))
    weather = np.transpose(weather, (1,0)) # tf Dense layers require fixed length in last dimension    
        
    climate = np.stack((DF, CG))    
    climate = climate[:,0] # just take the first values, all subsequent intervals have identical climate
    ##climate = np.transpose(climate, (1,0)) 
    #climate = tf.squeeze(climate, 1)


    # RETURN OUTPUT FEATURES
    # these features will be further transformed in the "transform_trials" function


    #heightmap = tf.expand_dims(heightmap, 0) # adding and removing dimensions as required to work with tf.
    #heightmap = tf.expand_dims(heightmap, -1)
    #heightmap = tf.image.sobel_edges(heightmap) ##/ 8 # 3x3 conv filter for gradients (normalized)
    #heightmap = tf.squeeze(heightmap, 0)
    #heightmap = tf.squeeze(heightmap, -2)

    # CONVERT OUTPUTS TO TENSORS


    arrival_init = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival_init),-1), tf.float32)
    arrival_final = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival_final),-1), tf.float32)
    heightmap   = tf.cast(tf.expand_dims(tf.convert_to_tensor(heightmap),-1), tf.float32)
    #landclass = tf.cast(tf.expand_dims(tf.convert_to_tensor(landclass),-1), tf.float32)


    weather = tf.cast(tf.convert_to_tensor(weather), tf.float32)
    climate = tf.cast(tf.convert_to_tensor(climate), tf.float32)
    transform = np.asarray(transform).astype('float32')
    transform = tf.cast(tf.convert_to_tensor(transform), tf.float32)

    # make binary
    ##arrival_init = tf.where(arrival_init > 0, 1.0, 0.0)
    ##arrival_final = tf.where(arrival_final > 0, 1.0, 0.0)
    
    
    features = (arrival_init, weather, climate, heightmap, landclass, transform)
    target = arrival_final

    #print(np.shape(arrival_init))
    #print(np.shape(weather))
    #print(np.shape(heightmap))
    #print(np.shape(landclass))
    #print(np.shape(transform))

    return features, target



def load_arrival(trial_number, dataframe, window_size=128, crop=False):
    
    # simply loads in and crops an arrival image
    # this is for training autoencoder component of the neural net
    
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    
    # GET DATA LOCATION FOR A SAMPLE
    #folder = dataframe.loc[trial_number]['sample_name'] + '/'
    #proc_path = cwd+'/proc/'
    
    interval_num = dataframe.loc[trial_number]['interval_num']
    
    at_path = root + dataframe.loc[trial_number]['arrival_time_proc']
    
     # extract points on fire perimeter for training
    sample_points = dataframe.loc[trial_number,'sample_points']
    n = np.random.randint(low=0, high=len(sample_points))
    point = sample_points[n]
    # apply jitter
    jitter = np.random.uniform(low=-offset_fraction*window_size, high=offset_fraction*window_size, size=(2))
    point = point + jitter

    glimpse_size = window_size

    # CREATE BOUNDING BOXES TO GLIMPSE IMAGE DATA
    left = int(point[1] - glimpse_size / 2)
    top = int(point[0] - glimpse_size / 2)

    x_left = left
    x_right = left + glimpse_size
    y_top = top
    y_bottom = top + glimpse_size
    # ensure values fall within image width, height limits.
    with rasterio.open(at_path) as src:
        height, width = src.shape

    # pad extends out the image in case the glimpse area exceeds its bounds
    left_pad, x_left     = max(0 - x_left, 0), max(x_left, 0)
    right_pad, x_right   = max(x_right - width, 0), min(x_right, width)
    top_pad, y_top       = max(0 - y_top, 0), max(y_top, 0)
    bottom_pad, y_bottom = max(y_bottom - height, 0), min(y_bottom, height)

    window = rasterio.windows.Window.from_slices((y_top, y_bottom), (x_left, x_right))
    padding = ((top_pad, bottom_pad), (left_pad, right_pad)) # padding for images without channels
    padding_c = ((0,0),(top_pad, bottom_pad), (left_pad, right_pad)) # padding for images with channels
    
    
    
   
    arrival = load_image(at_path, interval_num+1, pad=padding, window=window, zeros_pad=True, crop=True)
    
    arrival = tf.clip_by_value(arrival, 0, 1)
    #arrival = tf.where(arrival <= 0, -1, arrival) # make unburnt pixels have value of -1
    arrival = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival),-1), tf.float32)
    
    features = arrival
    target = arrival 
    
    
    return features, target


def load_arrival_backup(trial_number, dataframe, window_size=128, crop=False):
    
    # simply loads in and crops an arrival image
    # this is for training autoencoder component of the neural net
    
    # GET DATA LOCATION FOR A SAMPLE
    folder = dataframe.loc[trial_number]['sample_name'] + '/'
    proc_path = cwd+'/proc/'
    
    interval_num = dataframe.loc[trial_number]['interval_num']
    
    at_path = dataframe.loc[trial_number]['arrival_time_proc']
    
     # extract points on fire perimeter for training
    sample_points = dataframe.loc[trial_number,'sample_points']
    n = np.random.randint(low=0, high=len(sample_points))
    point = sample_points[n]
    # apply jitter
    jitter = np.random.uniform(low=-offset_fraction*window_size, high=offset_fraction*window_size, size=(2))
    point = point + jitter

    glimpse_size = window_size

    # CREATE BOUNDING BOXES TO GLIMPSE IMAGE DATA
    left = int(point[1] - glimpse_size / 2)
    top = int(point[0] - glimpse_size / 2)

    x_left = left
    x_right = left + glimpse_size
    y_top = top
    y_bottom = top + glimpse_size
    # ensure values fall within image width, height limits.
    with rasterio.open(at_path) as src:
        height, width = src.shape

    # pad extends out the image in case the glimpse area exceeds its bounds
    left_pad, x_left     = max(0 - x_left, 0), max(x_left, 0)
    right_pad, x_right   = max(x_right - width, 0), min(x_right, width)
    top_pad, y_top       = max(0 - y_top, 0), max(y_top, 0)
    bottom_pad, y_bottom = max(y_bottom - height, 0), min(y_bottom, height)

    window = rasterio.windows.Window.from_slices((y_top, y_bottom), (x_left, x_right))
    padding = ((top_pad, bottom_pad), (left_pad, right_pad)) # padding for images without channels
    padding_c = ((0,0),(top_pad, bottom_pad), (left_pad, right_pad)) # padding for images with channels
    
    
    
   
    arrival = load_image(at_path, interval_num+1, pad=padding, window=window, zeros_pad=True, crop=True)
    
    arrival = tf.clip_by_value(arrival, 0, 1)
    arrival = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival),-1), tf.float32)
    
    features = arrival
    target = arrival 
    
    
    return features, target

def load_lc(trial_number, dataframe, window_size=128, crop=False):
    
    # simply loads in and crops an arrival image
    # this is for training autoencoder component of the neural net
    
    # GET DATA LOCATION FOR A SAMPLE
    #folder = dataframe.loc[trial_number]['sample_name'] + '/'
    #proc_path = cwd+'/proc/'
    
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    
    interval_num = dataframe.loc[trial_number]['interval_num']
    
    at_path = root + dataframe.loc[trial_number]['arrival_time_proc']
    hm_path = root + dataframe.loc[trial_number]['heightmap_proc']
    lc_path = root + dataframe.loc[trial_number]['landclass_proc']
    
     # extract points on fire perimeter for training
    sample_points = dataframe.loc[trial_number,'sample_points']
    n = np.random.randint(low=0, high=len(sample_points))
    point = sample_points[n]
    # apply jitter
    jitter = np.random.uniform(low=-offset_fraction*window_size, high=offset_fraction*window_size, size=(2))
    point = point + jitter

    glimpse_size = window_size

    # CREATE BOUNDING BOXES TO GLIMPSE IMAGE DATA
    left = int(point[1] - glimpse_size / 2)
    top = int(point[0] - glimpse_size / 2)

    x_left = left
    x_right = left + glimpse_size
    y_top = top
    y_bottom = top + glimpse_size
    # ensure values fall within image width, height limits.
    with rasterio.open(at_path) as src:
        height, width = src.shape

    # pad extends out the image in case the glimpse area exceeds its bounds
    left_pad, x_left     = max(0 - x_left, 0), max(x_left, 0)
    right_pad, x_right   = max(x_right - width, 0), min(x_right, width)
    top_pad, y_top       = max(0 - y_top, 0), max(y_top, 0)
    bottom_pad, y_bottom = max(y_bottom - height, 0), min(y_bottom, height)

    window = rasterio.windows.Window.from_slices((y_top, y_bottom), (x_left, x_right))
    padding = ((top_pad, bottom_pad), (left_pad, right_pad)) # padding for images without channels
    padding_c = ((0,0),(top_pad, bottom_pad), (left_pad, right_pad)) # padding for images with channels
    
    
    landclass = load_image(lc_path, None, pad=padding_c, window=window, crop=True)
    
    # format landclass so each class falls in the last dimension
    landclass   = tf.transpose(landclass,perm=[1,2,0]) # move classes 
    landclass   = tf.cast(landclass, tf.float32) # convert to float type

    
    #topo = tf.concat([heightmap, landclass], axis=-1)
    
    features = landclass
    target = landclass
    
    
    return features, target

 
def load_start_stop_predict(trial_number, dataframe, transform=[0.0, 0.0, 0.0, 0.0], start=0, stop=-1):
    # transform is largely left as dummy values, so that "transform_trial" can be reused w/o modification
    # start and stop values determine the intervals over which the model predicts
    
    # GET DATA LOCATION FOR A SAMPLE
    #folder = dataframe.loc[trial_number]['sample_name'] + '/'
    #proc_path = cwd+'/proc/'
    
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    
    at_path = root+ dataframe.loc[trial_number]['arrival_time_proc']
    #at_path_init = os.path.dirname(dataframe.loc[trial_number]['arrival_time_proc']) + '/start_' +                     os.path.basename(dataframe.loc[trial_number]['arrival_time_proc'])
    hm_path = root+dataframe.loc[trial_number]['heightmap_proc']
    lc_path = root+dataframe.loc[trial_number]['landclass_proc']
    weather = dataframe.loc[trial_number, ['weather']][0]
    


    # extract all intervals in run
    interval_num = dataframe.loc[trial_number]['interval_num']
    
    if stop > interval_num[-1]:
        stop = interval_num[-1] # prevent execution exceeding length of simulated trail
    if stop < 0:
        stop = interval_num[-1] # allow -1 to represent longest value
    if start >= stop:
        sys.exit('Starting interval must be earlier than final interval')
        
    
    arrival_init = load_image(at_path, start+1) # starting  band
    ##arrival_init = tf.transpose(arrival_init,perm=[1,2,0])
    weather = weather[:, start:stop]
    
    heightmap = load_image(hm_path, 1)
    landclass = load_image(lc_path, None)
    arrival_final = load_image(at_path, stop+1) # stop band
    
    diff = np.float32(stop-start)

    arrival_init = tf.where(arrival_init > 1.0, 1.0, arrival_init)
    arrival_final = tf.where(arrival_final > diff, diff, arrival_final) # chloropleth starting at start
    
    # format landclass so each class falls in the last dimension
    landclass   = tf.transpose(landclass,perm=[1,2,0]) # move classes 
    
    
    # make unburnt pixels -1
    ##arrival_init = tf.where(arrival_init <= 0, -1.0, arrival_init)
    ##arrival_final = tf.where(arrival_final <= 0, -1.0, arrival_final)
    
    

    # expand weather # no need when weather isn't a single entry (ie. IS an array)
    #weather = np.expand_dims(weather, 1)
        
    # NORMALIZE WEATHER VALUES
    max_vals = dataframe.loc[0, ['g_max_wind', 'g_max_wind', 'g_max_wind', 'g_max_wind', 'g_max_temp', 'g_max_temp',\
'g_max_RH', 'g_max_RH', 'g_max_DF', 'g_max_CG']]
    max_vals = np.array(max_vals)
    min_vals = dataframe.loc[0, ['g_min_wind', 'g_min_wind', 'g_min_wind', 'g_min_wind', 'g_min_temp', 'g_min_temp', \
'g_min_RH', 'g_min_RH', 'g_min_DF', 'g_min_CG']]
    min_vals = np.array(min_vals)

    # unstack weather 
    wind_x_init  = weather[0,:] / max_vals[0]
    wind_x_final = weather[1,:] / max_vals[1]
    wind_y_init  = weather[2,:] / max_vals[2]
    wind_y_final = weather[3,:] / max_vals[3]
    temp_init    = (weather[4,:] - min_vals[4]) / (max_vals[4] - min_vals[4])
    temp_final   = (weather[5,:] - min_vals[5]) / (max_vals[5] - min_vals[5])
    RH_init      = (weather[6,:] - min_vals[6]) / (max_vals[6] - min_vals[6])
    RH_final     = (weather[7,:] - min_vals[7]) / (max_vals[7] - min_vals[7])
    DF           = (weather[8,:] - min_vals[8]) / (max_vals[8] - min_vals[8])
    CG           = (weather[9,:] - min_vals[9]) / (max_vals[9] - min_vals[9])

    weather = np.stack((wind_x_init, wind_x_final, wind_y_init, wind_y_final, temp_init, temp_final, RH_init, RH_final))
    weather = np.transpose(weather, (1,0)) # tf Dense layers require fixed length in last dimension    
        
    climate = np.stack((DF, CG))    
    climate = climate[:,0] # just take the first values, all subsequent intervals have identical climate
    ##climate = np.transpose(climate, (1,0)) 
    #climate = tf.squeeze(climate, 1)
        
    # CONVERT OUTPUTS TO TENSORS

    arrival_init = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival_init),-1), tf.float32)
    arrival_final = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival_final),-1), tf.float32)
    heightmap   = tf.cast(tf.expand_dims(tf.convert_to_tensor(heightmap),-1), tf.float32)
    #landclass = tf.cast(tf.expand_dims(tf.convert_to_tensor(landclass),-1), tf.float32)


    weather = tf.cast(tf.convert_to_tensor(weather), tf.float32)
    climate = tf.cast(tf.convert_to_tensor(climate), tf.float32)
    transform = np.asarray(transform).astype('float32')
    transform = tf.cast(tf.convert_to_tensor(transform), tf.float32)

    features = (arrival_init, weather, climate, heightmap, landclass, transform)
    target = arrival_final
    
    return features, target    
    
    
    
def load_predict(trial_number, dataframe, transform=[0.0, 0.0, 0.0, 0.0]):
    # transform is largely left as dummy values, so that "transform_trial" can be reused w/o modification
    
    # GET DATA LOCATION FOR A SAMPLE
    #folder = dataframe.loc[trial_number]['sample_name'] + '/'
    #proc_path = cwd+'/proc/'
    
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    
    at_path = root+dataframe.loc[trial_number]['arrival_time_proc']
    #at_path_init = os.path.dirname(dataframe.loc[trial_number]['arrival_time_proc']) + '/start_' +                     os.path.basename(dataframe.loc[trial_number]['arrival_time_proc'])
    hm_path = root+dataframe.loc[trial_number]['heightmap_proc']
    lc_path = root+dataframe.loc[trial_number]['landclass_proc']
    weather = dataframe.loc[trial_number, ['weather']][0]

    # extract all intervals in run
    interval_num = dataframe.loc[trial_number]['interval_num']
    
    arrival_init = load_image(at_path, None) # all bands
    arrival_init = tf.transpose(arrival_init,perm=[1,2,0])
    
    
    
    heightmap = load_image(hm_path, 1)
    landclass = load_image(lc_path, None)
    arrival_final = load_image(at_path, interval_num[-1] + 1) # final band

    arrival_init = tf.where(arrival_init > 1.0, 1.0, arrival_init)
    arrival_final = tf.where(arrival_final > 1.0, 1.0, arrival_final)
    
    # format landclass so each class falls in the last dimension
    landclass   = tf.transpose(landclass,perm=[1,2,0]) # move classes 
    
    
    # make unburnt pixels -1
    ##arrival_init = tf.where(arrival_init <= 0, -1.0, arrival_init)
    ##arrival_final = tf.where(arrival_final <= 0, -1.0, arrival_final)
    
    

    # expand weather # no need when weather isn't a single entry (ie. IS an array)
    #weather = np.expand_dims(weather, 1)
        
    # NORMALIZE WEATHER VALUES
    max_vals = dataframe.loc[0, ['g_max_wind', 'g_max_wind', 'g_max_wind', 'g_max_wind', 'g_max_temp', 'g_max_temp',\
'g_max_RH', 'g_max_RH', 'g_max_DF', 'g_max_CG']]
    max_vals = np.array(max_vals)
    min_vals = dataframe.loc[0, ['g_min_wind', 'g_min_wind', 'g_min_wind', 'g_min_wind', 'g_min_temp', 'g_min_temp', \
'g_min_RH', 'g_min_RH', 'g_min_DF', 'g_min_CG']]
    min_vals = np.array(min_vals)

    # unstack weather 
    wind_x_init  = weather[0,:] / max_vals[0]
    wind_x_final = weather[1,:] / max_vals[1]
    wind_y_init  = weather[2,:] / max_vals[2]
    wind_y_final = weather[3,:] / max_vals[3]
    temp_init    = (weather[4,:] - min_vals[4]) / (max_vals[4] - min_vals[4])
    temp_final   = (weather[5,:] - min_vals[5]) / (max_vals[5] - min_vals[5])
    RH_init      = (weather[6,:] - min_vals[6]) / (max_vals[6] - min_vals[6])
    RH_final     = (weather[7,:] - min_vals[7]) / (max_vals[7] - min_vals[7])
    DF           = (weather[8,:] - min_vals[8]) / (max_vals[8] - min_vals[8])
    CG           = (weather[9,:] - min_vals[9]) / (max_vals[9] - min_vals[9])

    weather = np.stack((wind_x_init, wind_x_final, wind_y_init, wind_y_final, temp_init, temp_final, RH_init, RH_final))
    weather = np.transpose(weather, (1,0)) # tf Dense layers require fixed length in last dimension    
        
    climate = np.stack((DF, CG))    
    climate = climate[:,0] # just take the first values, all subsequent intervals have identical climate
    ##climate = np.transpose(climate, (1,0)) 
    #climate = tf.squeeze(climate, 1)
        
    # CONVERT OUTPUTS TO TENSORS

    #arrival_init = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival_init),-1), tf.float32)
    arrival_final = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival_final),-1), tf.float32)
    heightmap   = tf.cast(tf.expand_dims(tf.convert_to_tensor(heightmap),-1), tf.float32)
    #landclass = tf.cast(tf.expand_dims(tf.convert_to_tensor(landclass),-1), tf.float32)


    weather = tf.cast(tf.convert_to_tensor(weather), tf.float32)
    climate = tf.cast(tf.convert_to_tensor(climate), tf.float32)
    transform = np.asarray(transform).astype('float32')
    transform = tf.cast(tf.convert_to_tensor(transform), tf.float32)

    features = (arrival_init, weather, climate, heightmap, landclass, transform)
    target = arrival_final
    
    return features, target
        
def load_half(trial_number, dataframe, transform=[0.0, 0.0, 0.0, 0.0]):
    # transform is largely left as dummy values, so that "transform_trial" can be reused w/o modification
    
    start_band = 12
    
    
    # GET DATA LOCATION FOR A SAMPLE
    #folder = dataframe.loc[trial_number]['sample_name'] + '/'
    #proc_path = cwd+'/proc/'
    
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    
    at_path = root+dataframe.loc[trial_number]['arrival_time_proc']
    #at_path_init = os.path.dirname(dataframe.loc[trial_number]['arrival_time_proc']) + '/start_' +                     os.path.basename(dataframe.loc[trial_number]['arrival_time_proc'])
    hm_path = root+dataframe.loc[trial_number]['heightmap_proc']
    lc_path = root+dataframe.loc[trial_number]['landclass_proc']
    weather = dataframe.loc[trial_number, ['weather']][0]
    
    weather = weather[:, 12:] # get a few slices from weather

    # extract all intervals in run
    interval_num = dataframe.loc[trial_number]['interval_num']
    
    arrival_init = load_image(at_path, start_band) # 11th band
    ##arrival_init = tf.transpose(arrival_init,perm=[1,2,0])
    arrival_init = tf.expand_dims(arrival_init, -1)
    
    
    heightmap = load_image(hm_path, 1)
    landclass = load_image(lc_path, None)
    arrival_final = load_image(at_path, interval_num[-1] + 1) # final band

    arrival_init = tf.where(arrival_init > 1.0, 1.0, arrival_init)
    arrival_final = tf.where(arrival_final > 1.0, 1.0, arrival_final)
    
    # format landclass so each class falls in the last dimension
    landclass   = tf.transpose(landclass,perm=[1,2,0]) # move classes 
    
    

    # expand weather # no need when weather isn't a single entry (ie. IS an array)
    #weather = np.expand_dims(weather, 1)
        
    # NORMALIZE WEATHER VALUES
    max_vals = dataframe.loc[0, ['g_max_wind', 'g_max_wind', 'g_max_wind', 'g_max_wind', 'g_max_temp', 'g_max_temp',\
'g_max_RH', 'g_max_RH', 'g_max_DF', 'g_max_CG']]
    max_vals = np.array(max_vals)
    min_vals = dataframe.loc[0, ['g_min_wind', 'g_min_wind', 'g_min_wind', 'g_min_wind', 'g_min_temp', 'g_min_temp', \
'g_min_RH', 'g_min_RH', 'g_min_DF', 'g_min_CG']]
    min_vals = np.array(min_vals)

    # unstack weather 
    wind_x_init  = weather[0,:] / max_vals[0]
    wind_x_final = weather[1,:] / max_vals[1]
    wind_y_init  = weather[2,:] / max_vals[2]
    wind_y_final = weather[3,:] / max_vals[3]
    temp_init    = (weather[4,:] - min_vals[4]) / (max_vals[4] - min_vals[4])
    temp_final   = (weather[5,:] - min_vals[5]) / (max_vals[5] - min_vals[5])
    RH_init      = (weather[6,:] - min_vals[6]) / (max_vals[6] - min_vals[6])
    RH_final     = (weather[7,:] - min_vals[7]) / (max_vals[7] - min_vals[7])
    DF           = (weather[8,:] - min_vals[8]) / (max_vals[8] - min_vals[8])
    CG           = (weather[9,:] - min_vals[9]) / (max_vals[9] - min_vals[9])

    weather = np.stack((wind_x_init, wind_x_final, wind_y_init, wind_y_final, temp_init, temp_final, RH_init, RH_final))
    weather = np.transpose(weather, (1,0)) # tf Dense layers require fixed length in last dimension    
        
    climate = np.stack((DF, CG))    
    climate = climate[:,0] # just take the first values, all subsequent intervals have identical climate
    ##climate = np.transpose(climate, (1,0)) 
    #climate = tf.squeeze(climate, 1)
        
    # CONVERT OUTPUTS TO TENSORS

    #arrival_init = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival_init),-1), tf.float32)
    arrival_final = tf.cast(tf.expand_dims(tf.convert_to_tensor(arrival_final),-1), tf.float32)
    heightmap   = tf.cast(tf.expand_dims(tf.convert_to_tensor(heightmap),-1), tf.float32)
    #landclass = tf.cast(tf.expand_dims(tf.convert_to_tensor(landclass),-1), tf.float32)


    weather = tf.cast(tf.convert_to_tensor(weather), tf.float32)
    climate = tf.cast(tf.convert_to_tensor(climate), tf.float32)
    transform = np.asarray(transform).astype('float32')
    transform = tf.cast(tf.convert_to_tensor(transform), tf.float32)

    features = (arrival_init, weather, climate, heightmap, landclass, transform)
    target = arrival_final
    
    return features, target

 
@tf.function
def fraction_under_threshold(v1, v2, threshold):
    if v1 / v2 < threshold:
        return True
    else: 
        return False

    
def transform_identity(features, target):
    
    return features, target
    
@tf.function
def transform_trial(features, target):
    """
    takes the dataset of the form: 
        features, target // (arrival_0, weather, heightmap, landclass), arrival_f
    and returns data of the same shape st. random flips are applied.
    random_set is a set of N random numbers (0-1) which are used to apply random 
    processes to transformations and crops
    """

        
    # EXTRACT DATA
    
    arrival, weather, climate, heightmap, landclass, transform = features
    
    # TRANSFORMATION PARAMETERS
    fliplr = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) < transform[0]

    # fraction of samples to flipud
    flipud = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) < transform[1]

    # fraction of samples to transpose
    transp = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) < transform[2]
  
    # fraction of samples to rotate 
    rotate = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) < transform[3]
    
    #wind_x_init, wind_x_final, wind_y_init, wind_y_final, temp_init, temp_final, RH_init, RH_final, DF, CG = weather
    
    wind_x_init = weather[:,0:1]
    wind_x_final = weather[:,1:2]
    wind_y_init = weather[:,2:3]
    wind_y_final = weather[:,3:4]
    weather_rest = weather[:,4:]
    
    ##wind_x_init = weather[0:1,:]
    ##wind_x_final = weather[1:2,:]
    ##wind_y_init = weather[2:3,:]
    ##wind_y_final = weather[3:4,:]
    ##weather_rest = weather[4:,:]
    #temp_init = weather[4:5,:]
    #temp_final = weather[5:6,:]
    #RH_init = weather[6:7,:]
    #RH_final = weather[7:8,:]
    #DF = weather[8:9,:]
    #CG = weather[9:10,:]
    
    
    # convert heightmaps to gradients
    heightmap = tf.expand_dims(heightmap, 0)
    heightmap = tf.image.sobel_edges(heightmap)
    heightmap = tf.squeeze(heightmap, -2) # sobel adds gradients to last channel
    heightmap = tf.squeeze(heightmap, 0)
    
    ## APPLY TRANSFORMATIONS
 

    # Transform spatial data
    
    #image = (bs, h, w, c)
    
    if fliplr:
        # apply left/right flip
        arrival   = tf.reverse(arrival, axis=[-2])
        target    = tf.reverse(target, axis=[-2])
        heightmap = tf.reverse(heightmap, axis=[-2])
        landclass = tf.reverse(landclass, axis=[-2])

        # reverse the X wind component
        wind_x_init = -1.0 * wind_x_init
        wind_x_final = -1.0 * wind_x_final
        
    
    if flipud:
        # apply up/down flip
        arrival   = tf.reverse(arrival, axis=[-3])
        target    = tf.reverse(target, axis=[-3])
        heightmap = tf.reverse(heightmap, axis=[-3])
        landclass = tf.reverse(landclass, axis=[-3])

        # reverse the Y wind component
        wind_y_init = -1.0 * wind_y_init
        wind_y_final = -1.0 * wind_y_final
        

    if transp:
        # apply x-y transposition    
        ##perm = [0, 2, 1, 3]
        ##perm_c = [0, 1, 3, 2, 4]
        
        perm = [0, 2, 1, 3]
        perm = [1,0,2] # (h, w, c)
        #perm_c = [0, 2, 1, 3]
        
        # not image transpose has y axis positive going down, so
        # x -> -y
        # y -> -x  #rather than a straight swap
        
        arrival   = tf.transpose(arrival,perm=perm)
        target    = tf.transpose(target,perm=perm)
        heightmap = tf.transpose(heightmap,perm=perm)
        landclass = tf.transpose(landclass,perm=perm)


        ##swap X and Y wind components
        ##moved below without need for copying
        #x0 = wind_x_init
        #y0 = wind_y_init
        #x1 = wind_x_final
        #y1 = wind_y_final
        #wind_x_init, wind_x_final = y0, y1
        #wind_y_init, wind_y_final = x0, x1

    
    # stack weather
    if transp:
        weather = tf.concat((-1.0*wind_y_init, -1.0*wind_y_final, -1.0*wind_x_init, -1.0*wind_x_final, weather_rest), 1)
    else:
        weather = tf.concat((wind_x_init, wind_x_final, wind_y_init, wind_y_final, weather_rest), 1)
    
    # return transformed images
    features = (arrival, weather, climate, heightmap, landclass)

    return features, target




def transform_test(features, target):
    """
    returns an untransformed dataset
    """
    arrival, wind_x, wind_y, other_weather, heightmap, landclass, ones_pad = features
   

    # extract winds
    #wind_0_xy = wind_patch_to_vector(wind_0)
    #wind_f_xy = wind_patch_to_vector(wind_f)
     
    wind_x = tf.expand_dims(tf.expand_dims(wind_x, 0), 0)
    wind_y = tf.expand_dims(tf.expand_dims(wind_y, 0), 0)
    
    weather = tf.concat([wind_x, wind_y, other_weather], axis=-1)
    
    ##wide_weather = tf.math.multiply(weather, ones_pad)
    wide_weather = weather
    
    # return transformed images
    features = (arrival, wide_weather, heightmap, landclass)
    
    # return transformed images
    features = (arrival, wide_weather, heightmap, landclass)

    return features, target


