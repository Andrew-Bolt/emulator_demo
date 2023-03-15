# DATA LOADING

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import random
import time
RNN = tf.keras.layers.RNN


import matplotlib.pyplot as plt

import os
cwd = os.getcwd()

import rasterio
    
IMAGE_WIDTH = None
IMAGE_HEIGHT = None
slices = None # number of time slices, None for unknown

discard_pad = 32


# import the model
from resnet_models import resnet_model_3 as resnet_model
##from resnet_models import downup_model_1 as downup_model

model = resnet_model
#model = downup_model


# define how to load in images
def load_image(image, band=None, pad=None, window=None, zeros_pad=False, crop=False):
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
def load_trial(trial_number, dataframe, variable_scales, window_size=128, crop=False, rotate_fraction=0):
    """
    takes a trial number and the trial dataframe
    returns: weather, heightmap, landclass, arrival_int, arrival_final
    """

    # get location of data for a trial 
    folder = dataframe.loc[trial_number]['sample_name'] + '/'
    proc_path = cwd+'/proc/'
    
    at_path = dataframe.loc[trial_number]['arrival_time_proc']
    at_path_init = os.path.dirname(dataframe.loc[trial_number]['arrival_time_proc']) + '/start_' +                     os.path.basename(dataframe.loc[trial_number]['arrival_time_proc'])
    hm_path = dataframe.loc[trial_number]['heightmap_proc']
    lc_path = dataframe.loc[trial_number]['landclass_proc']
    
    # interval number
    interval_num = dataframe.loc[trial_number]['interval_num']
    
    # extract points on perimeter for training
    sample_points = dataframe.loc[trial_number,'sample_points']
    sample_points = np.array(sample_points)
    
    
    padding = None
    padding_c = None
    window = None
    
    
    # rotate some examples
    # fraction of samples to rotate from 22.5 to 45 degrees
    rotate = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32) < rotate_fraction
    #rotate = False # 28/2/21
 
    if crop:

        # crop a section of window_size * window_size around the perimeter of the fire
        points = dataframe.loc[0, 'sample_points']
        n = tf.random.uniform([1], minval=0, maxval=len(points), dtype=tf.int32)[0]
        
        if rotate:
            glimpse_size = int(window_size * np.math.sqrt(2)) 
        else:
            glimpse_size = window_size
        
        
        
        # add an offset to the window
        offset_fraction = 0.1
        x_jitter = tf.random.uniform([1], minval=-offset_fraction*glimpse_size, maxval=offset_fraction*glimpse_size, dtype=tf.float64)[0]
        y_jitter = tf.random.uniform([1], minval=-offset_fraction*glimpse_size, maxval=offset_fraction*glimpse_size, dtype=tf.float64)[0]



        # create bounding box
        # loads in only the required subarea of the images
        x_coord = int(sample_points[n][1] + x_jitter - window_size/2) # jitter by original window size
        y_coord = int(sample_points[n][0] + y_jitter - window_size/2)

        x_left  =  x_coord# - window_size/2
        x_right =  x_coord + glimpse_size #-1
        y_top   =  y_coord# - window_size/2
        y_bottom=  y_coord + glimpse_size #-1

        # calculate image bounds
        with rasterio.open(at_path_init) as src:
            height, width = src.shape

        # get image load coordinates and padding amounts
        left_pad, x_left     = max(0 - x_left, 0), max(x_left, 0)
        right_pad, x_right   = max(x_right - width, 0), min(x_right, width)
        top_pad, y_top       = max(0 - y_top, 0), max(y_top, 0)
        bottom_pad, y_bottom = max(y_bottom - height, 0), min(y_bottom, height)

        window = rasterio.windows.Window.from_slices((y_top, y_bottom), (x_left, x_right))
        padding = ((top_pad, bottom_pad), (left_pad, right_pad))
        padding_c = ((0,0), (top_pad, bottom_pad), (left_pad, right_pad))


        ## no need for below. If rotating simply extract a larger patch
        
        # load in the initial arrival time image
        # grab larger window (will be rotated into smaller bounds)
        if False:
            # if rotating update window size and padding
            rot_window_size = int(window_size * np.math.sqrt(2))

            x_coord = int(sample_points[n][1] + x_jitter - rot_window_size/2)
            y_coord = int(sample_points[n][0] + y_jitter - rot_window_size/2)

            x_left  =  x_coord# - window_size/2
            x_right =  x_coord + rot_window_size #-1
            y_top   =  y_coord# - window_size/2
            y_bottom=  y_coord + rot_window_size #-1    

            # get image load coordinates and padding amounts
            left_pad, x_left     = max(0 - x_left, 0), max(x_left, 0)
            right_pad, x_right   = max(x_right - width, 0), min(x_right, width)
            top_pad, y_top       = max(0 - y_top, 0), max(y_top, 0)
            bottom_pad, y_bottom = max(y_bottom - height, 0), min(y_bottom, height)

            window = rasterio.windows.Window.from_slices((y_top, y_bottom), (x_left, x_right))
            padding = ((top_pad, bottom_pad), (left_pad, right_pad))
            padding_c = ((0,0), (top_pad, bottom_pad), (left_pad, right_pad))

                  
        # load in image data within bounds
        arrival_init = load_image(at_path_init, interval_num+1, pad=padding, window=window, zeros_pad=True,crop=True)
        heightmap = load_image(hm_path, 1, pad=padding, window=window, crop=True)
        landclass = load_image(lc_path, None, pad=padding_c, window=window, crop=True)
        arrival_final = load_image(at_path, interval_num+1, pad=padding, window=window, zeros_pad=True, crop=True)
                
                
    else: # if not cropping
        arrival_init = load_image(at_path_init, interval_num+1, pad=padding, zeros_pad=True)
        heightmap = load_image(hm_path, 1, pad=padding, window=window)
        landclass = load_image(lc_path, None, pad=padding_c, window=window)
        arrival_final = load_image(at_path, interval_num+1, pad=padding, window=window, zeros_pad=True)
    
    
    ## OLD WEATHER PROCESSING
    if False:
        weather = dataframe.loc[[trial_number],[ 'wind_x_start', 'wind_x_final', 'wind_y_start', 'wind_y_final', 'temp_start', 'temp_final', 'RH_start', 'RH_final', 'CG', 'CG', 'DF', 'DF']]

        # extract weather components
        weather = np.atleast_2d(weather.to_numpy())#.flatten()
        # get the number of weather intervals and channels
        n_intervals, n_channels = weather.shape
        weather = weather[0]
        wind_x = weather[0:1+1] / (variable_scales['wind_max']) #- variable_scales['wind_min'])
        wind_y = - weather[2:3+1] / (variable_scales['wind_max']) #- variable_scales['wind_min'])
        temp = (weather[4:5+1] - variable_scales['temp_min'])/ (variable_scales['temp_max'] - variable_scales['temp_min'])
        other_weather = np.concatenate((temp, weather[6:]), axis=0) # relative humidity, curing, drought factor
    
    ## NEW WEATHER PROCESSING
    climate = dataframe.loc[[trial_number],['CG', 'DF']] # curing and drought factor 
    
    wind_x_start = dataframe.loc[[trial_number],['wind_x_start']] /(variable_scales['wind_max'])
    wind_x_final = dataframe.loc[[trial_number],['wind_x_final']] /(variable_scales['wind_max'])
    wind_y_start = dataframe.loc[[trial_number],['wind_y_start']] /(variable_scales['wind_max'])
    wind_y_final = dataframe.loc[[trial_number],['wind_y_final']] /(variable_scales['wind_max'])    
    temp_start   = dataframe.loc[[trial_number],['temp_start']] / (variable_scales['temp_max'] - variable_scales['temp_min'])
    temp_final   = dataframe.loc[[trial_number],['temp_final']] / (variable_scales['temp_max'] - variable_scales['temp_min'])
    rhumid_start = dataframe.loc[[trial_number],['RH_start']]
    rhumid_final = dataframe.loc[[trial_number],['RH_final']]
    
    
    

    # convert to appropriate tensorflow objects
    heightmap = tf.convert_to_tensor(heightmap)
    heightmap = tf.expand_dims(heightmap,-1)

    heightmap = tf.cast(heightmap, tf.float32)
    
    # convert heightmap to gradients, and tensor object
    
    heightmap = tf.expand_dims(heightmap, 0)
    heightmap = tf.image.sobel_edges(heightmap) ##/ 8 # 3x3 conv filter for gradients (normalized)
    heightmap = tf.squeeze(heightmap, 0)
    heightmap = tf.squeeze(heightmap, -2) #* 100 # gradient has value of rise/run (ie 45 deg = 1)
    
    # convert to tensor object
    
    landclass = np.moveaxis(landclass, 0, -1) # move channels to end
    landclass = tf.convert_to_tensor(landclass)
    landclass = tf.cast(landclass, tf.float32)
     
    arrival_init = tf.convert_to_tensor(arrival_init)
    arrival_init = tf.expand_dims(arrival_init, -1)
    arrival_init = tf.cast(arrival_init, tf.float32)
    
    arrival_final = tf.convert_to_tensor(arrival_final)
    arrival_final = tf.expand_dims(arrival_final, -1)
    arrival_final = tf.cast(arrival_final, tf.float32)

    
    # apply rotations 
    if rotate:
        # rotation angle 
        # rotate from 22.5 to 45 degrees
        angle = tf.random.uniform([1], minval=tf.constant(np.pi / 8), maxval=tf.constant(np.pi / 4), dtype=tf.float32)[0]
        
        # fix rotation angle to zero and see if this solves the problem???
        ##ngle = 0
        
        # rotate images
        arrival_init  = tfa.image.rotate(arrival_init, angles=angle, interpolation='NEAREST')
        heightmap     = tfa.image.rotate(heightmap, angles=angle, interpolation='BILINEAR')
        landclass     = tfa.image.rotate(landclass, angles=angle, interpolation='BILINEAR')
        arrival_final = tfa.image.rotate(arrival_final, angles=angle, interpolation='NEAREST')
        
        # centrally crop to window size 
            ## DO NOT USE RESIZE_WITH_CROP_OR_PAD
            
        rot_window_size = int(window_size * np.math.sqrt(2))
        border = int((rot_window_size - window_size) / 2)
              
        arrival_init = tf.expand_dims(arrival_init, 0)
        heightmap = tf.expand_dims(heightmap, 0)
        landclass = tf.expand_dims(landclass, 0)
        arrival_final = tf.expand_dims(arrival_final, 0)
            
        arrival_init  = tf.image.extract_glimpse(arrival_init, size=(window_size, window_size), offsets=[[border, border]], normalized=False, centered=False)
        heightmap     = tf.image.extract_glimpse(heightmap, size=(window_size, window_size), offsets=[[border, border]], normalized=False, centered=False)
        landclass     = tf.image.extract_glimpse(landclass, size=(window_size, window_size), offsets=[[border, border]], normalized=False, centered=False)
        arrival_final = tf.image.extract_glimpse(arrival_final, size=(window_size, window_size), offsets=[[border, border]], normalized=False, centered=False) 
            
        arrival_init = tf.squeeze(arrival_init, 0)
        heightmap = tf.squeeze(heightmap, 0)
        landclass = tf.squeeze(landclass, 0)
        arrival_final = tf.squeeze(arrival_final, 0)    
            
        # OLD WEATHER PROCESSING
        # rotate winds 
        if False:
            wind_x_ = wind_x * np.cos(angle) - wind_y * np.sin(angle)
            wind_y_ = wind_x * np.sin(angle) + wind_y * np.cos(angle)
            wind_x, wind_y = wind_x_, wind_y_
        
        # NEW WEATHER PROCESSING
        wind_x_start_ = wind_x_start * np.cos(angle) - wind_y_start * np.sin(angle)
        wind_x_final_ = wind_x_final * np.cos(angle) - wind_y_final * np.sin(angle)
        wind_y_start_ = wind_x_start * np.sin(angle) - wind_y_start * np.cos(angle)
        wind_y_final_ = wind_x_final * np.sin(angle) - wind_y_final * np.cos(angle)
        wind_x_start, wind_x_final, wind_y_start, wind_y_final = wind_x_start_, wind_x_final_, wind_y_start_, wind_y_final_
       
    # OLD WEATHER PROCESSING
    if False:
        # get the size of the sample (used to pad out weather)
        height, width, channels = arrival_init.shape

        # pad out the weather
        ones_pad = tf.ones((height, width, 1), dtype=tf.float32)

        other_weather = tf.expand_dims(other_weather,-2) # width
        other_weather = tf.expand_dims(other_weather, -2)# height

        other_weather = tf.cast(other_weather, tf.float32) * tf.ones((1,1,8), tf.float32)
        
    # NEW WEATHER PROCESSING
    
    weather = [wind_x_start, wind_x_final, wind_y_start, wind_y_final, temp_start, temp_final, rhumid_start, rhumid_final]
        
    
    features = (arrival_init, weather, climate, heightmap, landclass)
    
    return features, arrival_final



# Convert winds into gradient images
def winds_to_gradient_image(wind_x, wind_y, width, height):
    """
    wind_x and wind_y are lists of wind values.
    height, width are the dimensions of the target window.
    """

    #intervals = len(wind_x) # number of timesteps
    
    x_component = np.tile(np.linspace(start=-wind_x, stop=wind_x, num = width), (height, 1))
    y_component = np.tile(np.linspace(start=-wind_y, stop=wind_y, num = height), (width, 1)).T
        
    array = x_component + y_component
    array = tf.expand_dims(tf.cast(array, tf.float32), -1)
    
    #for j in range(intervals):
    #    # 0 centred s.t. the middle of the image will have a pixel value of zero
    #    x_component = np.tile(np.linspace(start=-wind_x[j]*width/1000, stop=wind_x[j]*width/1000, num = width), (height, 1))
    #    y_component = np.tile(np.linspace(start=-wind_x[j]*height/1000, stop=wind_y[j]*height/1000, num = height), (width, 1)).T
    #    array[j,:,:] = x_component+y_component
    #array = tf.expand_dims(array, -1) 
    #array = tf.cast(array, tf.float32)    

    return array

# convert gradient winds to components
def wind_patch_to_vector(gradient_image):
    # returns a (2,) array of NS/EW wind components

    ns_comp = tf.cast(np.array([[1, 1], [-1, -1]]), tf.float32)
    ew_comp = tf.cast(np.array([[1, -1], [1, -1]]), tf.float32)

    w_x = tf.expand_dims(tf.math.reduce_mean(gradient_image * tf.expand_dims(ns_comp, axis=-1)),-1)
    w_y = tf.expand_dims(tf.math.reduce_mean(gradient_image * tf.expand_dims(ew_comp, axis=-1)),-1)

    w_x = tf.expand_dims(tf.expand_dims(w_x, 0), 0)
    w_y = tf.expand_dims(tf.expand_dims(w_y, 0), 0)
    
    wind = tf.keras.layers.concatenate([w_x, w_y], axis=-1)

    return wind






@tf.function
def fraction_under_threshold(v1, v2, threshold):
    if v1 / v2 < threshold:
        return True
    else: 
        return False

    
    
    
@tf.function
def transform_trial(features, target):
    """
    takes the dataset of the form: 
        features, target // (arrival_0, weather, heightmap, landclass), arrival_f
    and returns data of the same shape st. random flips are applied.
    random_set is a set of N random numbers (0-1) which are used to apply random 
    processes to transformations and crops
    """
    # CONSTANTS 

    
    # RANDOM OFFSETS
    
    # crop window number
    # elevation offset
    elevation_jitter = tf.random.uniform([1], minval=-0.2, maxval=0.2, dtype=tf.float32)[0]
    
    
    # TRANSFORMATION PARAMETERS
    bias = 0.5
    
    flipud = tf.random.uniform(shape=[1]) > bias
    fliplr = tf.random.uniform(shape=[1]) > bias
    transp = tf.random.uniform(shape=[1]) > 9# bias
        
    # EXTRACT DATA
    
    arrival, weather, climate, heightmap, landclass = features

    
    
    ## APPLY TRANSFORMATIONS
    

    # randomly adjust heightmap values +/- 0.2
    heightmap = heightmap + elevation_jitter
    

    # Transform spatial data
    
    if flipud:
        # apply up/down flip
        arrival   = tf.reverse(arrival, axis=[-3])
        target    = tf.reverse(target, axis=[-3])
        heightmap = tf.reverse(heightmap, axis=[-3])
        landclass = tf.reverse(landclass, axis=[-3])
        #wind_0    = tf.reverse(wind_0, axis=[-3])
        #wind_f    = tf.reverse(wind_f, axis=[-3])
        weather[[2,3],:] = weather[[3,2],:] # flip y

    if fliplr:
        # apply left/right flip
        arrival   = tf.reverse(arrival, axis=[-2])
        target    = tf.reverse(target, axis=[-2])
        heightmap = tf.reverse(heightmap, axis=[-2])
        landclass = tf.reverse(landclass, axis=[-2])
        #wind_0    = tf.reverse(wind_0, axis=[-2])
        #wind_f    = tf.reverse(wind_f, axis=[-2])
        weather[[0,1],:] = weather[[1,0],:] # flip x
        

    if transp:
        # apply x-y transposition    
        ##perm = [0, 2, 1, 3]
        ##perm_c = [0, 1, 3, 2, 4]
        
        perm = [1, 0, 2]
        perm_c = [0, 2, 1, 3]
        
        arrival   = tf.transpose(arrival,perm=perm)
        target    = tf.transpose(target,perm=perm)
        heightmap = tf.transpose(heightmap,perm=perm)
        landclass = tf.transpose(landclass,perm=perm)
        #wind_0    = tf.transpose(wind_0,perm=perm)
        #wind_f    = tf.transpose(wind_f,perm=perm)
        #other_weather = tf.transpose(other_weather,perm=perm)
        #ones_pad = tf.transpose(ones_pad, perm=perm)

        weather[[0,1,2,3],:] = weather[[2,3,0,1],:] # swap x and y
        

    
    
    # concatenate weather data together
    ##weather = tf.concat([wind_0_xy, wind_f_xy, other_weather], axis=-1)
    ##wind_x = tf.expand_dims(tf.expand_dims(wind_x, 0), 0)
    ##wind_y = tf.expand_dims(tf.expand_dims(wind_y, 0), 0)
    
    ##weather = tf.concat([wind_x, wind_y, other_weather], axis=-1)
        
    
    # expand weather height and width dimensions to match the input
    #ones = tf.keras.backend.ones([256,256,1], dtype=tf.float32) # make tensor same size as image (width, height, 1)
    #ones = tf.expand_dims(ones, 0)
    
    #wide_weather = tf.math.multiply(weather, ones_pad)
    #wide_weather = weather 
    
    
    
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
   
    
    
    
    
def crop_about_point(img, x, y, w):
    """
    crops around a tensorflow array centred at coordinate x, y with a square window of size w.
    0 value padding is added to the image to ensure the resulting image doesn't exceed image bounds
    """
    return None
    
    
    
def binned_mse_loss(y, y_pred):
    """
    calculates the mse loss between two images where emphasis is placed on pixels with non-trivial values.
    The motivation behind this is that many arrival images for fires have many pixels that have values of 1 or 0.
    When 1's dominate an arrival time image, the model predicts all 1's and achives good mse loss. 
    
    In this case we bin the loss for all pixels in the True image T into:
        1. those whose values are 0
        2. those whose values are 1
        3. remaining values
    The loss is then given by the average mse loss of each subsection of the image, regardless of the size (number of pixels) 
    in each bin. This also helps emphasise the loss of fires which take a relatively small proportion of the image space.
    """
    
    
    #n0 = tf.math.reduce_sum(tf.cast(y == 0, tf.float32))
    #nM = tf.math.reduce_sum(tf.cast(tf.logical_and(y > 0, y < 1), tf.float32))
    #n1 = tf.math.reduce_sum(tf.cast(y == 1, tf.float32))

        
    
    y_pred = tf.convert_to_tensor(y_pred)
    y = tf.convert_to_tensor(y)
    
    square_diff = tf.math.square(y-y_pred)
    
    mask0 = tf.cast(y==0, tf.float32)
    l0 = tf.reduce_sum(mask0 * square_diff / tf.reduce_sum(mask0), axis=None)
    
    maskM = tf.cast(tf.logical_and(y>0, y<1), tf.float32)
    lM = tf.reduce_sum(maskM * square_diff / tf.reduce_sum(maskM), axis=None)
    
    mask1 = tf.cast(y==1, tf.float32)
    l1 = tf.reduce_sum(mask1 * square_diff / tf.reduce_sum(mask1), axis=None)

    binned_mse = (l0+l1)/4 + lM/2
    
    return binned_mse
    

    
    
    
def scaled_mse_loss(y, y_pred):
    
    tol = 0.0001 # stops loss exploding to infinity 
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
    y_init = 1 - tf.cast(y == 0, tf.float32)
    
    num = tf.reduce_sum(tf.math.square(y_pred-y))
    den = tf.reduce_sum(tf.math.square(y_init-y))

    loss = num / (den + tol)
    
    return loss

def get_rand():
    
    return np.random.choice([True, False])

def l2_relative_loss(y, y_pred):
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
       
    y_init = tf.cast(y > 0.99, tf.float32)
    den = tf.reduce_sum(tf.math.square(y_init - y))
    num = tf.reduce_sum(tf.math.square(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ den) * nat_to_dec # converted to log_10 
    
    # the smaller num/dem the closer to perfect the model is
    # the log term ensures loss accelerates exponentially as the ratio approaches zero
    
    return loss

def l2_relative_loss_cropped(y, y_pred):
    
    if len(y.shape) == 3:
        y_pred = y_pred[32:-32, 32:-32]
        y      = y[32:-32, 32:-32]
    else:
        y_pred = y_pred[:, 32:-32, 32:-32, :]
        y      = y[:, 32:-32, 32:-32, :]
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
       
    y_init = tf.cast(y > 0.99, tf.float32)
    den = tf.reduce_sum(tf.math.square(y_init - y))
    num = tf.reduce_sum(tf.math.square(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ den) * nat_to_dec # converted to log_10 
    
    # the smaller num/dem the closer to perfect the model is
    # the log term ensures loss accelerates exponentially as the ratio approaches zero
    
    return loss

def l1_relative_loss(y, y_pred):
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
       
    y_init = tf.cast(y > 0.99, tf.float32)
    den = tf.reduce_sum(tf.math.abs(y_init - y))
    num = tf.reduce_sum(tf.math.abs(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ den) * nat_to_dec # converted to log_10 
    
    return loss


def l1_relative_loss_cropped(y, y_pred):
    
    if len(y.shape) == 3:
        y_pred = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
    else:
        y_pred = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
       
    y_init = tf.cast(y > 0.99, tf.float32)
    den = tf.reduce_sum(tf.math.abs(y_init - y))
    num = tf.reduce_sum(tf.math.abs(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ den) * nat_to_dec # converted to log_10 
    
    return loss

def l1_mod_relative_loss_cropped(y, y_pred):
    
    if len(y.shape) == 3:
        y_pred = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
    else:
        y_pred = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
     
    # old version    
    #y_init = tf.cast(y > 0.99, tf.float32)
    
    # new version
    y_init = tf.where(y < 1.0, 0.0, y)
    
    y_mod = tf.math.exp(y) # weighting on loss which more greatly penalizes errors near the extremedies of the fire
    den = tf.reduce_sum(tf.math.abs(y_init - y)*y_mod)
    num = tf.reduce_sum(tf.math.abs(y_pred - y)*y_mod)
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ den) * nat_to_dec # converted to log_10 
    
    return loss



def l1_rel_crop_aux(y, y_pred):
    
    
    if len(y.shape) == 3:
        
        y_pred = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
    else:
        y_pred = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
    
    
    y_init = tf.cast(y > 0.99, tf.float32)
    
    def get_loss(y_pred, y):
        y_pred = tf.convert_to_tensor(y_pred)
        y      = tf.convert_to_tensor(y)
        #print(y_pred.shape)
        #print(y.shape)
        #print('---')
        den = tf.reduce_sum(tf.math.abs(y_init - y))
        num = tf.reduce_sum(tf.math.abs(y_pred - y))
        nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
        loss = tf.math.log((num + 10e-12)/ den) * nat_to_dec # converted to log_10 
        return loss
    
    y_init = tf.squeeze(y_init, -1)
    y = tf.squeeze(y, -1)
    # calculate intermediate values
    y1 = tf.clip_by_value(tf.where(y < 0.001, 0., y+0.75), 0, 1)
    y2 = tf.clip_by_value(tf.where(y < 0.001, 0., y+0.5), 0, 1)
    y3 = tf.clip_by_value(tf.where(y < 0.001, 0., y+0.25), 0, 1)
    
    
    l1 = get_loss(y_pred[:,:,:,0], y1)
    l2 = get_loss(y_pred[:,:,:,1], y2)
    l3 = get_loss(y_pred[:,:,:,2], y3)
    l4 = get_loss(y_pred[:,:,:,3], y)
    
    loss = 0.125*l1 + 0.25*l2 + 0.5*l3 + l4
    #loss = l4
    
    return loss



def rel_mse_loss_central(y, y_pred):
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
    
    y_pred_total = tf.reduce_sum(y_pred)
    
    # avoid fire spilling over from edges from affecting mse if fraction < 1
    fraction = 0.5
    y_pred = tf.image.central_crop(y_pred, fraction)
    y      = tf.image.central_crop(y, fraction)
    
    y_pred_subtotal = tf.reduce_sum(y_pred)
    noise = y_pred_total - y_pred_subtotal
       
    y_init = tf.cast(y>0.99, tf.float32)
    den = tf.reduce_sum(tf.math.square(y_init - y))
    num = tf.reduce_sum(tf.math.square(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ den + noise) * nat_to_dec # converted to log_10 
    
    
    # the smaller num/dem the closer to perfect the model is
    # the log term ensures loss accelerates exponentially as the ratio approaches zero
    

    
    
    return loss

def iou_metric(y, y_pred):
    # intersection over union loss.
    # scaled by change in area
    

    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
    #y_max  = tf.math.reduce_max(y) #finds largest value in array
    

    y_pred = tf.where((y_pred > 0) & (y_pred <= 1), 1, 0) # area burnt during latest interval
    y      = tf.where((y > 0) & (y <= 1), 1, 0) # area burnt during latest interval

    Y = y + y_pred 

    intersection = tf.reduce_sum(tf.where(Y == 2, 1, 0))
    union = tf.reduce_sum(tf.where(Y >= 1, 1, 0))

    iou = intersection / union

    return iou

def iou_metric_cropped(y, y_pred):
    # intersection over union loss.
    # scaled by change in area

    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
    #y_max  = tf.math.reduce_max(y) #finds largest value in array
    fraction = 0.5
    y_pred = tf.image.central_crop(y_pred, fraction)
    y      = tf.image.central_crop(y, fraction)

    y_pred = tf.where((y_pred > 0) & (y_pred <= 1), 1, 0) # area burnt during latest interval
    y      = tf.where((y > 0) & (y <= 1), 1, 0) # area burnt during latest interval

    Y = y + y_pred 

    intersection = tf.reduce_sum(tf.where(Y == 2, 1, 0))
    union = tf.reduce_sum(tf.where(Y >= 1, 1, 0))

    iou = intersection / union

    return iou

    
def dice_metric(y, y_pred):
    # dice metric 
    # scaled by change in area
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
    #y_max  = tf.math.reduce_max(y) #finds largest value in array
    

    y_pred = tf.where((y_pred > 0) & (y_pred <= 1), 1, 0) # area burnt during latest interval
    y      = tf.where((y > 0) & (y <= 1), 1, 0) # area burnt during latest interval
    
    Y = y + y_pred 
    
    intersection = tf.reduce_sum(tf.where(Y == 2, 1, 0))
    sum_set_size = tf.reduce_sum(Y)
    
    dice = 2*intersection / sum_set_size
    
    return dice
   
def dice_metric_cropped(y, y_pred):
    # dice metric 
    # scaled by change in area
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
    #y_max  = tf.math.reduce_max(y) #finds largest value in array
    fraction = 0.5
    y_pred = tf.image.central_crop(y_pred, fraction)
    y      = tf.image.central_crop(y, fraction)

    y_pred = tf.where((y_pred > 0) & (y_pred <= 1), 1, 0) # area burnt during latest interval
    y      = tf.where((y > 0) & (y <= 1), 1, 0) # area burnt during latest interval
    
    Y = y + y_pred 
    
    intersection = tf.reduce_sum(tf.where(Y == 2, 1, 0))
    sum_set_size = tf.reduce_sum(Y)
    
    dice = 2*intersection / sum_set_size
    
    return dice