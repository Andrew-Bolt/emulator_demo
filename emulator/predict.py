## 1. Handle input arguments

import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    # Positional mandatory arguments
    
    # Optional arguments
    
    parser.add_argument("-si", "--START_INTERVAL", help="starting interval", type=int, default=0)
    parser.add_argument("-fi", "--STOP_INTERVAL", help="final interval", type=int, default=22)
    parser.add_argument("-idx", "--INDEX", help="sample index to use. -1 is random", type=int, default=-1)
    parser.add_argument("-ts", "--WEATHER_INPUT", help="use custom weather inputs", type=str, default=None)
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    # parse in the arguments
    args = parse_arguments()
    
    START_INTERVAL = args.START_INTERVAL
    STOP_INTERVAL = args.STOP_INTERVAL
    INDEX = args.INDEX
    if args.WEATHER_INPUT is not None:
        WEATHER_INPUT = eval(args.WEATHER_INPUT) # convert string into literal expression
    else:
        WEATHER_INPUT = None
    
    # run the rest of the script
    
    ## 1. LOAD LIBRARIES
    
    # libraries for data manipulation
    import pandas as pd
    import numpy as np
    import random
    import time
    import os

    # custom libraries for fire modelling
    import fire_tools as ft

    # tensorflow library
    import tensorflow as tf

    # libraries for plotting and graphics
    from PIL import Image
    from copy import copy
    import matplotlib.colors as colors
    import rasterio
    import matplotlib.pyplot as plt
    
    
    ## 2. SET FILEPATHS, LOAD DATAFRAMES
    
    # file paths
    cwd = os.path.dirname(os.getcwd())
    raw_path = os.path.dirname(cwd) + '/data/raw/'
    proc_path = os.path.dirname(cwd) + '/data/processed/' # new 'shared' location
    pick_dir = os.path.dirname(cwd) + '/data/'


    # csv dataframes
    # these contain information about a given sample, such as file locations and metadata.
    data_pkl = pick_dir + 'master.pkl'
    trials_pkl = pick_dir + 'trials.pkl'

    master_df = pd.read_pickle(data_pkl)
    trials_df = pd.read_pickle(trials_pkl)


    ## 3. CREATE TensorFlow Dataset OBJECT
    
    load_trial = ft.dataset.load_trial
    load_start_stop_predict = ft.dataset.load_start_stop_predict
    transform_identity = ft.dataset.transform_identity
    transform_trial = ft.dataset.transform_trial


    random.seed(42)


    def predict_start_stop_generator(index=-1, start=START_INTERVAL, stop=STOP_INTERVAL):

        if index==-1: #randomize

            # randomly order the trials to process each epoch
            predict_indices = master_df.index # search over all validation entries
            #predict_indices = trials_df[trials_df.train_or_test == 'test'].index

            #indices = predict_indices
            indices = np.random.choice(predict_indices, len(predict_indices), replace=False)

            for ind in indices:

                features, target = load_start_stop_predict(ind, master_df, transform = [0.0, 0.0, 0.0, 0.0], start=start, stop=stop)

                yield features, target

        else: # use specified index

            features, target = load_start_stop_predict(index, master_df, transform = [0.0, 0.0, 0.0, 0.0], start=start, stop=stop)

            yield features, target


    output_types   = ((tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32), tf.float32)

    output_shapes  = (
    (tf.TensorShape([None, None, None]), 
     tf.TensorShape([None, 8]), 
     tf.TensorShape([2]),
     tf.TensorShape([None, None, None]), 
     tf.TensorShape([None, None, None]),
     tf.TensorShape([4])),
     tf.TensorShape([None, None, None]),
    )

    padded_shapes  = (((None,None,None), (None, 8), (2,), (None,None,None), (None,None,None), (4,)),(None,None,None))


    ## Create a tf Dataset
    args = [-1, START_INTERVAL, STOP_INTERVAL]
    def create_tf_dataset(generator, transform, outout_types=output_types, output_shapes = output_shapes, args=args):

        tf_dataset = tf.data.Dataset.from_generator(generator, output_types=output_types, output_shapes=output_shapes, args=args)
        tf_dataset = tf_dataset.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tf_dataset = tf_dataset.batch(1)

        return tf_dataset
    
    
    ## 4. LOAD IN MODEL 
    
    leaky_10 = tf.keras.layers.LeakyReLU(alpha=.1)
    activation = leaky_10
    model = ft.model.cnn_rnn_model(activation)

    model.load_weights('./latest_model/saved_weights.ckpt')
    
    
    ## 5. PASS FEATURES TO MODEL AND GENERATE PREDICTION
    
    padding = 1 # minimum padding value

    # pass arguments to TensorFlow Dataset Object
    args = [INDEX, START_INTERVAL, STOP_INTERVAL] # trial index, starting interval, finishing interval
    tf_dataset = create_tf_dataset(predict_start_stop_generator, transform_trial, output_shapes = output_shapes, args=args)
    X, y = next(tf_dataset.take(1).as_numpy_iterator())
    
    # modify weather inputs if specified as an argument
    #WEATHER_INPUT # n by 4. need to reshape into n-1 * 8
    
    if WEATHER_INPUT is not None:
        
        WEATHER_INPUT = np.array(WEATHER_INPUT)
        
        # scale the weather
        x_wind = 60 * WEATHER_INPUT[:, 0] / master_df.loc[0, 'g_max_wind']
        y_wind = 60 * WEATHER_INPUT[:, 1] / master_df.loc[0, 'g_max_wind']
        
        temp   = (WEATHER_INPUT[:, 2] - master_df.loc[0, 'g_min_temp']) / (master_df.loc[0, 'g_max_temp'] - master_df.loc[0, 'g_min_temp'])
        rh     = (WEATHER_INPUT[:, 3] - master_df.loc[0, 'g_min_RH']) / (master_df.loc[0, 'g_max_RH'] - master_df.loc[0, 'g_min_RH'])
        
        x_wind = np.expand_dims(x_wind, axis=-1)
        y_wind = np.expand_dims(y_wind, axis=-1)
        temp   = np.expand_dims(temp, axis=-1)
        rh     = np.expand_dims(rh, axis=-1)
        
        WEATHER_INPUT = np.concatenate((x_wind, y_wind, temp, rh), axis=1)
        
        # convert into ML pipeline format
        
        new_weather = np.zeros((WEATHER_INPUT.shape[0]-1, WEATHER_INPUT.shape[1]*2))
        
        # interleave weather at time t and t+1
        new_weather[:, ::2] = WEATHER_INPUT[:-1, :]
        new_weather[:, 1::2]= WEATHER_INPUT[1:, :]
        
        #new_weather = np.array(np.concatenate((WEATHER_INPUT[:-1, :], WEATHER_INPUT[1:, :]), axis=1))
        new_weather = np.expand_dims(new_weather, axis=0)
      
        # update model input features
        
        X = [X[0], new_weather, X[2], X[3], X[4]]
        


        
    batch_size = len(y)
    # basic timing of the emulator 
    start_time = time.time()
    y_pred_out = model.predict((X))
    end_time = time.time()
    run_time = end_time - start_time
    print('emulator run time: ' + str(run_time))
    
    
    ## 6. EXTRACT DATA AND RESHAPE FOR PLOTTING
    
    batch_slice = 0 # for batch_size = 0 datasets

    # extract weather from dataset
    weather = X[1][0]
    t = len(weather[:,2]) # number of intervals

    # normalize weather
    max_wind = master_df.loc[0].g_max_wind
    res = master_df.loc[0].resolution
    interval_dur = float(master_df.loc[0].interval[:-1])
    w_scaling = max_wind * res / interval_dur

    max_temp = master_df.loc[0].g_max_temp
    min_temp = master_df.loc[0].g_min_temp
    t_scaling = max_temp - min_temp
    t_offset = min_temp


    # extract initial fire arrival map
    y_init = X[0][batch_slice][..., 0]

    # extract simulated fire arrival map
    y_ground = y[batch_slice][..., 0]

    # extract emulated fire arrival map
    y_pred = y_pred_out[batch_slice][..., 0]

    # make simulated and emulated arrays the correct shape to calculate loss/IOU scores
    y_g = np.expand_dims(y_ground, -1)
    y_p = np.expand_dims(y_pred, -1)
    y_g = np.expand_dims(y_g, 0)
    y_p = np.expand_dims(y_p, 0)


    # calculate the Jaccard (IOU score)
    metric = ft.metric_and_loss.iou_metric
    iou_val = metric(y_g, y_p).numpy()    
    print('IOU: ' + str(iou_val))


    img_size = X[0][batch_slice].size
    #meta_data = {'Subject': 'IOU : ' + str(iou_val) + ', start_int : ' + str(start) +', finish_int: '+
    #             str(finish) + ', size: ' + str(img_size) + ', run time: ' + str(run_time) +', DF: ' 
    #             + str(X[2][0][0]) + ', CG: ' + str(X[2][0][1])}  
    meta_data = None   


    
    ## 7. CREATE AND SAVE PLOTS

    im_path_prefix = 'static/images/'
    save_type = '.jpg' # '.pdf'

    # windspeed plot
    ft.graphics.plot_winds(weather, name= im_path_prefix + 'winds' + save_type, meta=meta_data, scaling=w_scaling, show=False)

    # temperature, rh plot
    ft.graphics.plot_temp(weather, name= im_path_prefix + 'temperature' + save_type, meta=meta_data, t_scaling=t_scaling, t_offset=t_offset, show=False)

    # crop arrival images
    a_pred = y_pred[padding:-padding, padding:-padding] # emulated
    a_init = y_init[padding:-padding, padding:-padding] # initial 
    a_ground = y_ground[padding:-padding, padding:-padding] # simulated

    # create plots for inital, simulated and emulated fire fronts

    # plot initial fire shape
    ft.graphics.plot_fire(a_init, title='Initial Fire Shape', name= im_path_prefix + 'initial' + save_type, meta=meta_data, show=False)  
    # plot simulated fire shape
    ft.graphics.plot_fire(a_ground, title='Simulated Fire Shape', name= im_path_prefix + 'simulated' + save_type, meta=meta_data, show=False)
    # plot emulated fire shape
    ft.graphics.plot_fire(a_pred, title='Emulated Fire Shape', iou=None, name= im_path_prefix + 'emulated' + save_type, meta=meta_data, show=False)


    # plot (simulated - emulated) fire fronts
    ft.graphics.plot_fire_difference(a_pred, a_ground, a_init, name= im_path_prefix + 'difference' + save_type, iou=None, meta=meta_data, show=False)

    # create a coloured landclass map
    land_colors = [[0.2,0.2,1,1], [0,0.7,0,1], [1,1,0,1], [1,0.5,0,1], [0.5,0.5,0,1], [0.7,0,0,1], [0.1,0.1,0.1,1]]
    land_colors = [[0,0,1,1], [0,0.8,0,1], [1,1,0.1,1], [1,0.5,0,1], [0.5,0.5,0,1], [1,0.1,0.2,1], [1,1,1,1]]
    land_names = ['water/unburnable','stringbark forrest','grazed grassland','mallee heath','spinifex grassland','heathland', 'pine plantation']
    landclass = X[4][batch_slice][padding:-padding, padding:-padding, :]

    # plot initial, emulated and simulated contours over a landclass map
    ft.graphics.plot_land_classes(landclass, a_pred, a_ground, a_init, name= im_path_prefix + 'land_classes' + save_type,
                                  land_names=land_names, land_colors=land_colors, iou=None, meta=meta_data, show=False)