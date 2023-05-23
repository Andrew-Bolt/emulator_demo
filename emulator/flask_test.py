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

from flask import Response

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


## 2b. FLASK SETUP

from flask import Flask, render_template, request, jsonify

# Test implementation of Flask

app = Flask(__name__, static_url_path='/static')

# main html processes
@app.route("/", methods=['GET', 'POST'])
def render_website(): 
    
    
    return render_template('basic_template.html')




# instantiate model
def instantiate_model():
    """
    loads in the packages for using the NN model. Loads the model and its weights.
    """
   
    
    ## 4. LOAD IN MODEL 
    
    leaky_10 = tf.keras.layers.LeakyReLU(alpha=.1)
    activation = leaky_10
    model = ft.model.cnn_rnn_model(activation)

    model.load_weights('./latest_model/saved_weights.ckpt')
    
    return model # return the NN model

model = instantiate_model()


@app.route("/input_value_checks", methods=['GET', 'POST'])
def input_value_checks():

    ## example of data from back end to front end
    # get time in seconds
    if request.is_json:
        #seconds = time.time()
        
        ## read in values of input fields. Try to interpret as integers
        try:
            sample_num = int(request.args.get('sample_num'))
        except:
            sample_num = None
        try:
            start_int = int(request.args.get('start_int'))
        except:
            start_int = None
        try:
            finish_int = int(request.args.get('finish_int'))
        except:
            finish_int = None
                   
        
        # default input values
        d_sample_num = random.randint(0, 169)
        d_start_int = 0
        d_finish_int = 22
        

        if sample_num not in range(0, 170):
            sample_num = d_sample_num
            
        if start_int not in range(0, 22):
            start_int = d_start_int
        if finish_int not in range(1, 23):
            finish_int = d_finish_int
        
        if finish_int <= start_int:
            start_int = finish_int-1
        
        
        return jsonify({#'seconds': seconds,
                        'sample_num':sample_num,
                        'start_int':start_int,
                        'finish_int':finish_int
                        })
    return "Nothing"

def make_array(string_input):
    
    x = np.array(eval('[' + str(string_input) + ']'))
    
    return x

# processes to execute without page refresh
@app.route('/run_model')
def run_model(wthr=None):
    """
    runs the model and produces figures
    """
    WEATHER = None
    
    r = int(request.args.get('r'))
    START_INTERVAL = int(request.args.get('start_int'))
    STOP_INTERVAL = int(request.args.get('finish_int'))
    INDEX = int(request.args.get('sample_num'))
    
    WEATHER = None
    if r == 1:
        try:
            ws = make_array(request.args.get('ws'))
            wd = make_array(request.args.get('wd'))
            tp = make_array(request.args.get('tp'))
            rh = make_array(request.args.get('rh'))

            WEATHER = np.array([[ws, wd, tp, rh]])
        except: 
            WEATHER = None

        
    #print(WEATHER)
    

    #if RANDOM_CHECKBOX == 'true':
    #    INDEX = -1
    
    if INDEX == -1:
        predict_indices = master_df.index
        index = np.random.choice(predict_indices) # get a random index
    else:
        index = INDEX
    
    ######
    
        ## 3. CREATE TensorFlow Dataset OBJECT
    
    load_trial = ft.dataset.load_trial
    load_start_stop_predict = ft.dataset.load_start_stop_predict
    transform_identity = ft.dataset.transform_identity
    transform_trial = ft.dataset.transform_trial


    random.seed(42)


    def predict_start_stop_generator(index=-1, start=START_INTERVAL, stop=STOP_INTERVAL):

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
    
    
    
    #### 5b. SETUP
    
    PADDING = 32 # minimum padding value

    # pass arguments to TensorFlow Dataset Object
    #print('weather TS: ' + str(WEATHER))
    print('trial number:' + str(index))
    #print('start interval: ' + str(START_INTERVAL))
    #print('finish interval: ' + str(STOP_INTERVAL))
    args = [index, START_INTERVAL, STOP_INTERVAL] # trial index, starting interval, finishing interval
    

    
    
    tf_dataset = create_tf_dataset(predict_start_stop_generator, transform_trial, output_shapes = output_shapes, args=args)
    X, y = next(tf_dataset.take(1).as_numpy_iterator())

    # modify weather inputs if specified as an argument
    #WEATHER_INPUT # n by 4. need to reshape into n-1 * 8
    WEATHER_INPUT = None
    
    
    if WEATHER is not None:
        #print('using custom weather...')
        WEATHER_INPUT = WEATHER
        WEATHER_INPUT = np.squeeze(WEATHER_INPUT, axis=0)
        WEATHER_INPUT = np.swapaxes(WEATHER_INPUT, 0, 1)
       
        # scale the weather
        #print(WEATHER_INPUT)
        
        x_vec = WEATHER_INPUT[:, 0] * np.cos((90 - WEATHER_INPUT[:, 1]) * np.pi / 180)
        y_vec = WEATHER_INPUT[:, 0] * np.sin((90 - WEATHER_INPUT[:, 1]) * np.pi / 180)
        
        # scale wind factors
        
        x_wind = 60 * x_vec / master_df.loc[0, 'g_max_wind']
        y_wind = 60 * y_vec / master_df.loc[0, 'g_max_wind']
        
        temp   = (WEATHER_INPUT[:, 2] - master_df.loc[0, 'g_min_temp']) / (master_df.loc[0, 'g_max_temp'] - master_df.loc[0, 'g_min_temp'])
        #rh     = (WEATHER_INPUT[:, 3]/100 - master_df.loc[0, 'g_min_RH']) / (master_df.loc[0, 'g_max_RH'] - master_df.loc[0, 'g_min_RH'])
        rh     = WEATHER_INPUT[:, 3]/100
        
        x_wind = np.expand_dims(x_wind, axis=-1)
        y_wind = np.expand_dims(y_wind, axis=-1)
        temp   = np.expand_dims(temp, axis=-1)
        rh     = np.expand_dims(rh, axis=-1)
        
        WEATHER_INPUT = np.concatenate((x_wind, y_wind, temp, rh), axis=1)
        
        print(WEATHER_INPUT)
        
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
    y_pred_out = model.predict((X), verbose=False)
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
    a_pred = y_pred[PADDING:-PADDING, PADDING:-PADDING] # emulated
    a_init = y_init[PADDING:-PADDING, PADDING:-PADDING] # initial 
    a_ground = y_ground[PADDING:-PADDING, PADDING:-PADDING] # simulated

    # create plots for inital, simulated and emulated fire fronts

    # plot initial fire shape
    ft.graphics.plot_fire(a_init, title='Initial Fire Shape', name= im_path_prefix + 'initial' + save_type, meta=meta_data, show=False)  
    # plot simulated fire shape
    ft.graphics.plot_fire(a_ground, title='Simulated Fire Shape', name= im_path_prefix + 'simulated' + save_type, meta=meta_data, show=False)
    
    ft.graphics.plot_fire(a_pred, title='Emulated Fire Shape', iou=None, name= im_path_prefix + 'emulated' + save_type, meta=meta_data, show=False)
    
    # attempting to stream test image
    # plot emulated fire shape
    emulated_plot = ft.graphics.plot_fire_embed(a_pred, title='Emulated Fire Shape', iou=None, name= im_path_prefix + 'emulated' + save_type, meta=meta_data, show=False)


    # plot (simulated - emulated) fire fronts
    ft.graphics.plot_fire_difference(a_pred, a_ground, a_init, name= im_path_prefix + 'difference' + save_type, iou=None, meta=meta_data, show=False)

    # create a coloured landclass map
    land_colors = [[0.2,0.2,1,1], [0,0.7,0,1], [1,1,0,1], [1,0.5,0,1], [0.5,0.5,0,1], [0.7,0,0,1], [0.1,0.1,0.1,1]]
    land_colors = [[0,0,1,1], [0,0.8,0,1], [1,1,0.1,1], [1,0.5,0,1], [0.5,0.5,0,1], [1,0.1,0.2,1], [1,1,1,1]]
    land_names = ['water/unburnable','stringbark forrest','grazed grassland','mallee heath','spinifex grassland','heathland', 'pine plantation']
    landclass = X[4][batch_slice][PADDING:-PADDING, PADDING:-PADDING, :]

    # plot initial, emulated and simulated contours over a landclass map
    ft.graphics.plot_land_classes(landclass, a_pred, a_ground, a_init, name= im_path_prefix + 'land_classes' + save_type,
                                  land_names=land_names, land_colors=land_colors, iou=None, meta=meta_data, show=False)

        
    if WEATHER_INPUT is None:
        new_weather = np.concatenate((weather[:, ::2], np.expand_dims(weather[-1, 1::2], axis=0)), axis=0)
    else: 
        new_weather = WEATHER_INPUT
        
    WX   = new_weather[:, 0]
    WY   = new_weather[:, 1]
    WS   = np.sqrt(WX ** 2 + WY ** 2) * w_scaling
    WD   = 90 - np.degrees(np.arctan2(WY, WX)) # yes, Y before X
        
    TEMP = new_weather[:, 2] * t_scaling + t_offset
    RH   = new_weather[:, 3] * 100 # make a percentage
    
    # perform rounding
    #WS = np.round(WS, 1)
    #WD = np.round(WD, 0)
    #TEMP = np.round(TEMP, 1)
    #RH = np.round(RH, 1)
    
    
    weather_df = np.concatenate((np.expand_dims(WS, 0),
                                 np.expand_dims(WD, 0),
                                 np.expand_dims(TEMP, 0),
                                 np.expand_dims(RH, 0)),
                                 axis=0)
    
    weather_df = pd.DataFrame(np.transpose(weather_df), columns=['WS', 'WD', 'TEMP', 'RH']) 
    
    
    # convert pandas df to json
    weather_json = weather_df.transpose().to_json()
    
    
    #print(weather_json)
    print('finished')
        
    return jsonify({'weather' : weather_json,
                    #'sample_num': index # allows update of index if -1 was passed
                    'emulated_plot' : 5,#emulated_plot streamed - currently not working
                    'iou' : iou_val
                   })    
        
    # end of "run_model"
    #return "Nothing" # dummy output

    


# allow the process to be executed as a python script
# by $: python3 flask_test.py

if __name__ == '__main__': 
    
    app.run()
    
    