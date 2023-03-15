import tensorflow as tf


IMAGE_WIDTH = None
IMAGE_HEIGHT = None
TIME_STEPS = None
slices = None
RNN = tf.keras.layers.RNN
InputSpec = tf.keras.layers.InputSpec


def test_model(activation='tanh'):
    """
    This model extracts features from landclass, heightmap, and weather. 
    A residual cell comprises inputs of these 'static' features, as well as the initial arrival image.
    An updated arrival image is produces. 
    Repeated calls are made to the residual cell, which attempts to generate the final arrival image
    """    
    
    past_arrival = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='arrival_inputs')
    weather = tf.keras.layers.Input(shape=[TIME_STEPS, 10], name='weather_inputs')
    heightmap = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 2], name='heightmap_inputs')
    landclass = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 6], name='landclass_inputs')
    
def model_down(activation='relu'):
    
    arrival = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='arrival_inputs')
         
    # downscale arrival features
    
    if False: # old encoding of latent arrival features
        a_down_1a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=4, padding='same', activation=activation, name='a_down_1a')
        a_down_1b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=4, padding='same', activation=activation, name='a_down_1b')
        a_down_2a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=8, padding='same', activation=activation, name='a_down_2a')
        a_down_2b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=8, padding='same', activation=activation, name='a_down_2b')
        a_down_3a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=16, padding='same', activation=activation, name='a_down_3a')
        a_down_3b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation, name='a_down_3b')

        latent_at = a_down_1a(arrival)
        latent_at = a_down_1b(latent_at)
        latent_at = tf.keras.layers.Dropout(0.25)(latent_at)
        latent_at = a_down_2a(latent_at)
        latent_at = a_down_2b(latent_at)
        latent_at = tf.keras.layers.Dropout(0.25)(latent_at)
        latent_at = a_down_3a(latent_at)
        latent_at = a_down_3b(latent_at)
        
    if True: # new downsampling of arrival features
        
        latent_at = tf.keras.layers.AveragePooling2D(pool_size=(2,2), padding='same')(arrival)
        latent_at = tf.nn.space_to_depth(latent_at, block_size=4, data_format='NHWC')

    model = tf.keras.Model(inputs=arrival, outputs=latent_at, name='model_down')        
    
    return model


def model_up(activation='relu'):
    
    latent_at = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 16], name='latent_at_inputs')
    
    if False: # older method
        # upscaling component

        a_up_1a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=8, padding='same', activation=activation, name='a_up_1a')
        a_up_1b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=8, padding='same', activation=activation, name='a_up_1b')
        a_up_2a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=8, padding='same', activation=activation, name='a_up_2a')
        a_up_2b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=4, padding='same', activation=activation, name='a_up_2b')
        a_up_3a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=4, padding='same', activation=activation, name='a_up_3a')
        a_up_3b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=1, padding='same', activation='sigmoid', name='a_up_3b') # in (0,1)

        x = a_up_1a(latent_at)
        x = a_up_1b(x)
        x = a_up_2a(x)
        x = a_up_2b(x)
        x = a_up_3a(x)
        arrival = a_up_3b(x)
        
    if True: # newer method
        
        x = tf.nn.depth_to_space(latent_at, block_size=4, data_format='NHWC')
        x = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=8, padding='same', activation=activation)(x)
        arrival = tf.keras.layers.Conv2D(kernel_size=3, filters=1, padding='same', activation='hard_sigmoid')(x)
        
        arrival = arrival * 2 -1 # correcting for hard sigmoid activation, where unburnt is -1

    model = tf.keras.Model(inputs=latent_at, outputs = arrival, name='model_up')
    
    return model

def model_autoencoder(activation='relu'):
    
    arrival = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='arrival_inputs')
    
    # downscaling component
    
    latent_at = model_down(activation=activation)(arrival)
    
    # upscalin component
    
    arrival_out = model_up(activation=activation)(latent_at)
    
    model = tf.keras.Model(inputs=arrival, outputs=arrival_out, name='model_autoencoder')        
    
    return model
    
def model_topo_down(activation='relu'):
    
    arrival = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 6], name='arrival_inputs')
         
    # downscale arrival features
    
    a_down_1a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=16, padding='same', activation=activation, name='a_down_1a')
    a_down_1b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation, name='a_down_1b')
    a_down_2a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=16, padding='same', activation=activation, name='a_down_2a')
    a_down_2b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation, name='a_down_2b')
    a_down_3a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=32, padding='same', activation=activation, name='a_down_3a')
    a_down_3b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation='sigmoid', name='a_down_3b')
    
    latent_at = a_down_1a(arrival)
    latent_at = a_down_1b(latent_at)
    #latent_at = tf.keras.layers.Dropout(0.25)(latent_at)
    latent_at = a_down_2a(latent_at)
    latent_at = a_down_2b(latent_at)
    #latent_at = tf.keras.layers.Dropout(0.25)(latent_at)
    latent_at = a_down_3a(latent_at)
    latent_at = a_down_3b(latent_at)

    model = tf.keras.Model(inputs=arrival, outputs=latent_at, name='model_topo_down')        

    return model

def model_topo_up(activation='relu'):
    
    latent_at = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 32], name='latent_at_inputs')
    
    # upscaling component
    
    a_up_1a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=32, padding='same', activation=activation, name='a_up_1a')
    a_up_1b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation, name='a_up_1b')
    a_up_2a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=16, padding='same', activation=activation, name='a_up_2a')
    a_up_2b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation, name='a_up_2b')
    a_up_3a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=16, padding='same', activation=activation, name='a_up_3a')
    a_up_3b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=6, padding='same', activation='sigmoid', name='a_up_3b') # in (0,1)
    
    x = a_up_1a(latent_at)
    x = a_up_1b(x)
    x = a_up_2a(x)
    x = a_up_2b(x)
    x = a_up_3a(x)
    arrival = a_up_3b(x)
    
    model = tf.keras.Model(inputs=latent_at, outputs = arrival, name='modeltopo__up')
    
    return model

def model_topo_autoencoder(activation='relu'):
    
    arrival = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 6], name='lc_inputs')
    
    # downscaling component
    
    latent_at = model_topo_down(activation=activation)(arrival)
    
    # upscalin component
    
    arrival_out = model_topo_up(activation=activation)(latent_at)
    
    model = tf.keras.Model(inputs=arrival, outputs=arrival_out, name='model_topo_autoencoder')        
    
    return model       
        
# current inner model class (RUI)    
class FCN_RNN7(tf.keras.layers.Layer):
    
    
    
    def __init__(self, activation='sigmoid', apply_dropout=True, apply_batchnorm=True):
        """
        The minimum number of filters are used on the outer layers. The number of filters double each sucessive layer
        towards the middle until the max_filters limit is reached.
        """
        super(FCN_RNN7, self).__init__()
        
        
        # ListWrapper([InputSpec(shape=(None, None, None, 16), ndim=4), InputSpec(shape=(None, None, None, 32), ndim=4)]
        
        self.state_size = (None, None, None)
        #self.dim = dim
        
        # define layers
        
        final_weather_filters = 6
        self.__setattr__('weather_filters', final_weather_filters)
        climate_features = 2
        self.__setattr__('climate_filters', climate_features)
        
        weather_1 = tf.keras.layers.Dense(6, activation=activation)
        weather_2 = tf.keras.layers.Dense(6, activation=activation)
        weather_3 = tf.keras.layers.Dense(final_weather_filters, activation=activation)
        self.__setattr__('weather_1', weather_1)
        self.__setattr__('weather_2', weather_2)
        self.__setattr__('weather_3', weather_3)

        concat_1 = tf.keras.layers.Concatenate(axis=-1)
        self.__setattr__('concat_1', concat_1)
        concat_2 = tf.keras.layers.Concatenate(axis=-1)
        self.__setattr__('concat_2', concat_2)
        
        conv_features = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)
        self.__setattr__('conv_features', conv_features)
        
        
        # first filter level to apply to concatenated arrival, topo, weather
        conv_c = tf.keras.layers.Conv2D(kernel_size=1, strides=1, filters=16, padding='same', activation=activation) #'sigmoid' 32/3/22
        self.__setattr__('conv_c', conv_c)
        
        # features
        conv_fs = tf.keras.layers.Conv2D(kernel_size=1, strides=1, filters=16, padding='same', activation=activation) #32
        self.__setattr__('conv_fs', conv_fs)
        
        
        conv_d1 = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=32, padding='same', activation=activation) #64
        conv_a1 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation) #64
        conv_d2 = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=64, padding='same', activation=activation) #64
        conv_a2 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation=activation) #64
        
        conv_u2 = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=32, padding='same', activation=activation) #32
        conv_b2 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation) #32
        
        conv_u1 = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=16, padding='same', activation=activation) #32
        conv_b1 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation) #32
        
        conv_f1 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation)

        self.__setattr__('conv_d1', conv_d1)
        self.__setattr__('conv_a1', conv_a1)
        self.__setattr__('conv_d2', conv_d2)
        self.__setattr__('conv_a2', conv_a2)
        
        self.__setattr__('conv_u2', conv_u2)
        self.__setattr__('conv_b2', conv_b2)
        self.__setattr__('conv_u1', conv_u1)
        self.__setattr__('conv_b1', conv_b1)
        
        self.__setattr__('conv_f1', conv_f1)
        
        #self.__setattr__('conv_d3', conv_d3)
        #self.__setattr__('conv_d4', conv_d4)
        #self.__setattr__('conv_d5', conv_d5)
        
        #x1_x2 = tf.keras.layers.add
        #x2_x3 = tf.keras.layers.add
        #x3_x4 = tf.keras.layers.add
        #x4_x5 = tf.keras.layers.add
        #self.__setattr__('x1_x2', x1_x2)
        #self.__setattr__('x2_x3', x2_x3)
        #self.__setattr__('x3_x4', x3_x4)
        #self.__setattr__('x4_x5', x4_x5)
        
        dropout_1 = tf.keras.layers.Dropout(0.25)
        dropout_2 = tf.keras.layers.Dropout(0.25)
        self.__setattr__('dropout_1', dropout_1)
        self.__setattr__('dropout_2', dropout_2)
        
        concat_3 = tf.keras.layers.Concatenate(axis=-1)
        self.__setattr__('concat_3', concat_3)
        
        
        smoother = tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation=activation)
        self.__setattr__('smoother', smoother)
        
        # number of filters in conv_f must match number of filteres in latent arrival time
        conv_f = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation='relu')
        self.__setattr__('conv_f', conv_f)
        
           
        
    def call(self, inputs, state):
        # INPUTS
        # state:arrival times,  inputs:weather series
        
        # OUTPUTS
        # state:arrival times 
        
        # state is a tuple if passed from RNN
        #if type(state) == tuple:
        #    state = state[0] # 
        
        arrival_full = state[0] # need to output an updated version of the arrival component of state
        
        ##landclass = state[1] # output the latent landclass component, scalar
        ##climate = state[2] # output the climate components, scalar
        ##heightmap_x = state[3] # output the latent heightmap component, vector
        ##heightmap_y = state[4]
        static_spatial = state[1]
        climate = state[2]
        
        w_inputs = inputs
        
        # add a couple of dense layers to weather 
        
        #w = tf.keras.layers.Dense(10, activation='relu')(w)
        #w = tf.keras.layers.Dense(16, activation='relu')(w)
        #w = tf.keras.layers.Dense(final_weather_filters, activation='relu')(w)
        
        
        
        weather_1 = self.__getattribute__('weather_1')
        weather_2 = self.__getattribute__('weather_2')
        weather_3 = self.__getattribute__('weather_3')
        
        #w = weather_1(w)
        #w = weather_2(w)
        #w = weather_3(w)
        #w = tf.expand_dims(w, 1)
        #w_inputs = tf.expand_dims(w, 1)
        
        # instantiate dropout layers
        dropout_1 = self.__getattribute__('dropout_1')
        dropout_2 = self.__getattribute__('dropout_2')

        
        # shape for 2d weather layer
        
        a_shape = tf.shape(arrival_full)
        
        weather_shape = tf.ones([a_shape[1],
                a_shape[2],
                self.__getattribute__('weather_filters')
                                ])
        
        climate_shape = tf.ones([a_shape[1],
                a_shape[2],
                self.__getattribute__('climate_filters')
                                ])
        c = tf.expand_dims(climate, 1)
        c = tf.expand_dims(c, 1)
        c_expanded = tf.math.multiply(climate_shape, c)
        
        # Start of weather interpolation

        w_o = w_inputs[:, 0::2] # inital weather
        w_f = w_inputs[:, 1::2] # final weather                      
                                
        
        #topo = tf.keras.layers.Concatenate(axis=-1)([landclass, heightmap_x, heightmap_y])
        #conv_fs = self.__getattribute__('conv_c')
        #fs = conv_fs(topo)
        
        
        
        ##arrival = tf.clip_by_value(arrival_full, clip_value_min=0.0, clip_value_max=1.0) # clip values
        
        slices = 4
        
        for j in range(slices): # j is a slice of time
        
            ##arrival = tf.clip_by_value(arrival_full, clip_value_min=0.0, clip_value_max=1.0 / slices) # clip values
            
            arrival = tf.clip_by_value(arrival_full, clip_value_min=0.0, clip_value_max=1.0 / slices)
            arrival_edges = tf.where(arrival <= 0.001, -1.0, arrival)
            ##arrival_edges = arrival
        
            frac = (j+0.5)/slices # fraction of init/final weather to mix

            w = w_o * (1-frac) + w_f * frac

            w = weather_1(w)
            w = weather_3(w) # skip weather_2 for simplicity
            w = tf.expand_dims(w, 1)
            w = tf.expand_dims(w, 1)

            #w_expanded = tf.math.multiply(tf.ones(tf.shape(arrival)), w)
            w_expanded = tf.math.multiply(weather_shape, w)


            concat_1 = self.__getattribute__('concat_1')
            features = tf.keras.layers.Concatenate(axis=-1)([w_expanded, c_expanded, static_spatial])
            conv_fs = self.__getattribute__('conv_fs')
            features = conv_fs(features)
            #features = tf.keras.layers.SpatialDropout2D(0.25)(features) # apply dropouts to entire channel
            
            
            x = tf.keras.layers.Concatenate(axis=-1)([arrival_edges, features])
            
            ##x = concat_1([arrival, w_expanded, c_expanded, static_spatial])

            # apply feature engineering layer to concated features (50 filters)
            conv_c = self.__getattribute__('conv_c')
            x = conv_c(x)
            
            #x = tf.keras.layers.Dropout(0.25)(x)



            # Small U-net

            # apply dilated convolutions / pyramid structure
            conv_d1 = self.__getattribute__('conv_d1') 
            conv_a1 = self.__getattribute__('conv_a1') 
            
            conv_d2 = self.__getattribute__('conv_d2')
            conv_a2 = self.__getattribute__('conv_a2')
            
            conv_u2 = self.__getattribute__('conv_u2') 
            conv_b2 = self.__getattribute__('conv_b2')
            
            conv_u1 = self.__getattribute__('conv_u1')
            conv_b1 = self.__getattribute__('conv_b1') 
            
            concat_2 = self.__getattribute__('concat_2')
            
            #downscale
            d1 = conv_d1(x)
            d1 = conv_a1(d1)
            #d1 = tf.keras.layers.Dropout(0.25)(d1)
            
            if False: #2nd downscaling component
            
                d2 = conv_d2(d1)
                d2 = conv_a2(d2)
                #d2 = tf.keras.layers.Dropout(0.25)(d2)


                # upscale
                u2 = conv_u2(d2)
                u2 = tf.keras.layers.Concatenate(axis=-1)([u2, d1])
                u2 = conv_b2(u2)
            else:
                u2 = conv_b2(d1) # single down behavior 24/4/22
            
            u1 = conv_u1(u2)
            u1 = tf.keras.layers.Concatenate(axis=-1)([u1, x])
            x = conv_b1(u1)
    
            # THERE IS A DISCONNECT ABOVE. u1 is not passed on, instead x is, missing all of the U-net component
                    
            #conv_f1 = self.__getattribute__('conv_f1')
            #x = conv_f1(x)
                
            conv_f = self.__getattribute__('conv_f')
            x = conv_f(x) / slices
        
            
        
            # residual compoent
        
            # update fire arrival latent values
            arrival = x + arrival # residual type component

            arrival = tf.where(arrival > 1.0 / slices, 1.0 / slices, arrival) # max addition to arrival full is 1.0 over loop
            
            #
            #arrival = tf.where(arrival > 0.99, 1.0, arrival) # clip values
            arrival = tf.where(arrival < 0.01 / slices, 0.0, arrival)

            arrival_full = arrival_full + arrival 
            
            
            # end of weather slice loop
        
        
        # new_state output is a tuple
        # ERROR arrival_full should only add values to arrival that are greater than 0.99?
        
        ##new_arrival = arrival + tf.where(arrival_full -1 > 0.0, arrival_full -1 , 0) # test
        #ew_arrival = arrival + arrival_full # current working
        #new_arrival = arrival + arrival_full#tf.where(arrival_full > 1, arrival_full, 0) # 30/3/23 test approach
        
        #new_arrival = tf.where(new_arrival < 0.01, 0.0, new_arrival)
        
        new_state = (arrival_full, static_spatial, climate) # current working
        
        
        ##new_state = (new_arrival, static_spatial, climate) # test
        
        
        ##output = arrival + arrival_full # current working
        output = arrival_full # test
            
        return output, new_state          
    
    
def cnn_rnn_model(activation='relu'): # outer model
    """
    Takes tensorflow tensor objects as a list of inputs. 
    Inputs = [arrival, heightmap, landclass]
    """
    arrival = tf.keras.layers.Input(shape=[IMAGE_WIDTH,IMAGE_HEIGHT,1], name='arrival')
    weather = tf.keras.layers.Input(shape=[TIME_STEPS, 8], name='weather_inputs')
    climate = tf.keras.layers.Input(shape=[2], name='climate_inputs')
    heightmap = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 2], name='heightmap_inputs')
    landclass = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 7], name='landclass_inputs')
        
    # create the input state from arrival and geospatial data
    ##topo = tf.keras.layers.Concatenate(axis=-1)([heightmap, landclass])
    ##state = (arrival, topo)
    
    # generate a reflect-padding layer
    reflect_pad = tf.keras.layers.Lambda(lambda x: tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), 'REFLECT'))
    
    w_inputs = weather
    
    
    
    # downscale spatial features
    
    
        
    if True: # old landclass extraction
        # extract landclass features
        #lc_feat_ext_1      = tf.keras.layers.Conv2D(kernel_size=1, strides=1, filters=2, padding='same', activation=activation)(landclass)
        #lc_feat_ext_2      = tf.keras.layers.Conv2D(kernel_size=1, strides=1, filters=2, padding='same', activation=activation)(landclass)

        # connect landclasses with climate conditions 
        # elipsis auto-fills in : for remaining dimensions

        ##climate_0 = tf.expand_dims(tf.expand_dims(tf.expand_dims(climate[..., 0], -1), -1), -1)
        ##climate_1 = tf.expand_dims(tf.expand_dims(tf.expand_dims(climate[..., 1], -1), -1), -1)
        ##climate_shape = tf.ones(tf.shape(landclass[0]
        
        ##lc_feat_ext_1 = tf.math.multiply(climate_0, lc_feat_ext_1) # scalar multiplied by tensor
        ##lc_feat_ext_2 = tf.math.multiply(climate_1, lc_feat_ext_2)
        
        
        #climate_shape = tf.ones([tf.shape(arrival)[1],
        #                         tf.shape(arrival)[2],
        #                         tf.shape(climate)[1]])
        
        #climate = tf.expand_dims(tf.expand_dims(climate, 1), 1)
        #wide_climate = tf.math.multiply(climate_shape, climate)
        
        
        
        #spatial_features = tf.keras.layers.Concatenate(axis=-1)([landclass, wide_climate])


        # downscale landclass (at full resolution, this is processed into 1 filter - ie. burnability)
        lc_burnability   = tf.keras.layers.Conv2D(kernel_size=1, strides=1, filters=4, padding='same', activation=activation)



        lc_down_1a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=4, padding='same', activation=activation)
        #lc_down_1b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=4, padding='same', activation=activation)
        lc_down_2a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=8, padding='same', activation=activation)
        #lc_down_2b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=8, padding='same', activation=activation)
        lc_down_3a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=16, padding='same', activation=activation)
        #lc_down_3b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation)


        #latent_lc = lc_burnability(landclass)
        latent_lc = lc_down_1a(landclass)
        #latent_lc = lc_down_1b(latent_lc)
        latent_lc = lc_down_2a(latent_lc)
        #latent_lc = lc_down_2b(latent_lc)
        latent_lc = lc_down_3a(latent_lc)
        #latent_lc = lc_down_3b(latent_lc)
        
    if False: # new landclass extraction


        latent_lc = tf.keras.layers.AveragePooling2D(pool_size=(2,2), padding='same')(landclass)
        
        ##climate_shape = tf.ones([tf.shape(latent_lc)[1],
        ##    tf.shape(latent_lc)[2],
        ##    int(2)])
     
        ##wide_climate = tf.math.multiply(tf.ones(climate_shape), climate)

        # incorporate static climate features
        ##latent_lc = tf.keras.layers.Concatenate([latent_lc, wide_climate])
        latent_lc = tf.nn.space_to_depth(latent_lc, block_size=4, data_format='NHWC')
        
    if False: # newest landclass extraction
        
        # instantiate the autoencoder component    
        ae_topo_model = model_topo_autoencoder(activation=activation)   
        ae_topo_model.load_weights('./latest_model/ae_topo_saved_weights.ckpt').expect_partial()
        # load in its weights for the model_down component
        layer_weights = ae_topo_model.layers[1].get_weights() # weights of model_down

        down_topo_layer = model_topo_down(activation=activation)
        down_topo_layer.set_weights(layer_weights)
        down_topo_layer.trainable = False # do not train this layer

        latent_lc = down_topo_layer(landclass) 
        
        
    if True: # old downscale gradients ie heighmap

        heightmap_x = tf.expand_dims(heightmap[..., 0], -1)
        heightmap_y = tf.expand_dims(heightmap[..., 1], -1)

        # downscale x components
        hm_down_1x = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=2, padding='same', activation=activation)
        hm_down_2x = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=4, padding='same', activation=activation)
        hm_down_3x = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=8, padding='same', activation=activation)

        latent_hm_x = hm_down_1x(heightmap_x)
        latent_hm_x = hm_down_2x(latent_hm_x)
        latent_hm_x = hm_down_3x(latent_hm_x)

        # downscale y components
        hm_down_1y = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=2, padding='same', activation=activation)
        hm_down_2y = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=4, padding='same', activation=activation)
        hm_down_3y = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=8, padding='same', activation=activation)

        latent_hm_y = hm_down_1x(heightmap_y)
        latent_hm_y = hm_down_2x(latent_hm_y)
        latent_hm_y = hm_down_3x(latent_hm_y)

    if False: # new downscale gradients
        
        heightmap_x = tf.expand_dims(heightmap[:,:,:,0], -1)
        heightmap_y = tf.expand_dims(heightmap[:,:,:,1], -1)
        
        latent_hm_x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), padding='same')(heightmap_x)
        latent_hm_y = tf.keras.layers.AveragePooling2D(pool_size=(2,2), padding='same')(heightmap_y)
        
        latent_hm_x = tf.nn.space_to_depth(latent_hm_x, block_size=4, data_format='NHWC')
        latent_hm_y = tf.nn.space_to_depth(latent_hm_y, block_size=4, data_format='NHWC')
             

    # downscale arrival features
    
    if True: # new way to incorporate downscaling from autoencoder component
    
        # instantiate the autoencoder component    
        ae_model = model_autoencoder(activation=activation)   
        ae_model.load_weights('./latest_model/ae_saved_weights.ckpt').expect_partial()
        # load in its weights for the model_down component
        layer_weights = ae_model.layers[1].get_weights() # weights of model_down

        down_layer = model_down(activation=activation)
        down_layer.set_weights(layer_weights)
        down_layer.trainable = False # do not train this layer

        latent_at = down_layer(arrival) 
    
    
    # create a set of static values in th e
    static_spatial = tf.keras.layers.Concatenate(axis=-1)([latent_hm_x, latent_hm_y, latent_lc])
    
    # create the latent state (tuple) of arrival and spatial inputs
    ###latent_state = (latent_at, latent_lc, climate, latent_hm_x, latent_hm_y)
    latent_state = (latent_at, static_spatial, climate)
        
    
    # create the recurrent layer 
    # this layer acts on latent features, which are then upscaled to create the final output
    CNNCell_1 = FCN_RNN7(activation=activation)
    CNNRNN_1  = RNN(CNNCell_1, return_sequences=False, name='P2P_RNN') # return all arrival time states
    # output from the recurrent layer
    output = CNNRNN_1(w_inputs, latent_state)
    
    num_intervals = tf.cast(tf.shape(w_inputs)[1], tf.float32)

    # clipping for FCN_RNN6
    ##output = tf.clip_by_value(output, clip_value_min=-1.0, clip_value_max=num_intervals) # if unburnt is -1.0
    
        
        
    output = output / num_intervals
    ##tf.where(output < 0.0, -1.0, output)

    
    # upscale arrival features
    
    if True: # new way to incorporate upscaling from autoencoder component
    
        # instantiate the autoencoder component    
        ae_model = model_autoencoder(activation=activation)   
        ae_model.load_weights('./latest_model/ae_saved_weights.ckpt').expect_partial()
        
            ## a.load_weights('name').expect_partial() # will silence warnings on partial used data (ie training config)
        
        # load in its weights for the model_up component
        layer_weights = ae_model.layers[2].get_weights() # weights of model_up

        up_layer = model_up(activation=activation)
        up_layer.set_weights(layer_weights)
        up_layer.trainable = False # do not train this layer

        arrival_final = up_layer(output) # equivalent of commented out above... hopefully
        # need to load in the model above using pretrained weights, if applicable    
        
        # clip small values (up_layer has soft sigmoid output)
        # if hard_sigmoid is used instead these steps can be omitted
        #arrival_final = tf.where(arrival_final < 0.01, 0.0, arrival_final)
        #arrival_final = tf.where(arrival_final > 0.99, 1.0, arrival_final)

    
    # scale output
    arrival_final = arrival_final * num_intervals
    
    arrival_final = tf.where(arrival_final < 0.0, 0.0, arrival_final)

    # sum over all arrival time states
    #summed_output = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=-4), name='reduce_sum')(output)
    
    
    
    model = tf.keras.Model(inputs=[arrival, w_inputs, climate, heightmap, landclass], outputs=arrival_final, name='rnn_model')
    
    return model    
    

    