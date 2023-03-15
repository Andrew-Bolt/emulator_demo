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
        arrival = tf.keras.layers.Conv2D(kernel_size=3, filters=1, padding='same', activation='sigmoid')(x)

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
    latent_at = tf.keras.layers.Dropout(0.25)(latent_at)
    latent_at = a_down_2a(latent_at)
    latent_at = a_down_2b(latent_at)
    latent_at = tf.keras.layers.Dropout(0.25)(latent_at)
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
        

class FCN_RNN3(tf.keras.layers.Layer):
    
    
    
    def __init__(self, activation='sigmoid', apply_dropout=True, apply_batchnorm=True):
        """
        The minimum number of filters are used on the outer layers. The number of filters double each sucessive layer
        towards the middle until the max_filters limit is reached.
        """
        super(FCN_RNN3, self).__init__()
        
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
        conv_fs = tf.keras.layers.Conv2D(kernel_size=1, strides=1, filters=32, padding='same', activation=activation)
        self.__setattr__('conv_fs', conv_fs)
        
        
        conv_d1 = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=64, padding='same', activation=activation)
        conv_a1 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation=activation)
        conv_d2 = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=64, padding='same', activation=activation)
        conv_a2 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=64, padding='same', activation=activation)
        
        conv_u2 = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=32, padding='same', activation=activation)
        conv_b2 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)
        
        conv_u1 = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=32, padding='same', activation=activation)
        conv_b1 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)
        
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
        conv_f = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation) #'sigmoid' 21/3/22
        self.__setattr__('conv_f', conv_f)
        
           
        
    def call(self, inputs, state):
        # INPUTS
        # state:arrival times,  inputs:weather series
        
        # OUTPUTS
        # state:arrival times 
        
        # state is a tuple if passed from RNN
        #if type(state) == tuple:
        #    state = state[0] # 
        
        arrival = state[0] # need to output an updated version of the arrival component of state
        #arrival = tf.clip(arrival_full, clip_value_min=0.0, clip_value_max=1.0) # clip values
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
        
        weather_shape = tf.ones([tf.shape(arrival)[1],
                tf.shape(arrival)[2],
                self.__getattribute__('weather_filters')
                                ])
        
        climate_shape = tf.ones([tf.shape(arrival)[1],
                tf.shape(arrival)[2],
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
        
        
        slices = 4
        for j in range(slices): # j is a slice of time
        
            frac = (j+0.5)/slices # fraction of init/final weather to mix

            w = w_o * frac + w_f * (1-frac)

            w = weather_1(w)
            w = weather_3(w) # skip weather_2 for simplicity
            w = tf.expand_dims(w, 1)
            w = tf.expand_dims(w, 1)

            #w_expanded = tf.math.multiply(tf.ones(tf.shape(arrival)), w)
            w_expanded = tf.math.multiply(weather_shape, w)


            concat_1 = self.__getattribute__('concat_1')
            x = concat_1([arrival, w_expanded, c_expanded, static_spatial])

            # apply feature engineering layer to concated features (50 filters)
            conv_c = self.__getattribute__('conv_c')
            x = conv_c(x)
            
            ##x = tf.keras.layers.Concatenate(axis=-1)([arrival, x])

            ##x = dropout_1(x)


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
            
            d2 = conv_d2(d1)
            d2 = conv_a2(d2)
            
            
            # upscale
            u2 = conv_u2(d2)
            u2 = tf.keras.layers.Concatenate(axis=-1)([u2, d1])
            u2 = conv_b2(u2)
            
            u1 = conv_u1(u2)
            u1 = tf.keras.layers.Concatenate(axis=-1)([u1, x])
            x = conv_b1(u1)
    
            # THERE IS A DISCONNECT ABOVE. u1 is not passed on, instead x is, missing all of the U-net component
                    
            #conv_f1 = self.__getattribute__('conv_f1')
            #x = conv_f1(x)
                
            conv_f = self.__getattribute__('conv_f')
            x = conv_f(x)
        
            # residual compoent
        
            # update fire arrival latent values
            arrival = x + arrival # residual type component
            
            #
            arrival = tf.where(arrival > 0.99, 1.0, arrival) # clip values
            arrival = tf.where(arrival < 0.001, 0.0, arrival)
            #arrival = tf.clip_by_value(arrival, clip_value_min=0.0, clip_value_max=1.0)

            
            
            # end of weather slice loop
        
        
        # new_state output is a tuple
        new_state = (arrival, static_spatial, climate)
        output = arrival
            
        return output, new_state    
    
    
    
class FCN_RNN2(tf.keras.layers.Layer):
    
    
    
    def __init__(self, activation='sigmoid', apply_dropout=True, apply_batchnorm=True):
        """
        The minimum number of filters are used on the outer layers. The number of filters double each sucessive layer
        towards the middle until the max_filters limit is reached.
        """
        super(FCN_RNN2, self).__init__()
        
        # ListWrapper([InputSpec(shape=(None, None, None, 16), ndim=4), InputSpec(shape=(None, None, None, 32), ndim=4)]
        
        self.state_size = (None, None, None, None)
        #self.dim = dim
        
        # define layers
        
        final_weather_filters = 16
        weather_1 = tf.keras.layers.Dense(10, activation=activation)
        weather_2 = tf.keras.layers.Dense(16, activation=activation)
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
        conv_c = tf.keras.layers.Conv2D(kernel_size=1, strides=1, filters=50, padding='same', activation=activation)
        self.__setattr__('conv_c', conv_c)
        
        conv_d1 = tf.keras.layers.Conv2D(kernel_size=3, dilation_rate=1, filters=10, padding='same', activation=activation)
        conv_d2 = tf.keras.layers.Conv2D(kernel_size=3, dilation_rate=2, filters=10, padding='same', activation=activation)
        conv_d3 = tf.keras.layers.Conv2D(kernel_size=3, dilation_rate=3, filters=10, padding='same', activation=activation)
        conv_d4 = tf.keras.layers.Conv2D(kernel_size=3, dilation_rate=4, filters=10, padding='same', activation=activation)
        conv_d5 = tf.keras.layers.Conv2D(kernel_size=3, dilation_rate=5, filters=10, padding='same', activation=activation)
        self.__setattr__('conv_d1', conv_d1)
        self.__setattr__('conv_d2', conv_d2)
        self.__setattr__('conv_d3', conv_d3)
        self.__setattr__('conv_d4', conv_d4)
        self.__setattr__('conv_d5', conv_d5)
        
        x1_x2 = tf.keras.layers.add
        x2_x3 = tf.keras.layers.add
        x3_x4 = tf.keras.layers.add
        x4_x5 = tf.keras.layers.add
        self.__setattr__('x1_x2', x1_x2)
        self.__setattr__('x2_x3', x2_x3)
        self.__setattr__('x3_x4', x3_x4)
        self.__setattr__('x4_x5', x4_x5)

        
        

        
        
        dropout_1 = tf.keras.layers.Dropout(0.25)
        dropout_2 = tf.keras.layers.Dropout(0.25)
        self.__setattr__('dropout_1', dropout_1)
        self.__setattr__('dropout_2', dropout_2)
        
        concat_3 = tf.keras.layers.Concatenate(axis=-1)
        self.__setattr__('concat_3', concat_3)
        
        
        smoother = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', activation=activation)
        self.__setattr__('smoother', smoother)
        
        # number of filters in conv_f must match number of filteres in latent arrival time
        conv_f = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation)
        self.__setattr__('conv_f', conv_f)
        
        
        
        
    def call(self, inputs, state):
        # INPUTS
        # state:arrival times,  inputs:weather series
        
        # OUTPUTS
        # state:arrival times 
        
        # state is a tuple if passed from RNN
        #if type(state) == tuple:
        #    state = state[0] # 
        
        arrival = state[0] # need to output an updated version of the arrival component of state
        landclass = state[1] # output the latent landclass component, scalar
        heightmap_x = state[2] # output the latent heightmap component, vector
        heightmap_y = state[3]
        w = inputs
        
        # add a couple of dense layers to weather 
        
        #w = tf.keras.layers.Dense(10, activation='relu')(w)
        #w = tf.keras.layers.Dense(16, activation='relu')(w)
        #w = tf.keras.layers.Dense(final_weather_filters, activation='relu')(w)
        
        
        weather_1 = self.__getattribute__('weather_1')
        weather_2 = self.__getattribute__('weather_2')
        weather_3 = self.__getattribute__('weather_3')
        
        w = weather_1(w)
        w = weather_2(w)
        w = weather_3(w)
        w = tf.expand_dims(w, 1)
        w_inputs = tf.expand_dims(w, 1)
        
        # instantiate dropout layers
        dropout_1 = self.__getattribute__('dropout_1')
        dropout_2 = self.__getattribute__('dropout_2')

        
        
        # Start of weather interpolation
        
        
        
        w_expanded = tf.math.multiply(tf.ones(tf.shape(arrival)), w_inputs)
        
        #concat = self.__getattribute__('concat_1')
        #features = concat([w_expanded, topo])
        #conv_features = self.__getattribute__('conv_features')
        #features = conv_features(features)

        
        concat_1 = self.__getattribute__('concat_1')
        x = concat_1([arrival, w_expanded, landclass, heightmap_x, heightmap_y])
       
        # apply feature engineering layer to concated features (50 filters)
        conv_c = self.__getattribute__('conv_c')
        x = conv_c(x)
        
        x = dropout_1(x)
        
        
        # Start of dilated pyramid structure
        
        # apply dilated convolutions / pyramid structure
        conv_d1 = self.__getattribute__('conv_d1') 
        conv_d2 = self.__getattribute__('conv_d2') 
        conv_d3 = self.__getattribute__('conv_d3') 
        conv_d4 = self.__getattribute__('conv_d4') 
        conv_d5 = self.__getattribute__('conv_d5') 
        
        x1 = conv_d1(x) # 10 filters
        x2 = conv_d2(x) # 10 filters
        x3 = conv_d3(x) # 10 filters
        x4 = conv_d4(x) # 10 filters
        x5 = conv_d5(x) # 10 filters
        
        # apply HFF *hierarchical feature fusion to consolodate pyramids
        
        x1_x2 = self.__getattribute__('x1_x2')
        x2_x3 = self.__getattribute__('x1_x2')
        x3_x4 = self.__getattribute__('x1_x2')
        x4_x5 = self.__getattribute__('x1_x2')
        
        x1x2 = x1_x2([x1, x2])
        x2x3 = x2_x3([x1x2, x3])
        x3x4 = x3_x4([x2x3, x4])
        x4x5 = x4_x5([x3x4, x5])
        
        concat_2 = self.__getattribute__('concat_2')
        pyramid_out = concat_2([x1, x1x2, x2x3, x3x4, x4x5])
        
        
        
        # residual component
        conv_f = self.__getattribute__('conv_f')
        smoother = self.__getattribute__('smoother')
        x = smoother(pyramid_out)
        x = conv_f(x)
        
        ###x = dropout_2(x)
        
        arrival_new = x + arrival # residual type component
        
        
        # new_state output is a tuple
        new_state = (arrival_new, landclass, heightmap_x, heightmap_y)
        output = arrival_new
            
        return output, new_state

    
    
class FCN_RNN(tf.keras.layers.Layer):
    
    
    
    def __init__(self, activation='sigmoid', apply_dropout=True, apply_batchnorm=True):
        """
        The minimum number of filters are used on the outer layers. The number of filters double each sucessive layer
        towards the middle until the max_filters limit is reached.
        """
        super(FCN_RNN, self).__init__()
        
        # ListWrapper([InputSpec(shape=(None, None, None, 16), ndim=4), InputSpec(shape=(None, None, None, 32), ndim=4)]
        
        self.state_size = (None, None)
        #self.dim = dim
        
        # define layers
        
        final_weather_filters = 32
        weather_1 = tf.keras.layers.Dense(10, activation=activation)
        weather_2 = tf.keras.layers.Dense(16, activation=activation)
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
        
        
        conv_x3 = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)
        conv_x5 = tf.keras.layers.Conv2D(kernel_size=5, strides=1, filters=32, padding='same', activation=activation)
        conv_x9 = tf.keras.layers.Conv2D(kernel_size=3, dilation_rate=3, filters=32, padding='same', activation=activation)
        self.__setattr__('conv_x3', conv_x3)
        self.__setattr__('conv_x5', conv_x5)
        self.__setattr__('conv_x9', conv_x9)
        
        
        down_1 = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=128, padding='same', activation=activation)
        up_1   = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=32, padding='same', activation=activation)
        self.__setattr__('down_1', down_1)
        self.__setattr__('up_1', up_1)
        
        dropout_1 = tf.keras.layers.Dropout(0.25)
        dropout_2 = tf.keras.layers.Dropout(0.25)
        self.__setattr__('dropout_1', dropout_1)
        self.__setattr__('dropout_2', dropout_2)
        
        concat_3 = tf.keras.layers.Concatenate(axis=-1)
        self.__setattr__('concat_3', concat_3)
        
        conv_c = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)
        # number of filters in conv_f must match number of filteres in latent arrival time
        conv_f = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation)
        
        self.__setattr__('conv_f', conv_f)
        self.__setattr__('conv_c', conv_c)
        
        
    def call(self, inputs, state):
        # INPUTS
        # state:arrival times,  inputs:weather series
        
        # OUTPUTS
        # state:arrival times 
        
        # state is a tuple if passed from RNN
        #if type(state) == tuple:
        #    state = state[0] # 
        
        arrival = state[0] # need to output an updated version of the arrival component of state
        topo = state[1] # need to output the same fixed topo component of state
        w = inputs
        
        # add a couple of dense layers to weather 
        
        #w = tf.keras.layers.Dense(10, activation='relu')(w)
        #w = tf.keras.layers.Dense(16, activation='relu')(w)
        #w = tf.keras.layers.Dense(final_weather_filters, activation='relu')(w)
        
        
        weather_1 = self.__getattribute__('weather_1')
        weather_2 = self.__getattribute__('weather_2')
        weather_3 = self.__getattribute__('weather_3')
        
        w = weather_1(w)
        w = weather_2(w)
        w = weather_3(w)
        w = tf.expand_dims(w, 1)
        w_inputs = tf.expand_dims(w, 1)
        
        
        #w_expanded = tf.math.multiply(topo, w_inputs)
        
        w_expanded = tf.math.multiply(tf.ones(tf.shape(topo)), w_inputs)
        
        #concat = self.__getattribute__('concat_1')
        #features = concat([w_expanded, topo])
        #conv_features = self.__getattribute__('conv_features')
        #features = conv_features(features)

        
        concat_2 = self.__getattribute__('concat_2')
        x = concat_2([arrival, w_expanded, topo])
       
        down_1 = self.__getattribute__('down_1')
        up_1 = self.__getattribute__('up_1')
    
        # apply a dropout
        
        # apply downscale
        
        x = down_1(x)
        
        # apply convolutional transformations
        
        conv_x3 = self.__getattribute__('conv_x3')
        #conv_x9 = self.__getattribute__('conv_x9')
        conv_x5 = self.__getattribute__('conv_x5')
       
        x = conv_x5(x)
        x = conv_x3(x)
        
        # apply upscale
        
        x = up_1(x)
        
        #x3 = conv_x3(x)
        #x9 = conv_x9(x)
        
        #concat_3 = self.__getattribute__('concat_3')
        #x = concat_3([x3, x9])
        
        # consolidate 
        
        conv_c = self.__getattribute__('conv_c')
        x = conv_c(x)
        
        # ensure 'x' has same depth as arrival map
        
        conv_f = self.__getattribute__('conv_f')
        x = conv_f(x)
        
        arrival_new = x + arrival # residual type component
        
        
        # new_state output is a tuple
        new_state = (arrival_new, topo)
        output = arrival_new
            
        return output, new_state
    

    
def cnn_rnn_model(activation='relu'):
    """
    Takes tensorflow tensor objects as a list of inputs. 
    Inputs = [arrival, heightmap, landclass]
    """
    arrival = tf.keras.layers.Input(shape=[IMAGE_WIDTH,IMAGE_HEIGHT,1], name='arrival')
    weather = tf.keras.layers.Input(shape=[TIME_STEPS, 8], name='weather_inputs')
    climate = tf.keras.layers.Input(shape=[2], name='climate_inputs')
    heightmap = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 2], name='heightmap_inputs')
    landclass = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 6], name='landclass_inputs')
        
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
        
        
    if True: # old downscale gradients

        heightmap_x = tf.expand_dims(heightmap[..., 0], -1)
        heightmap_y = tf.expand_dims(heightmap[..., 1], -1)

        hm_down_1x = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=2, padding='same', activation=activation)
        hm_down_2x = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=4, padding='same', activation=activation)
        hm_down_3x = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=8, padding='same', activation=activation)

        latent_hm_x = hm_down_1x(heightmap_x)
        latent_hm_x = hm_down_2x(latent_hm_x)
        latent_hm_x = hm_down_3x(latent_hm_x)

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
    CNNCell_1 = FCN_RNN3(activation=activation)
    CNNRNN_1  = RNN(CNNCell_1, return_sequences=False, name='P2P_RNN') # return all arrival time states
    # output from the recurrent layer
    output = CNNRNN_1(w_inputs, latent_state)

    
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
    
  


    # sum over all arrival time states
    #summed_output = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=-4), name='reduce_sum')(output)
    
    
    
    model = tf.keras.Model(inputs=[arrival, w_inputs, climate, heightmap, landclass], outputs=arrival_final, name='rnn_model')
    
    return model    
    

    
        
        
            

        
        
        
        
        



def resnet_model(activation='tanh'):
    """
    This model extracts features from landclass, heightmap, and weather. 
    A residual cell comprises inputs of these 'static' features, as well as the initial arrival image.
    An updated arrival image is produces. 
    Repeated calls are made to the residual cell, which attempts to generate the final arrival image
    """
    
    
    # HELPER LAYERS
    reflect_pad   = tf.keras.layers.Lambda(lambda x: tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), 'REFLECT'))
    leaky_20 = tf.keras.layers.LeakyReLU(alpha=.2)
   
    # INPUT LAYERS
    
    past_arrival = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 1], name='arrival_inputs')
    weather = tf.keras.layers.Input(shape=[TIME_STEPS, 10], name='weather_inputs')
    heightmap = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 2], name='heightmap_inputs')
    landclass = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 4], name='landclass_inputs')
    
   
    # FEATURE ENGINEERING
    weather_spatial_dimension = 8 

    # create spatial feature stack
    fs = tf.keras.layers.Concatenate(axis=-1, name='spatial_concat')([heightmap, landclass])
    fs = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=8, padding='same', activation=activation)(fs)
    fs = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=8, padding='same', activation=activation)(fs)
    
    # create weather features
    
    w = weather[:,0,:] # extract the first input, process needs iteration (later) (ie some sort of broadcast across t slices)
    w = tf.keras.layers.Dense(10, activation=activation)(w)
    w = tf.keras.layers.Dense(16, activation=activation)(w)
    w = tf.keras.layers.Dense(weather_spatial_dimension, activation=activation)(w)
    w = tf.expand_dims(w, 1)
    w = tf.expand_dims(w, 1)
    
    
    # downscale spatial features
    
    f_down_1a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=8, padding='same', activation=activation)
    f_down_1b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=8, padding='same', activation=activation)
    f_down_2a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=16, padding='same', activation=activation)
    f_down_2b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation)
    f_down_3a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=32, padding='same', activation=activation)
    f_down_3b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)
    
    latent_fs = f_down_1a(fs)
    latent_fs = f_down_1b(latent_fs)
    latent_fs = f_down_2a(latent_fs)
    latent_fs = f_down_2b(latent_fs)
    latent_fs = f_down_3a(latent_fs)
    latent_fs = f_down_3b(latent_fs)
                 
    # downscale arrival features
    
    a_down_1a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=8, padding='same', activation=activation)
    a_down_1b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=8, padding='same', activation=activation)
    a_down_2a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=16, padding='same', activation=activation)
    a_down_2b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation)
    a_down_3a = tf.keras.layers.Conv2D(kernel_size=4, strides=2, filters=32, padding='same', activation=activation)
    a_down_3b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)
    
    at = past_arrival
    latent_at = a_down_1a(at)
    latent_at = a_down_1b(latent_at)
    latent_at = a_down_2a(latent_at)
    latent_at = a_down_2b(latent_at)
    latent_at = a_down_3a(latent_at)
    latent_at = a_down_3b(latent_at)
        
    
    # JOIN WEATHER AND LATENT SPATIAL FEATURES
    
    w_fs = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=weather_spatial_dimension, padding='same', activation=activation)(latent_fs)
    w_fs = tf.math.multiply(w_fs, w)
    
    
    
    # UPDATE THE ARRIVAL FEATURES
    
    x = tf.keras.layers.Concatenate(axis=-1, name='latent_concat')([latent_at, latent_fs, w_fs])
    x = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)(x)
    x = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same', activation=activation)(x)
    
    
    # APPLY RESIDUAL STEP 
    
    x = latent_at + x
    
    # UPSCALE
    
    a_up_1a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=16, padding='same', activation=activation)
    a_up_1b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same', activation=activation)
    a_up_2a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=8, padding='same', activation=activation)
    a_up_2b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=8, padding='same', activation=activation)
    a_up_3a = tf.keras.layers.Conv2DTranspose(kernel_size=4, strides=2, filters=4, padding='same', activation=activation)
    a_up_3b = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=4, padding='same', activation=activation)
    
    x = a_up_1a(x)
    x = a_up_1b(x)
    x = a_up_2a(x)
    x = a_up_2b(x)
    x = a_up_3a(x)
    x = a_up_3b(x)
    
    output = tf.keras.layers.Conv2D(kernel_size=3, strides=1, filters=1, padding='same', activation=activation)(x)
    

    
    

    model = tf.keras.Model(inputs=[past_arrival, weather, heightmap, landclass], outputs=output, name='resnet_model_3')        
    return model








