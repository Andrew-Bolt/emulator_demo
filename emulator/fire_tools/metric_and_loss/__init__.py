import tensorflow as tf
import numpy as np

discard_pad = 32


# define a function composition. This allows the output of a padding function to be passed to a loss function
compose =  lambda loss: lambda pad: lambda *args, **kwargs: loss(*pad(*args, **kwargs))


def target_padding(y, y_pred):
    """pads the predicted image with the target image when the target image > predicted image.
    This is useful in training situations, and has the effect of forcing the predictions to be 0 around
    the margins"""
    
    if len(y.shape) == 2:
        
        #y_pred = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        #y      = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
        
        y_pred_top = y_pred[:discard_pad, :]
        y_top      = y[:discard_pad, :]
        top        = tf.math.maximum(y_pred_top, y_top)
        
        y_pred_bot = y_pred[-discard_pad:, :]
        y_bot      = y[-discard_pad:, :]
        bot        = tf.math.maximum(y_pred_bot, y_bot)
        
        y_pred_lef = y_pred[discard_pad:-discard_pad, :discard_pad]
        y_lef      = y[discard_pad:-discard_pad, :discard_pad]
        lef        = tf.math.maximum(y_pred_lef, y_lef)
        
        y_pred_rig = y_pred[discard_pad:-discard_pad, -discard_pad:]
        y_rig      = y[discard_pad:-discard_pad, -discard_pad:]
        rig        = tf.math.maximum(y_pred_rig, y_rig)
        
        # crop prediction
        y_pred     = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        
        # add padddings
        
        y_pred     = tf.concat([lef, y_pred, rig], axis=1)
        y_pred     = tf.concat([top, y_pred, bot], axis=0)

        
    else:
        #_pred = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        #      = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        
        y_pred_top = y_pred[:,:discard_pad, :]
        y_top      = y[:,:discard_pad, :]
        top        = tf.math.maximum(y_pred_top, y_top)
        
        y_pred_bot = y_pred[:,-discard_pad:, :]
        y_bot      = y[:,-discard_pad:, :]
        bot        = tf.math.maximum(y_pred_bot, y_bot)
        
        y_pred_lef = y_pred[:,discard_pad:-discard_pad, :discard_pad]
        y_lef      = y[:,discard_pad:-discard_pad, :discard_pad]
        lef        = tf.math.maximum(y_pred_lef, y_lef)
        
        y_pred_rig = y_pred[:,discard_pad:-discard_pad, -discard_pad:]
        y_rig      = y[:,discard_pad:-discard_pad, -discard_pad:]
        rig        = tf.math.maximum(y_pred_rig, y_rig)
        
        # crop prediction
        y_pred     = y_pred[:,discard_pad:-discard_pad, discard_pad:-discard_pad]
        
        # add padddings
        
        y_pred     = tf.concat([lef, y_pred, rig], axis=2)
        y_pred     = tf.concat([top, y_pred, bot], axis=1)  
        
    return y, y_pred #returns target (y) and padded prediction (y_pred)
 






def crop_padding(y, y_pred):
    
    if len(y.shape) == 3:
        
        #y_pred = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        #y      = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
        
        y_pred_crop = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        y_crop = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
        
    else:
        #_pred = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        #      = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        
        y_pred_crop = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        y_crop = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        

        
        return y_crop, y_pred_crop #returns target (y) and padded prediction (y_pred_pad)
    
    

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
    

def classifier_regression_mse_loss(y, y_pred):
    # takes the difference in times as a loss, as well as an additional 1.0 loss per pixel if burnt/unburnt
    # is incorrectly classified. (ensures that low values of arrival time are still penalized strongly for mismatch)
    
    tol = 0.01
    
    y_pred = tf.where(y_pred < tol, -1.0, y_pred)
    y = tf.where(y < tol, -1.0, y)
    
    y_init = tf.cast(y > 1-tol, tf.float32)
    y_init = tf.where(y > 1-tol, 1.0, -1.0)
    
    num =  tf.reduce_mean(tf.math.square(y_pred - y))
    den =  tf.reduce_mean(tf.math.square(y_init - y))
    
    loss = tf.math.log((num + 10e-12)/ (den + 10e-12))
    
    return loss
    
    
def classifier_regression_mae_loss(y, y_pred):
    # takes the difference in times as a loss, as well as an additional 1.0 loss per pixel if burnt/unburnt
    # is incorrectly classified. (ensures that low values of arrival time are still penalized strongly for mismatch)
    
    tol = 0.01
    
    y_pred = tf.where(y_pred < tol, -1.0, y_pred)
    y = tf.where(y < tol, -1.0, y)
    
    y_init = tf.cast(y > 1-tol, tf.float32)
    y_init = tf.where(y > 1-tol, 1.0, -1.0)
    
    num =  tf.reduce_mean(tf.math.abs(y_pred - y))
    den =  tf.reduce_mean(tf.math.abs(y_init - y))
    
    loss = tf.math.log((num + 10e-12)/ (den + 10e-12))
    
    return loss    
    
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
        y_pred = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
    else:
        y_pred = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
       
    y_init = tf.cast(y > 0.99, tf.float32)
    den = tf.reduce_sum(tf.math.square(y_init - y))
    num = tf.reduce_sum(tf.math.square(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ den + 10e-12) * nat_to_dec # converted to log_10 
    
    return loss

def l1_relative_loss(y, y_pred):
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
       
    y_init = tf.where(y < 1.0, 0.0, y)
    
    den = tf.reduce_sum(tf.math.abs(y_init - y))
    num = tf.reduce_sum(tf.math.abs(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ (den + 10e-12)) * nat_to_dec # converted to log_10 
    
    return loss
    

def l1_relativel_loss_padded(y, y_pred):
    
    if len(y.shape) == 3:
        
        y_pred = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
        
    else:
        y_pred = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
    
    y_init = tf.where(y < 1.0, 0.0, y)
    
    den = tf.reduce_sum(tf.math.abs(y_init - y))
    num = tf.reduce_sum(tf.math.abs(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ (den + 10e-12)) * nat_to_dec # converted to log_10 
    
    return loss
    

def l1_relative_loss_cropped(y, y_pred):
    
    
    if len(y.shape) == 3:
        #y_pred = y_pred[discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[discard_pad:-discard_pad, discard_pad:-discard_pad]
        paddings = tf.constant([[discard_pad, discard_pad,], [discard_pad, discard_pad], [0,0]])
        y       = tf.pad(y, paddings, mode="CONSTANT", constant_values=0.0)
        
    else:
        #y_pred = y_pred[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        y      = y[:, discard_pad:-discard_pad, discard_pad:-discard_pad]
        paddings = tf.constant([[0,0],[discard_pad, discard_pad,], [discard_pad, discard_pad], [0,0]])
        y       = tf.pad(y, paddings, mode="CONSTANT", constant_values=0.0)
    
    
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
       
    y_init = tf.where(y < 1.0, 0.0, y)
    
    den = tf.reduce_sum(tf.math.abs(y_init - y))
    num = tf.reduce_sum(tf.math.abs(y_pred - y))
    nat_to_dec = tf.convert_to_tensor(1/np.log(10), dtype=tf.float32)
    loss = tf.math.log((num + 10e-12)/ (den + 10e-12)) * nat_to_dec # converted to log_10 
    
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
    tol = 0.01
    
    maxval = tf.math.reduce_max(y) # equal to number of intervals

    y_pred = tf.where((y_pred > tol) & (y_pred < maxval-tol), 1.0, 0.0) # area burnt during latest interval
    y      = tf.where((y > tol) & (y < maxval-tol), 1.0, 0.0) # area burnt during latest interval

    Y = y + y_pred 

    intersection = tf.reduce_sum(tf.where(Y == 2, 1.0, 0.0))
    union = tf.reduce_sum(tf.where(Y >= 1, 1.0, 0.0))

    iou = intersection / union
    
    iou = tf.cast(iou, tf.float32)

    return iou

def dice_metric(y, y_pred):
    # dice metric 
    # scaled by change in area
    
    y_pred = tf.convert_to_tensor(y_pred)
    y      = tf.convert_to_tensor(y)
    #y_max  = tf.math.reduce_max(y) #finds largest value in array
    maxval = np.max(y)
    tol = 0.01

    y_pred = tf.where((y_pred > tol) & (y_pred < maxval-tol), 1, 0) # area burnt during latest interval
    y      = tf.where((y > tol) & (y < maxval-tol), 1, 0) # area burnt during latest interval
    
    Y = y + y_pred 
    
    intersection = tf.reduce_sum(tf.where(Y == 2, 1, 0))
    sum_set_size = tf.reduce_sum(Y)
    
    dice = 2*intersection / sum_set_size
    
    return dice




def iou(y, y_pred, y_init): # used to find IOU for multiple interval simulations
    tol = 0.001
    
    y_init = tf.convert_to_tensor(y_init)
    y_init = y_init > tol # create binary mask
    y_init = tf.cast(y_init, tf.float32)
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = y_pred - y_init > tol # create binary mask, with hole for init
    y_pred = tf.cast(y_pred, tf.int32)
    
    y = tf.convert_to_tensor(y)
    y = y - y_init > tol # create binary mask, with hole for init
    y = tf.cast(y, tf.int32)
    
    Y = y + y_pred
    
    intersection = tf.reduce_sum(tf.where(Y == 2, 1.0, 0.0))
    union = tf.reduce_sum(tf.where(Y >= 1, 1.0, 0.0))
    
    iou = intersection / union
    
    iou = tf.cast(iou, tf.float32)

    
    return iou
    
    

def iou_loss(y, y_pred):
    
    iou = iou_metric(y, y_pred)
    
    loss = tf.math.log(1.0-iou)
    
    return loss

    

   
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
