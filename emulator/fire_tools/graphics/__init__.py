from PIL import Image
from PIL.PngImagePlugin import PngInfo
from copy import copy
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from skimage import measure
from skimage.draw import polygon_perimeter

axis_fontsize=16
title_fontsize=18
legend_fontsize=14

def plot_winds(weather, name='winds.pdf', figsize=(7, 6), cmap = 'plasma_r', meta=None, scaling=1, show=True):
    
    """ creates a scatter plot of wind directions (x,y) with color as the time dimension. 
    inputs are the weather variable from the dataset in the form (timesteps, 8). The 8 
    features are wind_x_0, wind_x_f, wind_y_0, wind_y_f, ...
    
    scaling is used to convert from normalized values back into km/h
    """
    
    x_wind = weather[:,0] # collect all initial winds
    x_wind = np.append(x_wind, weather[-1,1]) # append last final wind
    x_wind = x_wind*scaling
    
    y_wind = weather[:,2]
    y_wind = np.append(y_wind, weather[-1,3])
    y_wind = y_wind * scaling

    t_ = range(len(y_wind))

    plt.figure(figsize=figsize)
    #plt.imshow(heightmap, cmap='plasma', origin='upper')
    plt.scatter(0, 0, s=50, c='g', marker='s')
    plt.scatter(x_wind, y_wind, s=30, c=t_, cmap = cmap)
    plt.colorbar().set_label('Interval', rotation=270)
    
    if scaling==1:
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.grid(True, 'both')
        plt.title('Normalized Wind Direction', fontsize=title_fontsize)
    else:
        plt.xlim([-scaling, scaling])
        plt.ylim([-scaling, scaling])
        plt.grid(True, 'both')
        plt.title('Wind Direction (km/h)', fontsize=title_fontsize)
        
    
    plt.savefig(name, matadata=meta, bbox_inches='tight', meta=None)
    
    if show:
        plt.show()
    

    
def plot_temp(weather, name='temperature.pdf', figsize=(12,6), meta=None, t_scaling=1, t_offset=0, show=True):
    
    temp = weather[:, 4] # initial temperatures
    temp = np.append(temp, weather[-1, 5]) # append last temp 
    temp = temp*t_scaling + t_offset
    
    
    rh   = weather[:, 6] # initial humidity
    rh   = np.append(rh, weather[-1, 7]) # append last humidity

    # create plot object
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(temp, '.', color='red')#, fontsize=legend_fontsize)
    ax.set_xlabel('Interval', fontsize=16)
    ax.set_ylabel('Temperature', fontsize=16, color='red')
    ax.set_ylim([t_offset, t_offset+t_scaling])

    ax2 = ax.twinx()
    ax2.plot(rh, '.', color='blue')
    ax2.set_ylabel('Relative Humidity', fontsize=16, color='blue')
    ax2.set_ylim([0, 1])
    
    if t_scaling == 1:
        plt.title('Normalized Temperature and Relative Humidity', fontsize=title_fontsize)  
    else:
        plt.title('Temperature and Relative Humidity', fontsize=title_fontsize) 
    
    plt.savefig(name, metadata=meta, bbox_inches='tight')
    
    if show:
        plt.show()
    
    
def make_image(img, interval, shading=0.25, land_rgba=None):

    img = np.where(img < interval, 0, img)
    # make previous segments transparent/white
    ##img = np.where(img > interval + 1, 0, img) #this actually controls chase/stay
     
    arr = cm.hot_r(img - interval, bytes=True) # hot

    # add transparency
    alpha = np.where(img < 0.01, 0, 255)

    # if using a background, only show the moving perimeter as solid
    shade = max(int(256 * shading) - 1, 0)
    alpha = np.where(img - interval >= 1, shade, alpha) # shade prev burn
    alpha = np.where(img > img.max()-0.01, shade, alpha) # shade out initial location
    arr[:,:,-1] = alpha

    rgb_img = Image.fromarray(arr, mode="RGBA")   

    if land_rgba is not None:
        # combine background and arrival image
        combined = Image.alpha_composite(land_rgba, rgb_img) # image2 OVER image1

        rgb_img= combined
        
        
    return rgb_img
    
    
def make_fire_gif(img, name='chloropleth', shading=0.25, duration=30, bg_img_arr=None, beta=0.3, meta=None):
    
    """
    img: image array
    name: name of output file, shading: transparency of previous burn area, duration: speed of GIF
    background_img: PIL image, typically a converted background array, beta: background watermarking
    metadata: a list of key: value pairs providing info about the run (eg. IOU, loss, duration etc.)
    """
    
    height, width = img.shape
    max_val = np.max(img)
    
    time_slices = np.arange(max_val, 0, -0.1) # increments of 0.1 * interval (ie 3 min)
    
    land_rgba = None
    
    if bg_img_arr is not None:
        
        # white wash the background image
        land_arr = (beta*bg_img_arr + (1-beta)*255).astype('uint8')
        land_arr[:,:,-1] = 255 # solid alpha channel
        
        # convert to an image type
        land_rgba = Image.fromarray(land_arr, mode="RGBA")

    #gif_img = Image.new('RGBA', (width, height))
    
    frames = []
    
    for time_slice in time_slices:
        
        frame = make_image(img, time_slice, shading=shading, land_rgba=land_rgba)
        frames.append(frame)
    
    # collect and save some metadata
    metadata = PngInfo()
    if meta is not None:
        for key, value in meta.items():
            metadata.add_text(str(key), str(value))
    
    
    if bg_img_arr is None:
            frames[0].save(name, format='GIF', append_images=frames[1:], 
                   save_all=True, duration=duration, loop=0, disposal=2, transparency=0)
    else:
        frames[0].save(name, format='GIF', append_images=frames[1:], 
                   save_all=True, duration=duration, loop=0, disposal=2)
    
    return None




    
def make_contour_image(img_list, color_list=None, interval=0, line_width=3, land_rgba=None):
    
    #line_width=3 # pixel width
    line_width = 3
    lw = line_width - 1
    line_rgba = ([1,0,0,1], [0,0,1,1], [0,0.7,0,1])# red, blue, green
    
    
    
    shape = list(img_list[0].shape)
    shape.append(4)
    arr = np.zeros(shape)
    
    for idx, img in enumerate(img_list):
        
        arr_0 = np.zeros(img.shape)
    
        contours = measure.find_contours(img, interval)
        for contour in contours:

            rr, cc = polygon_perimeter(contour[:,0], contour[:,1])
            if lw <= 1:
                arr_0[rr, cc] = 1
            else:
                for r, c in zip(rr, cc):
                    arr_0[r-lw:r+lw, c-lw:c+lw] = 1
        
        arr_0 = np.expand_dims(arr_0, -1)
        
        # color the line
        if color_list is None:
            arr_0 = arr_0 * np.array(line_rgba[idx])
        else:
            if color_list[idx] == 'r':
                line_color = [1,0,0,0.3]
            elif color_list[idx] == 'b':
                line_color = [0,0,1,1]
            elif color_list[idx] == 'g':
                line_color = [0,0.7,0,1]
            else:
                line_color = [1,0,0,1] # default red
            arr_0 = arr_0 * np.array(line_color)
        
        arr   += arr_0 * np.expand_dims((1- arr[:,:,-1]), -1) # ignore pixel update if previous value already set.
                    
                    
    #arr = cm.hot_r(arr_0, bytes=True) # we want to use a solid color for lines, not a colormap
    #alpha = arr_0 # invert, alpha is 1 where there is no line
    #arr[:,:,-1] = alpha
    
    
    ##arr_0 = np.expand_dims(arr_0, -1)
    
    ##arr = arr_0 * line_rgba
    
    
    
    rgb_img = Image.fromarray(np.uint8(arr*255), mode="RGBA") # convert to uint on range 0-255


    if land_rgba is not None:

        # combine background and arrival image
        combined = Image.alpha_composite(land_rgba, rgb_img) # image2 OVER image1

        rgb_img = combined
        
    
    return rgb_img


def make_contour_vid(img_list, color_list=None, name='perimeter', line_width=2, duration=30, bg_img_arr=None, beta=0.3):
     
    """
    img: tuple of image arrays
    name: name of output file, shading: transparency of previous burn area, duration: speed of GIF
    background_img: PIL image, typically a converted background array, beta: background watermarking
    """
    
    height, width = img_list[0].shape
    max_val = np.max(img_list[0])    
    
    if bg_img_arr is not None:
        
        # white wash the background image
        land_arr = (beta*bg_img_arr + (1-beta)*255).astype('uint8')
        land_arr[:,:,-1] = 255 # solid alpha channel
        
        # convert to an image type
        land_rgba = Image.fromarray(land_arr, mode="RGBA")
    else: 
        land_rgba = None
    
    
    time_slices = np.arange(max_val, 0, -0.1) # increments of  0.1 * interval (ie 3 min)
    
    
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
    
    for time_slice in time_slices:
        
        frame = make_contour_image(img_list, color_list, time_slice, line_width=3, land_rgba=land_rgba)
        
        image = np.array(frame)
        image = image[:,:,:-1] # erase alpha channel
        
        image = image[..., ::-1] # change from RGB to BGR for CV format
        
        out.write(image)
        
    out.release()
    
    return None
 


def make_contour_gif(img_list, name='perimeter', line_width=3, duration=30, bg_img_arr=None, beta=0.3):
    #ft.graphics.make_contour_gif(a_pred, name='test_c_gif', duration=30, bg_img_arr=land_arr, beta=0.3)
    
    width, height = img_list[0].shape
    max_val = np.max(img_list[0])    
    
    if bg_img_arr is not None:
        
        # white wash the background image
        land_arr = (beta*bg_img_arr + (1-beta)*255).astype('uint8')
        land_arr[:,:,-1] = 255 # solid alpha channel
        
        # convert to an image type
        land_rgba = Image.fromarray(land_arr, mode="RGBA")
    else: 
        land_rgba = None
    
    
    # collect and save some metadata
    metadata = PngInfo()
    if meta is not None:
        for key, value in meta.items():
            metadata.add_text(str(key), str(value))
    
    
    time_slices = np.arange(max_val, 0, -0.1) # increments of  0.1 * interval (ie 3 min)
    
    frames = []
    
    for time_slice in time_slices:
        
        frame = make_contour_image(img_list, time_slice, line_width=3, land_rgba=land_rgba)
        frames.append(frame)
       
    if land_rgba is not None:
        frames[0].save(name, format='GIF', append_images=frames[1:], 
               save_all=True, duration=duration, loop=0, disposal=2)
    else:
        frames[0].save(name, format='GIF', append_images=frames[1:], 
               save_all=True, duration=duration, loop=0, transparency=0, disposal=2)
    
    return None
 
    
def make_fire_vid(img, name='chloropleth', shading=0.25, duration=30, bg_img_arr=None, beta=0.3):
     
    """
    img: array
    name: name of output file, shading: transparency of previous burn area, duration: speed of GIF
    background_img: PIL image, typically a converted background array, beta: background watermarking
    """
    
    height, width = img.shape
    max_val = np.max(img)
    
    time_slices = np.arange(max_val, 0, -0.1) # increments of 0.1 * interval (ie 3 min)
    
    land_rgba = None
    
    if bg_img_arr is not None:
        
        # white wash the background image
        land_arr = (beta*bg_img_arr + (1-beta)*255).astype('uint8')
        land_arr[:,:,-1] = 255 # solid alpha channel
        
        # convert to an image type
        land_rgba = Image.fromarray(land_arr, mode="RGBA")
        
    
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
    
    for time_slice in time_slices:
        
        frame = make_image(img, time_slice, shading=shading, land_rgba=land_rgba)
        
        image = np.array(frame)
        image = image[:,:,:-1] # erase alpha channel
        
        image = image[..., ::-1] # change from RGB to BGR for CV format
        
        out.write(image)
        
    out.release()
    
    
    return None
    

    
    
    
def plot_fire(arrival, title='chloropleth', name='fire_plot.pdf', figsize=(12,9), cmap='plasma_r', iou=None, meta=None, show=True):
    
    max_val = np.max(arrival)
    
    arrival = max_val-arrival # invert arrival values to recover chronological order
    
    # set colormap levels
    levels = np.arange(0, max_val, max_val/100) # 100 colors
    #levels[-1] = 1-0.001 # minimum colormap value
    
    # create colormap palette
    palette = copy(plt.get_cmap(cmap))
    #palette.set_under('white', '1.0') # set opaque white for unburnt
    palette.set_over('white', '1.0') 
    
    norm = colors.BoundaryNorm(levels, ncolors=palette.N)
    
    # plot figure
    plt.figure(figsize=figsize)
    plt.imshow(arrival, norm=norm, cmap=palette, origin='upper') #vmax=1
    plt.colorbar(ticks=np.arange(0, max_val, 2))    
    plt.title(title, fontsize=title_fontsize) ## np.where(y_init>0.001, 1, 0)
    #plt.set_zlim([0, max_val])
    
    if iou is not None:
        # add text to display the IOU of the fire    
        plt.xlabel('IOU: {:.2f}'.format(iou), fontsize=axis_fontsize)
    
    plt.savefig(name, metadata=meta, bbox_inches='tight') # saves an output of the plotted image
    
    if show:
        plt.show()
    
    return None
    
    
def plot_fire_difference(arrival_pred, arrival_target, arrival_init=None, name='difference.pdf', figsize=(12,9), cmap='seismic', title='Predicted Minus Target', iou=None, meta=None, show=True):
    
    """ displays the difference between two arrival images (arrival_pred - arrival_target) and draws in a 
        contour for the starting location of the fire, as defined by image arrival_init """
    
    max_val = np.max(arrival_target)
    
    diff = arrival_pred - arrival_target
    plt.figure(figsize=figsize)
    plt.imshow(diff, cmap=cmap, vmin=-max_val, vmax=max_val, origin='upper')
    plt.colorbar()
    plt.title(title, fontsize=title_fontsize)
    
    if arrival_init is not None:
        # plot initial contour
        
        plt.contour(arrival_init, 0, colors='g', alpha=0.7) 
    
    if iou is not None:
        # add text to display the IOU of the fire    
        plt.xlabel('Blue: false negatives, Red: false positives, Green: starting shape \n IOU: {:.2f}'.format(iou), fontsize=axis_fontsize)
    else:
        plt.xlabel('Blue: false negatives, Red: false positives, Green: starting shape', fontsize=axis_fontsize)
    
    plt.savefig(name, metadata=meta, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return None

def landclass_to_image_array(landclass, land_colors=None):
    """ coverts a (width, height, classes) landclass map to a (width, height, 4) RGBA array using supplied colormap"""
    
    num_classes = np.shape(landclass)[-1]
    temp = np.zeros_like(landclass[:,:,0])
    for j in range(num_classes):
        temp = temp + landclass[:,:,j] * (j + 0) # + 1

    landclass = temp / num_classes

    if land_colors is not None:
        cmap = colors.ListedColormap(land_colors)
    else:
        cmap = cm.Pastel1
    
    land_arr = cmap(landclass, bytes=True) # convert to an array
    
    return land_arr
    
    
    

def create_landclass_colormap(landclass, land_colors=None):
   
    num_classes = np.shape(landclass)[-1]
    
    if land_colors is None:
        if num_classes == 7:
            cmap = colors.ListedColormap([[0.2,0.2,1,1], [0,0.8,0,1], [1,1,0,1], [1,0.5,0,1], [0.5,0.5,0,1], [0.7,0,0,1], [0.3,1,0.2,1]])
        else:
            cmap = 'Pastel1'
    else:
        cmap = colors.ListedColormap(land_colors)
        
    return cmap
    
    
def plot_land_classes(landclass, arrival_pred=None, arrival_target=None, arrival_init=None, name='landclasses.pdf', figsize=(12,9), land_names=None, land_colors=None, title='Fire Perimeters and Landclasses', iou=None, meta=None, show=True):
    
    """ Converts landclass from a binary (height, width, channels) to a (height, width) image wih integer values for each 
        type of land. Contours are drawn for initial, predicted and final perimeters if supplied.
        Colors for each type of landclass can be supplied manually as an array of RGB values ie. [[1,0,0], [0.7,0.7,0], ...]
        Names for each landclass can be supplied as an array of strings eg. ['water', 'forest', 'grass' ...]
        """

    num_classes = np.shape(landclass)[-1]

    # convert landclass
    temp = np.zeros_like(landclass[:,:,0])
    for j in range(num_classes):
        temp = temp + landclass[:,:,j] * (j + 0) # + 1

    landclass = temp
    
    
    if False:# moved to create_landclass_colormap function
        if land_colors is None:
            if num_classes == 7:
                cmap = colors.ListedColormap([[0.2,0.2,1,1], [0,0.8,0,1], [1,1,0,1], [1,0.5,0,1], [0.5,0.5,0,1], [0.7,0,0,1], [0.3,1,0.2,1]])
            else:
                cmap = cm.Pastel1
        else:
            cmap = colors.ListedColormap(land_colors)
    
    cmap = create_landclass_colormap(landclass, land_colors)
    
    ## LANDCLASS WITH OVERLAY

    fig = plt.figure(frameon=False)
    plt.figure(figsize=figsize)
    
    plt.imshow(landclass, cmap=cmap, vmin=0, vmax=num_classes-1, alpha=0.3)
    
    ticks = np.linspace(0, num_classes-1, num_classes*2+1)[1::2]
    
    if land_names is None:
        
        # create dummy names
        names = []
        for j in range(num_classes):
    
            names.append('class ' + str(j))
        
        plt.colorbar(ticks=ticks).set_ticklabels(names, update_ticks=True)
    else:
        plt.colorbar(ticks=ticks).set_ticklabels(land_names, update_ticks=True)

        
    x_label = ''
    # plot contours
    if arrival_pred is not None:
        if type(arrival_pred) is list:
            for pred in arrival_pred:
                plt.contour(pred, 0, colors='r', alpha=0.3) # predicted edges
        else:
            plt.contour(arrival_pred, 0, colors='r', alpha=0.7) # predicted edge
        x_label += 'red: predicted. '
    if arrival_target is not None:
        plt.contour(arrival_target, 0, colors='b', linestyle=':', alpha=0.7) # target edge
        x_label += 'blue: target. '
    if arrival_init is not None:
        plt.contour(arrival_init, 0, colors='g', alpha=0.7) # initial edge
        x_label += 'green: initial.'
        
            
    if iou is not None:
        # add text to display the IOU of the fire    
        plt.xlabel(x_label + '\n IOU: {:.2f}'.format(iou), fontsize=axis_fontsize)
    else:
        plt.xlabel(x_label, fontsize=axis_fontsize)
    plt.title(title, fontsize=title_fontsize)
    
    plt.savefig(name, metadata=meta, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return None
    

    