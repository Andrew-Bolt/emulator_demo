import ntpath
import rasterio
import numpy as np
import pandas as pd
import skimage
from skimage import measure
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon
import random
from PIL import Image, ImageDraw
from tifffile import imsave, imread

class trial_data:
    def __init__(self, arrivaltime, heightmap, landclass, timeseries):
    # takes processed arrivaltimes, heightmaps etc. 
    
        # give the trial a name
        self.name = ntpath.basename(arrivaltime)

        # location of data files
        self.arrivaltime = arrivaltime
        self.heightmap = heightmap
        self.landclass = landclass
        self.timeseries = timeseries

        # spatial information
        with rasterio.open(arrivaltime) as src:
            self.bounds = src.bounds
            self.res = src.res
            self.crs = src.crs
            self.height = src.height
            self.width = src.width
            self.endval = np.max(src.read(1))

        # temporal information
        time_series = pd.read_csv(timeseries)
        self.starttime = time_series.index[0]
        self.endtime = time_series.index[-1]
        self.interval = pd.infer_freq(time_series.index)
        self.intval = np.ceil(self.endval / (len(time_series)-1))
        self.timeindex = time_series.index
        # store time series data (- relatively small filesize)
        self.time_series = time_series
        
    def calculate_perimeters(self):
    # calculates arrival time perimeters for each time period in 'timeseries'
    
        contours = {}
    
        with rasterio.open(self.arrivaltime) as src:
            image = src.read(1)
            # replace large -'ve (NaN) values with large +'ve values
            image = np.where(image < 0, -image, image)
            # re-orient image
            image = np.fliplr(np.transpose(image))
            
            for ind, time in enumerate(self.timeindex):
                
                contour = contour_data(image, time, self.starttime)
                
                #contours.append(contour)
                contours.update({ind:contour})
                
        self.contours = contours
        
    def get_arrivaltime(self):
        # returns the target arrivaltime array
        
        return imread(self.arrivaltime)
        
    def get_heightmap(self):
        # returns a heightmap array
        
        return imread(self.heightmap)
    
    def get_landclass(self):
        # returns the landclass array
        lc = imread(self.landclass) # (channels, width, height)
        lc = np.swapaxes(lc,0,2)
        lc = np.swapaxes(lc,0,1) # (width, height, channels)
        
        return lc
    
    def get_temporal(self, time_interval):
    
        ts = self.time_series.iloc[time_interval:time_interval+2]
        #final_td = self.time_series.iloc[time_interval+1]
        
        return ts #, final_td
        
    
        
    def get_feature_array(landclass_file, heightmap_file):
        """ takes input tif files and returns an array of features suitable for ML processing """
    
        class_array = imread(cwd+'/tmp/' + ntpath.basename(landclass_file))
        with rasterio.open(cwd+'/tmp/' + ntpath.basename(heightmap_file)) as hm:
            height_array = np.expand_dims(hm.read(1),2)
            feature_array = np.concatenate((height_array, class_array), axis=2)
            return feature_array, hm.res

    # THE BELOW DEFINITION NEEDS TO BE COMPLETED
    def get_arrival_window(self, time):
        # returns the arrival time image transformed so that 0 is the initial time and 1 is the final time
        # for the given time interval. Times outside this range are reduced to 0 or 1.
        # time is an interval number
        
        
        at = np.array(imread(self.arrivaltime))
        
        # want to clip over these two values
        #print(self.timeindex[time])
        #print(self.timeindex[time+1])
        #print(self.timeindex)
        
        # 
        at = np.where(at < 0, -at, at)
        
        # set initial time to 0, final time to 1
        at = at / self.intval - time
        
        # clip max and min values
        #clip_mapping = np.vectorize(clipping_func)
        #scaled_at = clip_mapping(at)
        
        at = np.where(at < 0, 0, at)
        at = np.where(at > 1, 1, at)
        
        
        scaled_at = at
        
        return scaled_at
        
        
        
def clipping_func(x):
    #, minval=0, maxval=1
    """
    Takes an array element and transforms it.
    Values below minval are set to minval.
    Values above maxval are set to maxval.
    """
    minval=0
    maxval=1
    
    if x < minval:
        return minval
    elif x > maxval:
        return maxval
        
        
class contour_data:
    def __init__(self, arrivaltime_image, time, start_time):
        """calculates the contour polygons of a timeslice of the arrival time image"""
        
        dt_time = pd.to_datetime(time)
        dt_start_time = pd.to_datetime(start_time)
        total_seconds = (dt_time-dt_start_time).total_seconds()

        # store times
        self.initial_time = dt_time
        self.simulation_seconds = total_seconds
        
        # store image shape
        self.width = arrivaltime_image.shape[0]
        self.height  = arrivaltime_image.shape[1]
        
        # store polygons
        contours = measure.find_contours(arrivaltime_image, total_seconds)
            
        self.shapes = contours
        areas = []
        lengths = []
        is_ccw   = []
        
        
        for shape in contours:
            poly = Polygon(shape)
            
            areas.append(poly.area)
            lengths.append(poly.length)
            is_ccw.append(poly.exterior.is_ccw)
        
        self.areas = areas
        self.lengths = lengths
        self.is_ccw = is_ccw

        

        # returns Multipolygon object
        # contains complex polygons (those with outsides and holes)

        # find ccw and cw shapes
        ccw_array = np.array(self.is_ccw)
        ccw_indices = np.argwhere(ccw_array)
        cw_indices = np.argwhere(ccw_array==False)

        # extract array of contour points
        contours = self.shapes

        outer_poly = [Polygon(contours[x]) for x in ccw_indices[:,0]]
        inner_poly = [Polygon(contours[x]) for x in cw_indices[:,0]]


        # generate polygons with holes, then generate a multipolygon

        polygons = []
        for ind_o, outer in enumerate(outer_poly):
            perimeter = contours[ccw_indices[:,0][ind_o]]
            holes = []
            for ind_i, inner in enumerate(inner_poly):
                if outer.contains(inner):
                    holes.append(contours[cw_indices[:,0][ind_i]])
            polygon = Polygon(perimeter, holes)

            # create complex polygon object
            polygons.append(polygon)

        # create multipolygon object
        multipolygon = MultiPolygon(polygons)

        # save multipolygon object
        self.polygons = multipolygon

    
    
    def get_polygon(self, N):
        # return the Nth polygon
        
        polygons = Polygon(self.shapes[N])
        
        return polygons
   
        
    
    def get_mask(self):
        # want to return a burn/unburnt map for each pixel, based on the contour map
        
        img = Image.new('L', (self.width, self.height), 1) # base 'unburnt' map
        
        zipped = zip(self.shapes, self.is_ccw)
        # sort by orientation, ccw shapes first
        zipped_sorted = sorted(zipped, key=lambda x:-x[1]) 
        
        # need to draw all ccw images first
        for shape, is_ccw in zipped_sorted:
            
            polygon = shape.flatten().tolist()
            fill = 1-is_ccw
            
            # update image drawing
            ImageDraw.Draw(img).polygon(polygon, outline=None, fill=fill)
            
        mask = np.flipud(np.array(img))
        
        return mask
    
    def get_delta_loss(self, contour_data_object, N=1000, p=2, c=5):
        """
        calculate the delta loss (from Badelley et al 1992) between two contour data objects
        """
        
        # generate necessary polygons
        poly_A = self.polygons
        poly_B = contour_data_object.polygons
        
        A_union_B = poly_A.union(poly_B)
        A_symm_diff_B = A.symmetric_difference(B)
        A_intersection_B = A.intersection(B)
        
        img_size = (self.width, self.height)
        
        # want to sample points within A_symm_diff_B and calculate distance to A_union_B
        
        bin_array = skimage.draw.polygon2mask(img_size, poly_union)
        
        
        coords = np.argwhere(bin_array == 1)
        samples = random.choices(coords, k=N)
        
        dist = 0
        
        for coord in samples:
            pnt = Point(coord)
            dist += abs(poly_A.distance(pnt) - poly_B.distance(pnt))**p
            
        dist /= N # normalize
        loss = dist**(1/p)    
        
        return loss

