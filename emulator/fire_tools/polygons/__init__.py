import shapely
import numpy as np
import random
import math
import cv2
from shapely.geometry import Point, Polygon, MultiPolygon

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
    areas = [triangle.area for triangle in triangles]

    # sample triangles within polygon, weighted by area
    indices = np.random.choice(np.array(range(0, len(triangles))), N, areas)

    # return a list of sampled triangle polygons
    sampled_triangles = [triangles[ind] for ind in indices]

    # return list of triangle vertices
    vertices = [np.reshape(np.asarray(sample.exterior.xy)[:,0:3], 6) for sample in sampled_triangles]

    # return set of randomly sampled points
    points = [point_in_triangle(pnts[[0,3]], pnts[[1,4]], pnts[[2,5]]) for pnts in vertices]

    return points


def delta_b_loss(polyA, polyB, N=1000, p=2):
    """
    Calculates the binary Baddeley delta loss between two filled polygons
    """

    A_symm_diff_B = polyA.symmetric_difference(polyB)
    A_intersect_B = polyA.intersection(polyB)

    points = sample_points_in_polygon(A_symm_diff_B, 1000)
    distances = [A_intersect_B.distance(shapely.geometry.Point(point)) for point in points]

    excluded_area = A_intersect_B.area
    total_area = A_symm_diff_B.area + A_intersect_B.area
    fractional_area = excluded_area / total_area

    delta_loss = (np.sum(fractional_area * np.array(distances)**p)/N)**(1/p)
    
    # normalize
    delta_loss = delta_loss / total_area

    return delta_loss

def mask_for_polygons(polygons, im_size):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return np.flipud(img_mask)


    
def delta_g_loss(polyA, polyB, polyA0, imageA, imageB, N=1000, p=2):
    """
    Calculates the grayscale Baddeley delta loss between two polygons
    
    Here polyA is the true perimeter at the final interval
    polyB is the predicted perimeter at the final interval
    polyA0 is the true perimeter at the first time interval
    """
    
    binary_loss = delta_b_loss(polyA, polyB, N=N, p=p)
    gray_loss = 0 # extract pixel vales from intersection of polyA, polyB take abs diff & sum
    
    intersect = MultiPolygon([polyA.intersection(polyB)])
    
    C = MultiPolygon([polyA.intersection(polyB)])
    target_polygon = C.difference(polyA0)
    
    mask = mask_for_polygons(target_polygon, imageA.shape)
    
    gray_loss = np.mean(np.multiply(np.abs(imageA - imageB), mask)) / target_polygon.area
    
    
    total_loss = binary_loss + gray_loss
    
    return total_loss
    
    