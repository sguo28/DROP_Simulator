
# from config.settings import LON_WIDTH, LAT_WIDTH, MIN_LAT, MIN_LON, MAP_WIDTH, MAP_HEIGHT, DELTA_LON, DELTA_LAT
import numpy as np
import geopandas as gpd
from scipy.spatial.kdtree import KDTree
# EDGE = 0.42 # km
# DELTA= 2*EDGE #unit KM

hexes=gpd.read_file('./data/NYC_shapefiles/reachable_hexes.shp') # tagged_cluster_hex
hex_kdtree = KDTree(hexes[['lon','lat']])

def convert_hexid_to_lonlat(hex_id):
    lon,lat = hexes.loc[hex_id,['lon', 'lat']]
    return lon, lat
def convert_lonlat_to_hexid(lon,lat):
    
    _,idx=hex_kdtree.query((lon,lat))
    return idx

'''

def convert_lonlat_to_xy(lon, lat):
    x = (1/sin(radians(60)))*(lon - MIN_LON)/DELTA
    x = int(min(max(x, 0), MAP_WIDTH - 1))
    y = (lat - MIN_LAT) / DELTA
    y = int(min(max(y, 0), MAP_HEIGHT - 1))
    return x, y


def convert_xy_to_lonlat(x, y):
    lon = MIN_LON + DELTA_LON * (int(min(max(x, 0), MAP_WIDTH - 1)) + 0.5)
    lat = MIN_LAT + DELTA_LAT * (int(min(max(y, 0), MAP_HEIGHT - 1)) + 0.5)
    return lon, lat

def lon2X(lons):
    X = np.int32(np.minimum(np.maximum((lons - MIN_LON) / DELTA_LON, 0), MAP_WIDTH - 1))
    return X

def lat2Y(lats):
    Y = np.int32(np.minimum(np.maximum((lats - MIN_LAT) / DELTA_LAT, 0), MAP_HEIGHT - 1))
    return Y

def X2lon(X):
    lons = MIN_LON + DELTA_LON * (np.minimum(np.maximum(X, 0), MAP_WIDTH - 1) + 0.5)
    return lons

def Y2lat(Y):
    lats = MIN_LAT + DELTA_LAT * (np.minimum(np.maximum(Y, 0), MAP_HEIGHT - 1) + 0.5)
    return lats

'''
