import polyline
import os
import pickle
import numpy as np
from config.hex_setting import DATA_DIR, MAX_MOVE
from common import mesh, geoutils


class FastRoutingEngine(object):
    def __init__(self):
        self.tt_map = np.load(os.path.join(DATA_DIR, 'tt_map.npy'))
        self.routes = pickle.load(open(os.path.join(DATA_DIR, 'routes.pkl'), 'rb'))

        # self.dist_map = pickle.load(open(os.path.join(DATA_DIR, 'hex_dist_map.pkl'), 'rb'))

        d = self.tt_map.copy()
        for x in range(d.shape[0]):
            origin_lon = mesh.X2lon(x)
            for y in range(d.shape[1]):
                origin_lat = mesh.Y2lat(y)
                for axi in range(d.shape[2]):
                    x_ = x + axi - MAX_MOVE
                    destin_lon = mesh.X2lon(x_)
                    for ayi in range(d.shape[3]):
                        y_ = y + ayi - MAX_MOVE
                        destin_lat = mesh.Y2lat(y_)
                        d[x, y, axi, ayi] = geoutils.great_circle_distance(
                            origin_lon, origin_lat, destin_lon, destin_lat)
        self.ref_d = d  # Distance in meters
    # (Origin - destination) pairs
    def route_hex(self, od_pairs):
        results = []
        for (origin_lat, origin_lon), (dest_lat, dest_lon) in od_pairs:
            x, y = mesh.convert_lonlat_to_xy(origin_lon, origin_lat)
            x_, y_ = mesh.convert_lonlat_to_xy(dest_lon, dest_lat)
            ax, ay = max(0,min(x_,52)) - max(0,min(x,52)), max(0,min(y_,68)) - max(0,min(y,68))
            axi = x_ - x + MAX_MOVE
            ayi = y_ - y + MAX_MOVE

            trajectory = polyline.decode(self.routes[(x, y)][(ax, ay)]) # Route from origin to destination
            triptime = self.tt_map[x, y, axi, ayi]
            results.append((trajectory, triptime))
        return results

    # def route_hex(self,od_list):
    #     origin_hex_id, destination_hex_id = od_list
    #     resultlist = []
    #     trajectory = self.routes[origin_hex_id][destination_hex_id] # Route from origin to destination
    #     triptime = self.tt_map[origin_hex_id][destination_hex_id]
    #     resultlist.append(trajectory,triptime)
    #     return resultlist

    # def route_hex(self, od_list, decode=False):
    #     """Input list of Origin-Destination latlong pairs, return
    #     tuple of (trajectory latlongs, distance, triptime)"""
    #
    #     responses = self.async_requester.combine_async_route(od_list)
    #     # print(responses)
    #     resultlist = []
    #     for res in responses:
    #         if "routes" not in res:
    #             continue
    #         route = res["routes"][0]  # Getting the next route available
    #         triptime = route["duration"]
    #         if decode:
    #             trajectory = polyline.decode(route['geometry'])
    #         else:
    #             trajectory = route['geometry']
    #         resultlist.append((trajectory, triptime))
    #     return resultlist

    # Estimating arrival (Duration) continously until we reach destination
    def eta_many_to_many(self, origins, destins):
        origins_lon, origins_lat = zip(*origins)
        destins_lon, destins_lat = zip(*destins)
        origins_lon, origins_lat, destins_lon, destins_lat = map(np.array, [origins_lon, origins_lat, destins_lon, destins_lat])
        d = geoutils.great_circle_distance(origins_lon[:, None],origins_lat[:, None],
                                           destins_lon, destins_lat)
        return d



    def get_distance_matrix(self, origin_ids, destination_ids):
        '''
        todo: use table
        :param origin_ids:
        :param destination_ids:
        :return:
        '''
        oid = zip(*origin_ids)
        did = zip(*destination_ids)
        oid, did = map(np.array, [oid,did])
        # for oid in origin_ids:
        #     for did in destination_ids:
        #         dist[oid][did] = self.dist_map[oid][did]
        # return dist

