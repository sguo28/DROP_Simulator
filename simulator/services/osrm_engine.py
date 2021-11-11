"""Modified RoutingService.route to accept od_pairs list and make asynchronous requests to it"""

import polyline
from .async_requester import AsyncRequester
from config.hex_setting import OSRM_HOSTPORT
# from common import geoutils
import numpy as np
class OSRMEngine(object):
    """Sends and parses asynchronous requests from list of O-D pairs"""
    def __init__(self, n_threads=8):
        self.async_requester = AsyncRequester(n_threads)
        self.route_cache = {}

    def nearest_road(self, points):
        """Input list of Origin-Destination (lat,lon) pairs, return
        tuple of (trajectory latlongs, distance, triptime)"""
        # list of urls that contain the nearest map for each point
        urllist = [self.get_nearest_url(point) for point in points]
        # List of responses
        #responses = self.async_requester.send_async_requests(points)
        responses = self.async_requester.combine_async(points)
        resultlist = []
        for res in responses:
            nearest_point = res["waypoints"][0]
            location = nearest_point["location"]
            distance = nearest_point["distance"]
            resultlist.append((location, distance))
        return resultlist

    def route(self, od_list, decode=True):
        """Input list of Origin-Destination latlong pairs, return
        tuple of (trajectory latlongs, distance, triptime)"""
        urllist = [self.get_route_url(origin, destin) for origin, destin in od_list]
        responses = self.async_requester.send_async_requests(urllist)
        resultlist = []
        for res in responses:
            if "routes" not in res:
                continue
            route = res["routes"][0]    # Getting the next route available
            triptime = route["duration"]
            if decode:
                trajectory = polyline.decode(route['geometry'])
            else:
                trajectory = route['geometry']
            resultlist.append((trajectory, triptime))

        return resultlist

    def route_hex(self, od_list, decode=False, annotations = False):
        """Input list of Origin-Destination latlong pairs, return
        tuple of (trajectory latlongs, distance, triptime)"""
        # print(od_list[1])

        step=10000
        resultlist = []
        for i in range(0,len(od_list),step):
            responses = self.async_requester.sequential_route(od_list[i:i+step],annotations)
            for res in responses:
                if "routes" not in res:
                    # continue
                    resultlist.append([[],[],[]])
                else:
                    ##we return the first/last coordinate of each step segment of routes, and the corresponding distance and travel time.
                    route = res["routes"][0]    # Getting the next route available

                    trajectory=[[item['geometry']['coordinates'][0], item['geometry']['coordinates'][-1]] for item in
                     route['legs'][0]['steps'][:-1]]  #not including the last one, which is 0
                    triptime = [item['duration'] for item in
                     route['legs'][0]['steps'][:-1]]  #not including the last one, which is 0
                    distance = [item['distance'] for item in
                     route['legs'][0]['steps'][:-1]]
                    resultlist.append([trajectory, triptime,distance])
            print('Iteration {} finished'.format(i))
                # else: # for more detailed info (used for FHV trajectory query)
                #     route = res['routes'][0]
                #     trajectory = [[item['geometry']['coordinates'][0],item['geometry']['coordinates'][-1]] for item in route['legs'][0]['steps'][:-1]]
                #     duration = [item['duration'] for item in route['legs'][0]['steps'][:-1]]
                #     distance = [item['distance'] for item in route['legs'][0]['steps'][:-1]]
                #     precise_node_id = route['legs'][0]['annotation']['nodes']
                #     node_id = [item for item in precise_node_id if item < int(1e8)]
                #     precise_speed = route['legs'][0]['annotation']['speed']
                #     precise_duration = route['legs'][0]['annotation']['duration']
                #     precise_distance = route['legs'][0]['annotation']['distance']
                #     resultlist.append([trajectory, duration, distance,node_id,precise_speed,precise_duration,precise_distance])
        return resultlist

    # Getting trajectory, time from cache if exists, and storing it to the cache if it does not exsist
    # def get_route_cache(self, l, a):
    #     if l in self.route_cache:
    #         if a in self.route_cache[l]:
    #             trajectory, triptime = self.route_cache[l][a]
    #             return trajectory[:], triptime
    #     else:
    #         self.route_cache[l] = {}
    #     x, y = l
    #     ax, ay = a
    #     origin = convert_xy_to_lonlat(x, y)
    #     destin = convert_xy_to_lonlat(x + ax, y + ay)
    #     self.route_cache[l][a] = self.route([(origin, destin)])[0]      # Storing the route to the cache
    #     trajectory, triptime = self.route_cache[l][a]
    #     return trajectory[:], triptime

    def get_route_cache_by_lonlat(self,o_lonlat,d_lonlat):
        if o_lonlat in self.route_cache:
            if d_lonlat in self.route_cache[o_lonlat]:
                trajectory, triptime = self.route_cache[o_lonlat][d_lonlat]
                return trajectory[:], triptime
        else:
            self.route_cache[o_lonlat] = {}
        self.route_cache[o_lonlat][d_lonlat] = self.route([(o_lonlat, d_lonlat)])[0]      # Storing the route to the cache
        trajectory, triptime = self.route_cache[o_lonlat][d_lonlat]
        return trajectory[:], triptime

    # Estimating Duration
    def eta_one_to_many(self, origin_destins_list):
        urllist = [self.get_eta_one_to_many_url([origin] + destins) for origin, destins in origin_destins_list]
        responses = self.async_requester.send_async_requests(urllist)
        resultlist = []
        for res in responses:
            eta_list = res["durations"][0][1:]
            resultlist.append(eta_list)
        return resultlist

    def eta_many_to_one(self, origins_destin_list):
        urllist = [self.get_eta_one_to_many_url(origins + [destin]) for origins, destin in origins_destin_list]
        responses = self.async_requester.send_async_requests(urllist)
        resultlist = []
        for res in responses:
            eta_list = [d[0] for d in res["durations"][:-1]]
            resultlist.append(eta_list)
        return resultlist
    # def eta_many_to_many(self, origins, destins):
    #     origins_lon, origins_lat = zip(*origins)
    #     destins_lon, destins_lat = zip(*destins)
    #     origins_lon, origins_lat, destins_lon, destins_lat = map(np.array, [origins_lon, origins_lat, destins_lon, destins_lat])
    #     d = geoutils.great_circle_distance(origins_lon[:, None],origins_lat[:, None],
    #                                        destins_lon, destins_lat)
    #     return d

    # def eta_many_to_many(self, origins, destins):
    #     '''
    #     todo: finish wrapping it
    #     :param origins:
    #     :param destins:
    #     :return: trip duration matrix
    #     '''
    #     res = [self.async_requester.combine_async_route([origin,destin])[0] for origin,destin in zip(origins,destins)]
    #     # url = self.get_eta_many_to_many_url(origins, destins)
    #     # res = self.async_requester.send_async_requests([url])[0]
    #     try:
    #         eta_matrix = res["durations"]
    #     except:
    #         print(origins, destins, res)
    #         raise
    #     return eta_matrix #it's a time matrix, think about *SPEED


    def get_route_url(cls, from_lonlat, to_lonlat):
        """Get URL for osrm backend call for arbitrary to/from latlong pairs"""

        urlholder = """http://{hostport}/route/v1/driving/{lon0},{lat0};{lon1},{lat1}?overview=full""".format(
            hostport=OSRM_HOSTPORT,
            lon0=from_lonlat[0],
            lat0=from_lonlat[1],
            lon1=to_lonlat[0],
            lat1=to_lonlat[1]
            )
        return urlholder


    def get_nearest_url(cls, latlon):
        urlholder = """http://{hostport}/nearest/v1/driving/{lon},{lat}?number=1""".format(
            hostport=OSRM_HOSTPORT,
            lon=latlon[1],
            lat=latlon[0]
            )
        return urlholder

    def get_eta_one_to_many_url(cls, latlon_list):
        urlholder = """http://{hostport}/table/v1/driving/polyline({coords})?sources=0""".format(
            hostport=OSRM_HOSTPORT,
            coords=polyline.encode(latlon_list, 5)
        )
        return urlholder

    def get_eta_many_to_one_url(cls, latlon_list):
        urlholder = """http://{hostport}/table/v1/driving/polyline({coords})?destinations={last_idx}""".format(
            hostport=OSRM_HOSTPORT,
            coords=polyline.encode(latlon_list, 5),
            last_idx=len(latlon_list) - 1
        )
        return urlholder

    def get_eta_many_to_many_url(cls, from_lonlat_list, to_lonlat_list):
        lonlat_list = from_lonlat_list + to_lonlat_list
        ids = range(len(lonlat_list))
        urlholder = """http://{hostport}/table/v1/driving/polyline({coords})?sources={sources}&destinations={destins}""".format(
            hostport=OSRM_HOSTPORT,
            coords=polyline.encode(lonlat_list, 5),
            sources=';'.join(map(str, ids[:len(from_lonlat_list)])),
            destins=';'.join(map(str, ids[len(from_lonlat_list):]))
        )
        return urlholder
