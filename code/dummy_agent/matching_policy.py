# from sys import flags
from numpy.core.fromnumeric import mean
import numpy as np
from novelties import status_codes
from common import mesh
from common.geoutils import great_circle_distance
from collections import defaultdict
from config.hex_setting import MAP_WIDTH, MAP_HEIGHT
from simulator.services.routing_service import RoutingEngine
import pandas as pd


class MatchingPolicy(object):
    def match(self, current_time, vehicles, requests):
        return []

    # def match_RS(self, current_time, vehicles, requests):
    #     return []

    def find_available_vehicles(self, vehicles):
        idle_vehicles = vehicles[
            ((vehicles.status == status_codes.V_IDLE) |
            (vehicles.status == status_codes.V_CRUISING)) &
            (vehicles.idle_duration > 0)
        ]
        v_list = []
        v_charging_list = []
        for index, v in idle_vehicles.iterrows():
            # print(v['SOC'] ,v['charging_threshold'],v['mile_of_range'])
            # if v['current_capacity'] != v['max_capacity']:
            if v['SOC'] > v['charging_threshold']*v['mile_of_range']:
                v_list.append(v)
            else:
                v_charging_list.append(v)
        # print(len(v_charging_list))
        return pd.DataFrame(v_list) , pd.DataFrame(v_charging_list)

    # Craeting matching dictionary assciated with each vehicle ID
    def create_matching_dict(self, vehicle_id, customer_id, duration, distance):
        match_dict = {}
        match_dict["vehicle_id"] = vehicle_id
        match_dict["customer_id"] = customer_id
        match_dict["duration"] = duration
        match_dict["distance"] = distance
        return match_dict

class RoughMatchingPolicy(MatchingPolicy):
    def __init__(self, reject_distance=5000):
        self.reject_distance = reject_distance  # in meters

    # Matching requests to the nearest available vehicle
    def match(self, current_time, vehicles, requests):
        assignments = []
        vehicles,vehicles_tocharge = self.find_available_vehicles(vehicles)
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return assignments
        # List of distances for all available vehicles to requests' origin points
        d = great_circle_distance(vehicles.lat.values, vehicles.lon.values,
                                  requests.origin_lat.values[:, None], requests.origin_lon.values[:, None])

        for ridx, request_id in enumerate(requests.index):
            vidx = d[ridx].argmin()     # Retrieving the min distance (nearest vehicle to request)
            # Check if it is within the acceptable range of travelling distance
            if d[ridx, vidx] < self.reject_distance:
                vehicle_id = vehicles.index[vidx]
                duration = d[ridx, vidx] / 8.0
                distance = d[ridx, vidx]
                assignments.append(self.create_matching_dict(vehicle_id, request_id, duration, distance))
                d[:, vidx] = float('inf')
            else:
                continue
            if len(assignments) == n_vehicles:
                return assignments
        return assignments


class GreedyMatchingPolicy(MatchingPolicy):
    def __init__(self, reject_distance=5000):
        self.reject_distance = reject_distance  # meters
        self.reject_wait_time = 15 * 60         # seconds
        self.k = 1                              # the number of mesh to aggregate
        self.unit_length = 500                  # mesh size in meters
        self.max_locations = 40                 # max number of origin/destination points
        self.max_cs_batch = 40                  # num of piles per match
        self.reject_range = int(self.reject_distance / self.unit_length / self.k) + 1
        self.cs_searching_range = max(MAP_HEIGHT,MAP_WIDTH)
        self.routing_engine = RoutingEngine.create_engine()


    def get_coord(self, lon, lat):
        x, y = mesh.convert_lonlat_to_xy(lon, lat)
        return (int(x / self.k), int(y / self.k))

    def coord_iter(self):
        for x in range(int(MAP_WIDTH / self.k)):
            for y in range(int(MAP_HEIGHT / self.k)):
                yield (x, y)

    # Candidate Vehicle IDs from the mesh
    def find_candidates(self, coord, n_requests, V, reject_range):
        x, y = coord
        candidate_vids = V[(x, y)][:]
        for r in range(1, reject_range):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    r_2 = dx ** 2 + dy ** 2
                    if r ** 2 <= r_2 and r_2 < (r + 1) ** 2:
                        candidate_vids += V[(x + dx, y + dy)][:]
                if len(candidate_vids) > n_requests * 2:
                    break
        return candidate_vids

    # Returns list of assignments
    def assign_nearest_vehicle(self, ori_ids, dest_ids, T, dist):
        assignments = []
        for di, did in enumerate(dest_ids):
            if len(assignments) >= len(ori_ids):
                break
            # Reuturns the min distance
            oi = T[di].argmin()
            
            # t_queue = ChargingRepository.get_average_wait_time(dest_ids)
            # oi = dist[di].argmin()
            tt = T[di, oi] # - t_queue
            dd = dist[di, oi]
            # print("Chosen t: ", tt)
            # print("Chosen D: ", dd)
            if tt > self.reject_wait_time:
                continue
            oid = ori_ids[oi]

            assignments.append((oid, did, tt, dd))
            T[:, oi] = float('inf')
        return assignments

    # Return list of candidate vehicle in order of the nearest to the request
    def filter_candidates(self, vehicles, requests):
        d = great_circle_distance(vehicles.lat.values, vehicles.lon.values,
                                  requests.origin_lat.mean(), requests.origin_lon.mean())

        within_limit_distance = d < self.reject_distance + self.unit_length * (self.k - 1)
        candidates = vehicles.index[within_limit_distance]
        d = d[within_limit_distance]
        return candidates[np.argsort(d)[:2 * len(requests) + 1]].tolist()

    def filter_CS_candidates(self, vehicles, charging_stations):
        d = great_circle_distance(vehicles.lat.values, vehicles.lon.values,
                                mean(charging_stations[:,0]), mean(charging_stations[:,1]))
        within_limit_distance = d < 1e3*(self.reject_distance + self.unit_length * (self.k - 1)) # a large enough number
        candidates = vehicles.index[within_limit_distance]
        d = d[within_limit_distance]
        return candidates[np.argsort(d)[:2 * len(charging_stations) + 1]].tolist()

    # def match(self, current_time, vehicles, requests):
    #     match_list = self.match_available(current_time, vehicles, requests)
    #     charging_list = self.match_charging_piles(current_time, vehicles,charging_stations)
    #     return match_list+ charging_list

###### match requests to vehicles ######
    def match_requests(self, current_time, vehicles, requests):
        match_list = []
        vehicles,_ = self.find_available_vehicles(vehicles)
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return match_list

        v_latlon = vehicles[["lat", "lon"]]
        V = defaultdict(list)
        vid2coord = {}
        for vid, row in v_latlon.iterrows():
            coord = self.get_coord(row.lon, row.lat) # x, y
            vid2coord[vid] = coord
            V[coord].append(vid)

        r_latlon = requests[["origin_lat", "origin_lon"]]
        R = defaultdict(list)
        for rid, row in r_latlon.iterrows():
            coord = self.get_coord(row.origin_lon, row.origin_lat)
            R[coord].append(rid)
        # V and R are two statuses: vehicle and request per zone.
        
        for coord in self.coord_iter(): 
            if not R[coord]:
                continue

            for i in range(int(np.ceil(len(R[coord]) / self.max_locations))):

                target_rids = R[coord][i * self.max_locations : (i + 1) * self.max_locations] # matching per batch

                candidate_vids = self.find_candidates(coord, len(target_rids), V, self.reject_range)
                if len(candidate_vids) == 0:
                    continue

                target_latlon = r_latlon.loc[target_rids]
                candidate_vids = self.filter_candidates(v_latlon.loc[candidate_vids], target_latlon)
                if len(candidate_vids) == 0:
                    continue
                candidate_latlon = v_latlon.loc[candidate_vids]
                T, dist = self.eta_matrix(candidate_latlon, target_latlon)
                # Calcualte Distance from the nearest vehicle's location to thr request's origin location
                # print("T.T: ", T.T)
                # print(dist_matrix)
                #T.T here .T is for Transpose
                assignments = self.assign_nearest_vehicle(candidate_vids,target_rids,T.T, dist.T)
                for vid, rid, tt, d in assignments:
                    match_list.append(self.create_matching_dict(vid, rid, tt, d))
                    V[vid2coord[vid]].remove(vid)

        return match_list

    
###### match vehicles to available charging station(piles) ######
    def match_charging_stations(self, current_time, vehicles, charging_stations):
        charge_list = []
        _ , tbc_vehicles = self.find_available_vehicles(vehicles) # the vehilce to be charged
        n_tbc_vehicles = len(tbc_vehicles)
        if n_tbc_vehicles == 0:
            return charge_list
        
        v_latlon = tbc_vehicles[["lat", "lon"]]
        V = defaultdict(list)
        vid2coord = {}

        for vid, row in v_latlon.iterrows():
            coord = self.get_coord(row.lon, row.lat) # x, y
            vid2coord[vid] = coord
            V[coord].append(vid)
        # proposed charging station format: c_lat, c_lon (duplicated with num), 
        # here the c_lat, c_lon are the centroid of their corresponding grid. so we need to check back to see how demand is fitted to grid.
        
        c_latlon = [[float(cs.get_cs_location()[0]),float(cs.get_cs_location()[1])] for cs in charging_stations] # charging_stations[["c_lat", "c_lon"]]
        C = defaultdict(list)
        for cid, row in enumerate(c_latlon):
            coord = self.get_coord(row[1],row[0])
            C[coord].append(cid)
        # V and C are: vehicle and charging piles (be duplicated if with same zone).
        # reject_range = int(self.reject_distance / self.unit_length / self.k) + 1
        self.max_cs_batch = len(C[coord])

        for coord in self.coord_iter(): # grid x, y
            if not C[coord]:
                continue
            for i in range(int(np.ceil(len(C[coord]) / self.max_cs_batch))):
                target_cids = C[coord][i * self.max_cs_batch : (i + 1) * self.max_cs_batch]
                
                candidate_vids = self.find_candidates(coord, len(target_cids), V, reject_range=self.reject_range) # self.cs_searching_range 
                if len(candidate_vids) == 0:
                    continue
                cs_latlon = np.array([c_latlon[i] for i in target_cids])
                candidate_vids = self.filter_CS_candidates(v_latlon.loc[candidate_vids],cs_latlon)
                if len(candidate_vids) == 0:
                    continue
                veh_latlon = v_latlon.loc[candidate_vids]
                T, dist = self.eta_matrix(veh_latlon, cs_latlon) # from veh to cs
                cs_wait_time = np.array([cs.get_average_waiting_time() for cid,cs in enumerate(charging_stations) if cid in target_cids])
                
                
                # T_new = [[a+b  for a,b in zip(T[i],cs_wait_time)] for i in range(len(T))]
                T_new = T + cs_wait_time
                assignments = self.assign_nearest_vehicle(candidate_vids,target_cids, T_new.T, dist.T) # the order is right (assgining veh to cs) in the original function
                for vid, cid, tt, d in assignments:
                    charge_list.append(self.create_matching_dict(vid, cid, tt-cs_wait_time, d)) # the matching records veh_id, cs_id, time duration, and distance
                    V[vid2coord[vid]].remove(vid)
        return charge_list


    def eta_matrix(self, origins_array, destins_array):
        try:
            destins = [(lat, lon) for lat, lon in destins_array.values]
        except AttributeError:
            destins = [(loc[0],loc[1]) for loc in destins_array]
        origins = [(lat, lon) for lat, lon in origins_array.values]
        # origin_set = list(set(origins))
        origin_set = list(origins)
        latlon2oi = {latlon: oi for oi, latlon in enumerate(origin_set)}
        T, d = np.array(self.routing_engine.eta_many_to_many(origin_set, destins), dtype=np.float32)
        T[np.isnan(T)] = float('inf')
        d[np.isnan(d)] = float('inf')
        T = T[[latlon2oi[latlon] for latlon in origins]]
        # print("T: ", T)
        # print("D: ", d.shape)
        return [T, d]

'''
###### match requests to vehicles ######
    def match_requests(self, current_time, vehicles, requests):
        match_list = []
        vehicles,_ = self.find_available_vehicles(vehicles)
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return match_list

        v_latlon = vehicles[["lat", "lon"]]
        V = defaultdict(list)
        vid2coord = {}
        for vid, row in v_latlon.iterrows():
            coord = self.get_coord(row.lon, row.lat) # x, y
            vid2coord[vid] = coord
            V[coord].append(vid)

        r_latlon = requests[["origin_lat", "origin_lon"]]
        R = defaultdict(list)
        for rid, row in r_latlon.iterrows():
            coord = self.get_coord(row.origin_lon, row.origin_lat)
            R[coord].append(rid)

        # V and R are two statuses: vehicle and request per zone.
        reject_range = int(self.reject_distance / self.unit_length / self.k) + 1
        for coord in self.coord_iter():
            if not R[coord]:
                continue

            for i in range(int(np.ceil(len(R[coord]) / self.max_locations))):

                target_rids = R[coord][i * self.max_locations : (i + 1) * self.max_locations] # matching per batch

                candidate_vids = self.find_candidates(coord, len(target_rids), V, reject_range)
                if len(candidate_vids) == 0:
                    continue

                target_latlon = r_latlon.loc[target_rids]
                candidate_vids = self.filter_candidates(v_latlon.loc[candidate_vids], target_latlon)
                if len(candidate_vids) == 0:
                    continue
                candidate_latlon = v_latlon.loc[candidate_vids]
                T, dist = self.eta_matrix(candidate_latlon, target_latlon)

                # Calcualte Distance from the nearest vehicle's location to thr request's origin location
                # print("T.T: ", T.T)
                # print(dist_matrix)
                #T.T here .T is for Transpose
                assignments = self.assign_nearest_vehicle(target_rids, candidate_vids, T.T, dist.T)
                for vid, rid, tt, d in assignments:
                    match_list.append(self.create_matching_dict(vid, rid, tt, d))
                    V[vid2coord[vid]].remove(vid)

        return match_list

    
###### match vehicles to available charging station(piles) ######
    def match_charging_stations(self, current_time, vehicles, charging_stations):
        match_list = []
        _,vehicles = self.find_available_vehicles(vehicles) # the vehilce that needs to charge
        n_vehicles = len(vehicles)
        if n_vehicles == 0:
            return match_list

        v_latlon = vehicles[["lat", "lon"]]
        V = defaultdict(list)
        vid2coord = {}
        for vid, row in v_latlon.iterrows():
            coord = self.get_coord(row.lon, row.lat) # x, y
            vid2coord[vid] = coord
            V[coord].append(vid)

        c_latlon = charging_stations[["c_lat", "c_lon"]]
        C = defaultdict(list)
        for cid, row in c_latlon.iterrows():
            coord = self.get_coord(row.c_lon, row.c_lat)
            C[coord].append(cid)
        # V and C are: vehicle and charging piles (be duplicated if with same zone).
        # reject_range = int(self.reject_distance / self.unit_length / self.k) + 1
        for coord in self.coord_iter():
            if not V[coord]:
                continue

            for i in range(int(np.ceil(len(V[coord]) / self.max_cs_batch))):
                
                target_cids = C[coord][i * self.max_cs_batch : (i + 1) * self.max_cs_batch]

                candidate_vids = self.find_candidates(coord, len(target_cids), V, reject_range=self.cs_searching_range)
                if len(candidate_vids) == 0:
                    continue
                target_latlon = c_latlon.loc[target_cids]
                candidate_vids = self.filter_candidates(v_latlon.loc[candidate_vids], target_latlon)
                if len(candidate_vids) == 0:
                    continue
                candidate_latlon = v_latlon.loc[candidate_vids]
                T, dist = self.eta_matrix(candidate_latlon, target_latlon)

                assignments = self.assign_nearest_vehicle(target_cids, candidate_vids, T.T, dist.T) # yes the order is messed in the original function
                for vid, cid, tt, d in assignments:
                    match_list.append(self.create_matching_dict(vid, -cid, tt, d)) # - means to charge
                    V[vid2coord[vid]].remove(vid)

        return match_list
'''