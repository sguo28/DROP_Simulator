import numpy as np
from .models.charging_pile.charging_pile import charging_station
from .services.demand_generation_service import DemandGenerator
from common.time_utils import get_local_datetime
from logger import sim_logger
from logging import getLogger
from .models.vehicle.vehicle_state import VehicleState
from .models.vehicle.vehicle import Vehicle
from .models.zone.matching_zone import matching_zone
from .models.zone.hex_zone import hex_zone
from novelties import agent_codes, status_codes
from random import randrange
import geopandas as gpd
import time
from scipy.spatial import KDTree
from config.hex_setting import FLAGS, NUM_REACHABLE_HEX, NUM_NEAREST_CS, ENTERING_TIME_BUFFER, charging_station_data_path, hex_route_file, STORE_TRANSITION_CYCLE, INPUT_DIM
import pickle
import ray
from dqn_agent.dqn_agent import DeepQNetworkAgent

class Simulator(object):
    def __init__(self, start_time, timestep):
        self.reset(start_time, timestep)
        self.last_vehicle_id = 1
        self.vehicle_queue = []
        sim_logger.setup_logging(self)
        self.logger = getLogger(__name__)
        self.demand_generator = DemandGenerator()
        self.route_cache = {}
        self.current_dummyV = 0
        self.current_dqnV = 0
        # containers as dictionaries
        self.match_zone_collection = []
        self.hex_zone_collection = {}
        self.vehicle_collection = {}
        # DQN for getting actions and dumping transitions
        self.dqn_agent = DeepQNetworkAgent()
        self.all_transitions = []
        self.charging_station_collections = []
        self.num_match = 0
        self.total_num_arrivals = 0
        self.total_num_removed_pass = 0
        self.charging_station_ids = []
        # initialize ray
        ray.init(num_cpus=48)

    def reset(self, start_time=None, timestep=None):
        """
        todo: don't need to init CustomerRepo, consider move it
        """
        if start_time is not None:
            self.__t = start_time
            self.start_time = start_time
        if timestep is not None:
            self.__dt = timestep
        # VehicleRepository.init()
        # CustomerRepository.init()

    def process_trip(self, filename):
        '''
        todo: make the following variables (hours, nhex,nhex) as input or some global vars
        :param filename: trip time OD data or trip count OD data, preprocessed from taxi-records-201605
        :return: 3d numpy array of time OD or count OD
        '''
        nhex = NUM_REACHABLE_HEX
        # process the line based file into a hour by
        data = np.zeros((24, nhex, nhex))
        with open(filename, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')
                h, o, d, t = line[1:]  # hour, oridin, dest, trip_time/num of trip
                data[int(h), int(o), int(d)] = float(t)
        return data

    def init_charging_station(self, file_name):
        """
        todo: delete later: now we assume infinity and supercharging with 1e5 charging piles
        """
        with open(file_name, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')
                num_l2, num_dc, ilat, ilon, hex_id = line
                hex=self.hex_zone_collection[int(hex_id)]
                self.charging_station_collections.append(
                    charging_station(n_l2=int(float(num_l2)), n_dcfast=int(float(num_dc)), lat=float(ilat), lon=float(ilon),
                                     hex_id=int(hex_id),hex=hex))

    def init(self, file_hex, file_charging, trip_file, travel_time_file, n_nearest=NUM_NEAREST_CS):
        '''
        todo: finalize the location of each file and some simulation setting in a config file
        :param file_hex:
        :param file_charging:
        :param trip_file:
        :param travel_time_file:
        :param n_nearest:
        :return:
        '''
        df = gpd.read_file(file_hex)  # tagged_cluster_hex './data/NYC_shapefiles/reachable_hexes.shp'
        charging_stations = gpd.read_file(file_charging)  # point geometry # 'data/NYC_shapefiles/processed_cs.shp'
        self.charging_kdtree = KDTree(charging_stations[['lon', 'lat']])
        self.hex_kdtree = KDTree(df[['lon', 'lat']])
        with open(hex_route_file,'rb') as f:
            self.hex_routes=pickle.load(f)

        matchzones = np.unique(df['cluster_la'])

        hex_ids = df.index.tolist()
        print('Number of total hexagons:', len(hex_ids))

        hex_coords = df[['lon', 'lat']].to_numpy()  # coord
        hex_to_match = df['cluster_la'].to_numpy()  # corresponded match zone id

        demand = self.process_trip(trip_file)
        travel_time = self.process_trip(travel_time_file)  #

        # preprocessed od time mat from OSRM engine
        od_time = np.zeros([NUM_REACHABLE_HEX,NUM_REACHABLE_HEX])
        for (o,d) in self.hex_routes.keys():
            od_time[o,d] = sum(self.hex_routes[(o,d)]['travel_time'])
        od_time[np.isnan(od_time)] = 1e8 # set a large enough number

        epoch_length = 60 * 24 * 7  # this is the total number of ticks set for simulation, change this value.'
        t_unit = 60  # number of time steps per hour

        # we initiaze the set of hexagone zones first
        maxdemand = 0
        total_demand = 0
        charging_coords = charging_stations[['lon', 'lat']].values.tolist()
        charging_hexes = charging_stations[['hex_id']].values.tolist()
        charging_hexes=[item[0] for item in charging_hexes]
        for h_idx, coords, match_id in zip(hex_ids, hex_coords, hex_to_match):
            neighbors = df[df.geometry.touches(df.geometry[h_idx])].index.tolist()  # len from 0 to 6
            _, charging_idx = self.charging_kdtree.query(coords, k=n_nearest)  # charging station id
            if sum(demand[0, h_idx, :]) / 60 > maxdemand: maxdemand = sum(demand[0, h_idx, :]) / 60
            total_demand += sum(demand[0, h_idx, :])
            self.hex_zone_collection[h_idx] = hex_zone(h_idx, coords, hex_coords, match_id, neighbors, charging_idx,
                                                       charging_hexes, charging_coords, demand[:, h_idx, :],
                                                       travel_time[:, h_idx, :], t_unit, epoch_length)
        # print('highest demand per tick=',maxdemand,'total demand for the first hour=',total_demand)


        # ray init hex, try this
        hex_collects = []
        for m_idx in matchzones:
            h_ids = df[df['cluster_la'] == m_idx].index.tolist()
            hex_collects.append([self.hex_zone_collection[hid] for hid in h_ids])

        # we initialize the matching zones through ray
        self.match_to_hex = {}  # a local map of hexagones to matching zones

        [self.match_zone_collection.append(matching_zone.remote(idx, hexs, od_time)) for idx, hexs in
         zip(matchzones, hex_collects)]
        print('matching zone initialized')

        for idx, hexs in zip(matchzones, hex_collects):
            ray.get(self.match_zone_collection[idx].get_info.remote())
            self.match_to_hex[idx] = hexs  # a local container
        print('ray initialize match zone complete')

        # init charging station
        self.init_charging_station(charging_station_data_path)

        # init entering-market vehicle queue


        vehicle_hex_ids = [hex_ids[i] for i in
                           np.random.choice(len(hex_ids), size=FLAGS.vehicles)]  # , p=p)]
        # vehicle_hex_ids = [hex_ids[76] for _ in range(FLAGS.vehicles)]  # , p=p)]
        n_vehicles = len(vehicle_hex_ids)
        vehicle_ids = range(self.last_vehicle_id, self.last_vehicle_id + n_vehicles)
        self.last_vehicle_id += n_vehicles
        entering_time = np.random.uniform(self.__t, self.__t + ENTERING_TIME_BUFFER, n_vehicles).tolist()
        q = sorted(zip(entering_time, vehicle_ids, vehicle_hex_ids))
        self.vehicle_queue = q
        print('initialize vehicle queue compelte')

    def par_step(self):  # we use parallel update to call the step function.
        '''
        Parallel run of the simulator that involves the following key steps:
        1. conduct the matching for each matching zone
        2. Update passenger status
        3. Update vehicle status
        4. Dispatch vehicles
        5. Generate new passengers
        :return:
        '''
        # conduct matching first
        tick = self.__t - self.start_time  # unit: sec
        t_start = time.time()

        [m.match.remote() for m in self.match_zone_collection]  # force this to complete

        # dispatched vehicles which have been attached dispatch actions.
        ray.get([m.dispatch.remote(tick) for m in self.match_zone_collection])

        #self.download_match_zone_metrics()
        # update passenger status
        [m.update_passengers.remote() for m in self.match_zone_collection]

        # update vehicle status
        t1 = time.time()
        self.download_vehicles()
        t_d_update = time.time() - t1

        self.update_vehicles(self.__t) # push routes inside if needed


        t1=time.time()
        #update charging stations...
        [cs.step(self.__dt, self.__t) for cs in self.charging_station_collections]

        cs_time=time.time()-t1

        self.enter_market()

        t1 = time.time()
        self.push_vehicles(self.__dt,self.__t)
        t_p_update = time.time() - t1

        #after the push, call the vehicle to step for all vehicles in each zone
        # t1 = time.time()
        # self.vehicle_step_update(self.__dt,self.__t) # interpolate routes and update vehicle status
        # t_v_update=time.time()-t1

        # update the demand for each matching zone

        ray.get([c.async_demand_gen.remote(tick) for c in self.match_zone_collection])

        '''
        todo: add a download and push function for charging stations. 
        '''


        self.download_match_zone_metrics()

        STORE_TRANSITION_CYCLE=1 #store everytick
        if self.__t % STORE_TRANSITION_CYCLE == 0: # STORE_TRANSITION_CYCLE = 60*60 sec
            self.store_transitions_from_veh()
        self.__update_time()
        if self.__t % 3600 == 0:
            self.logger.info("Elapsed : {}".format(get_local_datetime(self.__t)))
        t_end = time.time() - t_start

        print('Iteration {} completed, total cpu time={:.3f}, time for vehicle status download={:.3f}, push time={:.3f}'.format(tick / 60, t_end,
                                                                                                    t_d_update,t_p_update))
        # finally identify new vehicles, and update location of existing vehicles
        # the results is a list of list of dictionaries.

    def download_match_zone_metrics(self):
        metrics = ray.get([m.get_metrics.remote() for m in self.match_zone_collection])
        # print("Metrics from matching zones are:",metrics)
        self.num_match = sum([item[0] for item in metrics])
        self.total_num_arrivals = sum([item[1] for item in metrics])
        self.total_num_removed_pass = sum([item[2] for item in metrics])

    def download_vehicles(self):
        # copy remote information to local
        # ray return read-only items. we need to make a deep copy first

        # copy_veh=ujson.dumps(ray.get([m.get_vehicles_by_hex.remote() for m in self.match_zone_collection]))
        all_vehs = ray.get([m.get_vehicles_by_hex.remote() for m in self.match_zone_collection])
        for mid, vehs in zip(self.match_to_hex.keys(), all_vehs):
            for hexs, veh in zip(self.match_to_hex[mid], vehs):
                hexs.vehicles = veh  # make a hard copy
                # print('Download success, Number of vehicle for hex {} is {}:'.format(hexs.hex_id,len(veh)))

    def update_vehicles(self,tick):
        '''
        1. loop through all hexagones and update the vehicle status
        2. add veh to charging station
        3. do relocation: attach vehicle's action id
        :return:
        '''

        #add vehicles to charging stations and remove from the hex zone

        vehs_to_update=[veh for hex in self.hex_zone_collection.values() for veh in hex.vehicles.values()]
        [vehicle.update_info(self.hex_zone_collection, self.hex_routes) for vehicle in vehs_to_update]

        self.charging_station_ids = [vehicle.get_assigned_cs_id()\
                                     for vehicle in vehs_to_update if vehicle.state.status == status_codes.V_WAITPILE]

        [self.charging_station_collections[vehicle.get_assigned_cs_id()].add_arrival_veh(vehicle) \
         for vehicle in vehs_to_update if vehicle.state.status == status_codes.V_WAITPILE]

        dqn_agents = [vehicle for hex in self.hex_zone_collection.values() for vehicle in hex.vehicles.values() if
                      vehicle.state.agent_type == agent_codes.dqn_agent and vehicle.state.status == status_codes.V_IDLE]

        if len(dqn_agents) > 0:
            state_batches = [veh.dump_states(tick) for veh in dqn_agents]  # (tick, veh_id, hex_id, SOC)
            num_valid_relos = [len([0] + self.hex_zone_collection[veh.get_hex_id()].neighbor_hex_id) for veh in
                               dqn_agents]  # [0] means stay still
            action_ids = self.dqn_agent.get_actions(state_batches, num_valid_relos)
            for veh,action_id in zip(dqn_agents,action_ids):
                veh.send_to_dispatching_pool(action_id)
        total_running_veh=sum([len(hex.vehicles.keys()) for hex in self.hex_zone_collection.values()])
        total_charge_wait_veh=sum([len(cs.queue) for cs in self.charging_station_collections])
        total_charging_vec=sum([sum([1 for p in cs.piles if p.occupied==True]) for cs in self.charging_station_collections])
        total_veh=total_running_veh+total_charge_wait_veh+total_charging_vec
        average_SOC=np.mean([veh.state.SOC for veh in vehs_to_update])
        print('Total number of vehicles in the system={}, total running={}, total waiting for charge={}, total charging={}, Average SOC={}'\
              .format(total_veh,total_running_veh,total_charge_wait_veh,total_charging_vec,average_SOC))

    def push_vehicles(self,timestep,timetick):
        '''
        send local information to remote
        and execute update after the vehicles are pushed to remote.
        :return:
        '''
        # all_vehs = [[] for _ in range(len(self.match_to_hex.keys()))]  # create default length

        nvehicles = 0
        pushed_veh=[]
        for mid in self.match_to_hex.keys():
            # all_vehs[mid] = [hex.vehicles for hex in self.match_to_hex[mid]]
            pushed_veh.append(ray.put([hex.vehicles for hex in self.match_to_hex[mid]]))
            nvehicles += sum([len(hex.vehicles.keys()) for hex in self.match_to_hex[mid]])
        ray.get([m.set_vehicles_by_hex.remote(vehs,timestep,timetick) for m, vehs in zip(self.match_zone_collection, pushed_veh)])

    def vehicle_step_update(self,timestep,tick):
        [m.update_vehicles.remote(timestep,tick) for m in self.match_zone_collection]

    def match_zone_step_wrapper(self, zone):
        '''
        This is a wrapper to be fed to the parallel pool in each iteration
        '''
        tick = self.__t - self.start_time
        t1 = time.time()
        zone.step(tick)  # call the step function for the matching zone
        return time.time() - t1

    # def thread_update(self, zones):
    #     '''
    #     this will create a parallel pool for run the step function of each matching zone in parallel
    #     # the update is performed in place, so no return is required
    #     :param zones:
    #     :return:
    #     '''
    #     results = self.pool.amap(self.match_zone_step_wrapper, zones)
    #     results = results.get()
    #     print('Time of different zones:', results)

    def sequential_update(self, zones):
        '''
        Perform sequential update
        :param zones:
        :return:
        '''
        times = []
        for zone in zones:
            t = self.match_zone_step_wrapper(zone)
            times.append(t)
       # print('Sequential time for each zone:', times)

        tick = self.__t - self.start_time
        t1 = time.time()
        r = ray.get([mz.demand_gen_async.remote(len(mz.hex_zones), mz.hex_zones[0], tick) for mz in zones])
     #   print('Ray demand gen time:', time.time() - t1)

    def enter_market(self):
        #print('Length of entering queue:', len(self.vehicle_queue))
        while len(self.vehicle_queue) > 0:

            t_enter, vehicle_id, vehicle_hex_id = self.vehicle_queue[0]

            if self.__t >= t_enter:
                self.vehicle_queue.pop(0)  # no longer queueing
                self.populate_vehicle(vehicle_id, vehicle_hex_id)
            else:
                break

    def populate_vehicle(self, vehicle_id, vehicle_hex_id):
        agent_type = 0
        r = randrange(2)
        if r == 0 and self.current_dummyV < FLAGS.dummy_vehicles:
            agent_type = agent_codes.dummy_agent
            self.current_dummyV += 1

        # If r = 1 or num of dummy agent satisfied
        elif self.current_dqnV < FLAGS.dqn_vehicles:
            agent_type = agent_codes.dqn_agent
            self.current_dqnV += 1

        else:
            agent_type = agent_codes.dummy_agent
            self.current_dummyV += 1

        location = (self.hex_zone_collection[vehicle_hex_id].lon, self.hex_zone_collection[
            vehicle_hex_id].lat)  # update its coordinate with the centroid of the hexagon

        self.hex_zone_collection[vehicle_hex_id].add_veh(Vehicle(VehicleState(vehicle_id, location, vehicle_hex_id,
                                                                              agent_type)))  # append this new available vehicle to the hexagon zone

    def __update_time(self):
        self.__t += self.__dt

    def store_transitions_from_veh(self):
        """
        vehicle.dump_transition() returns a list of list. [[s,a,s_next,r,flag]]
        """
        self.all_transitions = []

        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.all_transitions += vehicle.dump_transitions()

    def dump_transition_to_dqn(self):
        """
        todo: change type to list?
        convert transitions to batches of state, action, next_state, and off-duty flag.
        :return:
        """
        all_transitions = np.array(self.all_transitions,dtype=object)
        if len(all_transitions)>0:
            # print('First row of Transitions are:',all_transitions[1])
            [state, action, next_state, reward, flag] = [all_transitions[:,i] for i in range(INPUT_DIM)] # 5 is dim of transition
            return state,action,next_state,reward,flag
        else:
            return None

    def get_current_time(self):
        return self.__t

    def summarize_metrics(self):
        '''
        :todo: find where to export get_num_of_match: matching was posted to remote.
        get metrics of all DQN vehicles
        '''
        all_vehicles = [vehicle for hex in self.hex_zone_collection.values() for vehicle in hex.vehicles.values() if
                        vehicle.state.agent_type == agent_codes.dqn_agent]
        num_idle = sum([veh.state.status == status_codes.V_IDLE for veh in all_vehicles])
        num_serving = sum([veh.state.status == status_codes.V_OCCUPIED for veh in all_vehicles])
        num_cruising = sum([veh.state.status == status_codes.V_CRUISING for veh in all_vehicles])
        num_assigned = sum([veh.state.status == status_codes.V_ASSIGNED for veh in all_vehicles])
        num_offduty = sum([veh.state.status == status_codes.V_OFF_DUTY for veh in all_vehicles])
        num_tobedisptached = sum([veh.state.status == status_codes.V_TOBEDISPATCHED for veh in all_vehicles])
        num_waitpile = sum([len(cs.queue) for cs in self.charging_station_collections])
        num_charging = sum(
            [sum([1 for p in cs.piles if p.occupied == True]) for cs in self.charging_station_collections])
        n_matches = self.num_match
        # total_num_arrivals = sum(
        #     [hex.get_num_arrivals() for mid in self.match_to_hex.keys() for hex in self.match_to_hex[mid]])
        # total_removed_passengers = sum(
        #     [hex.get_removed_passengers() for mid in self.match_to_hex.keys() for hex in self.match_to_hex[mid]])
        total_num_arrivals = self.total_num_arrivals
        total_removed_passengers = self.total_num_removed_pass
        return num_idle, num_serving, num_charging, num_cruising, n_matches, total_num_arrivals,\
               total_removed_passengers, num_assigned, num_waitpile, num_tobedisptached, num_offduty

    def get_charging_station_ids(self):
        return self.charging_station_ids
