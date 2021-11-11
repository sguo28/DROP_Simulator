import pickle
import time
from scipy.spatial import cKDTree
from logging import getLogger
from random import randrange
import random
import geopandas as gpd
import pandas as pd
import numpy as np
from config.hex_setting import NUM_REACHABLE_HEX, NUM_NEAREST_CS, ENTERING_TIME_BUFFER, HEX_ROUTE_FILE, \
    ALL_HEX_ROUTE_FILE, CS_DATA_PATH, STORE_TRANSITION_CYCLE, SIM_DAYS, ALL_SUPERCHARGING, N_VEHICLES, \
    N_DUMMY_VEHICLES, N_DQN_VEHICLES, MAP_WIDTH, MAP_HEIGHT, NUM_CHANNELS, HEX_DIFFUSION_PATH,TERMINAL_STATE_SAVE_PATH, OPTION_DIM, DEMAND_SCALE
from logger import sim_logger
from novelties import agent_codes, status_codes
from .models.charging_pile.charging_pile import charging_station
from .models.vehicle.vehicle import Vehicle
from .models.vehicle.vehicle_state import VehicleState
from .models.zone.hex_zone import hex_zone
from .models.zone.matching_zone_sequential import matching_zone
import ijson



class Simulator(object):
    def __init__(self, start_time, timestep, isoption=False,islocal=True,ischarging=True):
        self.reset_time(start_time, timestep)
        self.last_vehicle_id = 1
        self.vehicle_queue = []
        # sim_logger.setup_logging(self)
        # self.logger = getLogger(__name__)
        self.route_cache = {}
        self.current_dummyV = 0
        self.current_dqnV = 0
        # containers as dictionaries
        self.match_zone_collection = []
        self.hex_zone_collection = {}
        # DQN for getting actions and dumping transitions
        self.all_transitions = []
        self.prime_transitions = []
        self.f_transitions=[]
        self.fo_transitions=[]
        self.charging_station_collections = []
        self.snapped_hex_coords_list = [[0,0]]
        self.num_match = 0
        self.total_num_arrivals = 0
        self.total_num_removed_pass = 0
        self.total_num_served_pass = 0
        self.total_num_longwait_pass = 0
        self.total_idled_vehicles = 0
        self.global_state_tensor = np.zeros((NUM_CHANNELS, MAP_HEIGHT, MAP_WIDTH))  # t, c, w, h
        # self.global_state_tensor[2] = 1e6  # queue length
        self.global_state_buffer=dict()

        self.with_charging = ischarging
        self.local_matching = islocal
        self.with_option = isoption
        self.hex_diffusions = None

    def reset_time(self, start_time=None, timestep=None):
        if start_time is not None:
            self.__t = start_time
            self.start_time = start_time
        if timestep is not None:
            self.__dt = timestep

    def process_trip(self, filename):
        '''
        :param filename: trip time OD data or trip count OD data, preprocessed from taxi-records-201605
        :return: 3d numpy array of time OD or trip count OD
        '''
        nhex = NUM_REACHABLE_HEX
        # process the line based file into a hour by
        data = np.zeros((24, nhex, nhex))
        with open(filename, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')
                h, o, d, t = line[1:]  # hour, oridin, dest, trip_time/num of trip
                data[int(h), int(o), int(d)] = float(t)*DEMAND_SCALE
        return data

    def travel_time(self, filename):
        '''
        :param filename: trip time OD data or trip count OD data, preprocessed from taxi-records-201605
        :return: 3d numpy array of time OD or trip count OD
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

    def read_terminal(self):
        middle_terminal = dict()
        for oid in range(OPTION_DIM):
            with open('saved_f/term_states_%d.csv' % oid, 'r') as ts:
                next(ts)
                for lines in ts:
                    line = lines.strip().split(',')
                    hr, hid = line  # option_network_id, hour, hex_ids in terminal state
                    if (oid, int(hr)) in middle_terminal.keys():
                        middle_terminal[(oid, int(hr))].append(int(hid))
                    else:
                        middle_terminal[(oid, int(hr))] = [int(hid)]
        return middle_terminal

    def init_charging_station(self, file_name):
        with open(file_name, 'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')

                _, num, ilat, ilon, hex_id, type = line
                if ALL_SUPERCHARGING:
                    type = 1
                _,hex_id=self.hex_kdtree.query((ilon,ilat)) #match the hex_id
                hex = self.hex_zone_collection[int(hex_id)]
                hex.n_charges += 1
                if type == 0:
                    self.charging_station_collections.append(
                        charging_station(n_l2=int(float(num)), n_dcfast=int(0), lat=float(ilat), lon=float(ilon),
                                         hex_id=int(hex_id), hex=hex, row_col_coord=(hex.row_id, hex.col_id)))
                elif type == 1:
                    self.charging_station_collections.append(
                        charging_station(n_l2=int(0), n_dcfast=int(float(num)), lat=float(ilat), lon=float(ilon),
                                         hex_id=int(hex_id), hex=hex, row_col_coord=(hex.row_id, hex.col_id)))
    def get_hex_diffusions(self, diff_file, row_col_coords):
        with open(diff_file, "rb") as f:
            hex_diffusions = pickle.load(f)  # with key: hex_id
        mat = np.zeros((NUM_REACHABLE_HEX, MAP_HEIGHT,MAP_WIDTH))

        for key_id, diffusions in hex_diffusions.items():
            for hex_id, diff in enumerate(diffusions):
                row_id, col_id = row_col_coords[hex_id]
                mat[key_id, row_id, col_id] = diff
        return mat

    def init(self, file_hex, file_charging, trip_file, travel_time_file, n_nearest=NUM_NEAREST_CS):
        '''
        :param file_hex:
        :param file_charging:
        :param trip_file:
        :param travel_time_file:
        :param n_nearest:
        :return:
        '''

        #read demand
        filename = "../data/trips_on_day0.csv"
        n = sum(1 for line in open(filename)) - 1  # number of records in file (excludes header)
        s = int(n*0.3)  # desired sample size
        random.seed(10)
        skip = sorted(
            random.sample(range(1, n + 1), n - s))  # the 0-indexed header will not be included in the skip list


        # alltrips=pd.read_csv('../data/trips_on_day0.csv',skiprows=skip,dtype=int).groupby('origin_hid')


        local_rand_seed=np.random.randint(3) #randomly generating a seed for hex areas

        alltrips = pd.read_csv('../data/trips_on_day{}.csv'.format(local_rand_seed), dtype=int).groupby('origin_hid')

        df = gpd.read_file(file_hex)  # tagged_cluster_hex
        charging_stations = gpd.read_file(file_charging)  # point geometry
        self.charging_kdtree = cKDTree(charging_stations[['snap_lon', 'snap_lat']])
        self.hex_kdtree = cKDTree(df[['snap_lon', 'snap_lat']])
        with open(ALL_HEX_ROUTE_FILE, 'rb') as f:
            # self.hex_routes = pickle.load(f)
            self.hex_routes=pickle.load(f)

        # if not self.local_matching:
        df['cluster_la'] = 0 # set a mask to combine to one matching zone.


        matchzones = np.unique(df['cluster_la'])




        hex_ids = df.index.tolist()
        print('Number of total hexagons:', len(hex_ids))

        hex_coords = df[['snap_lon', 'snap_lat']].to_numpy()  # coord
        self.snapped_hex_coords_list = df[['snap_lon', 'snap_lat']].values.tolist()
        hex_to_match = df['cluster_la'].to_numpy()  # corresponded match zone id
        row_col_coords = df[['row_id', 'col_id']].to_numpy()
        demand = self.process_trip(trip_file)
        travel_time = self.travel_time(travel_time_file)


        self.hex_diffusions = self.get_hex_diffusions(HEX_DIFFUSION_PATH, row_col_coords)

        # preprocessed od time mat from OSRM engine
        od_time = 600*np.ones([NUM_REACHABLE_HEX, NUM_REACHABLE_HEX])
        for (o, d) in self.hex_routes.keys():
            od_time[o, d] = sum(self.hex_routes[(o, d)]['travel_time'])
            if od_time[o,d]==0:
                od_time[o,d]=60
                # print(o,d,self.hex_routes[(o,d)])
                # self.hex_routes[(o,d)]['travel_time']=[60]
        od_time[np.isnan(od_time)] = 1e8  # set a large enough number

        self.od_time=od_time

        epoch_length = 60 * 24 * SIM_DAYS  # this is the total number of ticks set for simulation, change this value.'
        t_unit = 60  # number of time steps per hour



        # we initiaze the set of hexagone zones first
        maxdemand = 0
        total_demand = 0
        charging_coords = charging_stations[['snap_lon', 'snap_lat']].values.tolist()
        _,charging_hexes=self.hex_kdtree.query(charging_coords)

        self.all_neighbors=np.zeros((NUM_REACHABLE_HEX,7)).astype(int) #store the index here
        for h_idx, coords, row_col_coord, match_id in zip(hex_ids, hex_coords, row_col_coords, hex_to_match):
            #neighbors = df[df.geometry.touches(df.geometry[h_idx])].index.tolist()  # len from 0 to 6
            _,neighbors= self.hex_kdtree.query(coords,k=7)
            # neighbors=list(neighbors[1:]) #return the closet 6 locations
            neighbors=list(neighbors)
            # time_to_destination=[od_time[h_idx,k] for k in neighbors]
            #             # print(h_idx,neighbors,time_to_destination)
            self.all_neighbors[h_idx,:]=neighbors
            if h_idx!=neighbors[0]:
                print('error encountered for h_idx',h_idx,neighbors)
                new_neighbors=[h_idx]+[i for i in neighbors if i!=h_idx]
                self.all_neighbors[h_idx,:]=new_neighbors
                neighbors=new_neighbors
                print('corrected for h_idx', h_idx, new_neighbors)
            # if neighbors==[]:
            #     print('isolated zone with id',h_idx)
            #     if h_idx<NUM_REACHABLE_HEX-1:
            #         neighbors=[h_idx+2]
            #     else:
            #         neighbors=[h_idx-2]
            _, charging_idx = self.charging_kdtree.query(coords, k=n_nearest)  # charging station id
            if sum(demand[0, h_idx, :]) / 60 > maxdemand: maxdemand = sum(demand[0, h_idx, :]) / 60
            try:
                demand_df=alltrips.get_group(int(h_idx))
            except KeyError:
                demand_df=pd.DataFrame({'tick':[]})
            total_demand += sum(demand[0, h_idx, :])
            self.hex_zone_collection[h_idx] = hex_zone(h_idx, coords, row_col_coord, hex_coords, match_id, neighbors,
                                                       charging_idx, charging_hexes, charging_coords,
                                                       demand[:, h_idx, :], travel_time[:, h_idx, :],
                                                       t_unit, epoch_length,local_rand_seed,demand_df)
        hex_collects = []
        for m_idx in matchzones:
            h_ids = df[df['cluster_la'] == m_idx].index.tolist()
            hex_collects.append([self.hex_zone_collection[hid] for hid in h_ids])

        # initialize the matching zones
        self.match_to_hex = {}  # a local map of hexagons to matching zones

        [self.match_zone_collection.append(matching_zone(idx, hexs, od_time)) for idx, hexs in
         zip(matchzones, hex_collects)]
        print('matching zone initialized')

        for idx, hexs in zip(matchzones, hex_collects):
            self.match_zone_collection[idx].get_info()
            self.match_to_hex[idx] = hexs  # a local container

        # init charging station
        self.init_charging_station(CS_DATA_PATH)

        # init entering-market vehicle queue
        random.seed(10)
        vehicle_hex_ids=[random.choice(hex_ids) for _ in range(N_VEHICLES)]
        # vehicle_hex_ids = [hex_ids[i] for i in rands]  # , p=p)]

        n_vehicles = len(vehicle_hex_ids)
        vehicle_ids = range(self.last_vehicle_id, self.last_vehicle_id + n_vehicles)
        self.last_vehicle_id += n_vehicles
        entering_time = np.random.uniform(self.__t, self.__t + ENTERING_TIME_BUFFER, n_vehicles).tolist()
        q = sorted(zip(entering_time, vehicle_ids, vehicle_hex_ids))
        self.vehicle_queue = q
        print('initialize vehicle queue compelte')

    def step(self):  # we use parallel update to call the step function.
        '''
        1. conduct the matching for each matching zone
        2. Update passenger status
        3. Update vehicle status
        4. Dispatch vehicles
        5. Generate new passengers
        :return:
        '''
        # print('match time:', time.time()-t1)
        #update vehicle locations
        # self.update_vehicles()  # push routes into vehicles
        # update passenger status
        # t1 = time.time()
        [m.update_passengers() for m in self.match_zone_collection]

        # update the demand for each matching zone
        # t1 = time.time()
        [c.async_demand_gen(self.__t) for c in self.match_zone_collection]

        # dispatched vehicles which have been attached dispatch actions.

        # t1 = time.time()


        [m.dispatch(self.__t) for m in self.match_zone_collection]
        # print('dispatch time:', time.time() - t1)

        # t1 = time.time()
        [m.match(self.__t) for m in self.match_zone_collection]  # force this to complete


        # print('update_pass', time.time() - t1)

        # t1 = time.time()
        self.update_vehicles()  # push routes into vehicles
        # print('update vehicle time:', time.time() - t1)
        self.enter_market()

        # update charging stations...
        # t1 = time.time()
        # [cs.step(self.__dt, self.__t) for cs in self.charging_station_collections]
        # # print('charging time:', time.time() - t1)
        # t_v_update=time.time()-t1
        # t1 = time.time()
        self.vehicle_step_update(self.__dt, self.__t)  # interpolate routes and update vehicle status

        # print('veh step time', time.time() - t1)


        # first get the global state, based on which the agents take actions.
        # self.store_global_states()
        self.download_match_zone_metrics()

    def update(self):

        # print('demand gen time', time.time() - t1)
        self.__update_time()

        # if self.__t % 3600 == 0:
        #     self.logger.info("Elapsed : {}".format(get_local_datetime(self.__t)))  # added origin UNIX time inside func.

        # print('Iteration {} completed, push time={:.3f}'.format(tick / 60, t_p_update))
        # finally identify new vehicles, and update location of existing vehicles
        # the results is a list of list of dictionaries.

    def download_match_zone_metrics(self):
        metrics = [m.get_metrics() for m in self.match_zone_collection]
        self.num_match = sum([item[0] for item in metrics])
        self.total_num_arrivals = sum([item[1] for item in metrics])
        self.total_num_longwait_pass = sum([item[2] for item in metrics])
        self.total_num_served_pass = sum([item[3] for item in metrics])
        self.total_idled_vehicles = sum([item[4] for item in metrics])
        total_assigned=sum([item[5] for item in metrics])
        total_v=sum([item[6] for item in metrics])
        # print(self.num_match,self.total_num_arrivals,self.total_num_served_pass,self.total_idled_vehicles,total_assigned,total_v)

    def update_vehicles(self):
        '''
        1. loop through all hexagones and update the vehicle status
        2. add veh to charging station
        3. do relocation: attach vehicle's action id
        :return:
        '''
        # add vehicles to charging stations and remove from the hex zone

        vehs_to_update = [veh for hex in self.hex_zone_collection.values() for veh in hex.vehicles.values()]


        # veh_hex_to_update = [veh for veh in vehs_to_update if veh.state.status == status_codes.V_CRUISING and veh.state.real_time_location]
        if vehs_to_update:
            _, real_time_hex_id = self.hex_kdtree.query([veh.state.real_time_location for veh in vehs_to_update])

            for hex_id, veh in zip(real_time_hex_id,vehs_to_update):
                veh.state.current_hex = hex_id

        [vehicle.update_info(self.hex_zone_collection, self.hex_routes, self.snapped_hex_coords_list, self.__t) for vehicle in vehs_to_update]
        #
        # [self.charging_station_collections[vehicle.get_assigned_cs_id()].add_arrival_veh(vehicle) \
        #  for vehicle in vehs_to_update if vehicle.state.status == status_codes.V_WAITPILE]

    def get_local_states(self):
        state_batches = []
        num_valid_relocations = []
        assigned_option_ids=[]
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                if vehicle.state.agent_type == agent_codes.dqn_agent and vehicle.state.status == status_codes.V_IDLE:
                    state_batches.append(vehicle.dump_states(self.__t))  # (tick, hex_id, SOC)
                    num_valid_relocations.append(len([0] + self.hex_zone_collection[vehicle.get_hex_id()].neighbor_hex_id))
                    assigned_option_ids.append(vehicle.assigned_option)

        return state_batches, num_valid_relocations,assigned_option_ids

    def get_local_infos(self):
        state_batches = []
        num_valid_relocations = []
        excluded_od_pairs = []
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                if vehicle.state.agent_type == agent_codes.dqn_agent and vehicle.state.status == status_codes.V_IDLE:
                    state_batches.append(vehicle.dump_states(self.__t))  # (tick, hex_id, SOC)
                    num_valid_relocations.append(len([0] + self.hex_zone_collection[vehicle.get_hex_id()].neighbor_hex_id))
                if vehicle.require_online_query:
                    excluded_od_pairs += [vehicle.non_included_od_pair]
        return state_batches, num_valid_relocations, excluded_od_pairs



    def get_global_state(self):
        self.store_global_states()
        # global_state_slice = np.zeros((NUM_CHANNELS, MAP_WIDTH, MAP_HEIGHT))  # t, c, w, h
        # global_state_slice[2] = 1e6  # queue length
        # for hex_i in self.hex_zone_collection.values():
        #     self.global_state_tensor[0:2, hex_i.x, hex_i.y] = [sum(hex_i.arrivals[-15:]), len(hex_i.vehicles.keys())]
        #
        # for cs in self.charging_station_collections:
        #     self.global_state_tensor[t, 2, cs.x_coord, cs.y_coord] += len(cs.queue) / self.hex_zone_collection[
        #         cs.hex_id].n_charges
        global_state_slice = self.global_state_buffer[self.__t // 60]

        return global_state_slice  # take the last slice that was stored at the start of this tick.


    def dump_global(self):
        global_dict=self.global_state_buffer
        self.global_state_buffer=dict()
        return global_dict

    def store_global_states(self):
        """
        1. # of request per hex_zone
        2. # of available vehicle per hex_zone
        3. # of queueing vehicle per hex_zone
        :return:
        """
        t=self.__t//60
        tmp_state=np.array(self.global_state_tensor)
        for hex_i in self.hex_zone_collection.values():
            tmp_state[0:2, hex_i.row_id, hex_i.col_id] = [sum(hex_i.arrivals[-15:]), len(hex_i.vehicles.keys())]

        for cs in self.charging_station_collections:
            tmp_state[2, cs.row_id, cs.col_id] += len(cs.queue) / self.hex_zone_collection[
                cs.hex_id].n_charges
        self.global_state_buffer[t]=tmp_state

    def attach_actions_to_vehs(self, action_select, action_to_execute,opts=None,contd_options=None):
        dqn_agents = [vehicle for hex in self.hex_zone_collection.values() for vehicle in hex.vehicles.values() if
                      vehicle.state.agent_type == agent_codes.dqn_agent and \
                      vehicle.state.status == status_codes.V_IDLE]

        if opts is None:
            for veh, a_sel,a_exe in zip(dqn_agents, action_select,action_to_execute):
                veh.send_to_dispatching_pool(a_sel,a_exe) # and interpret the actions at hex_zone dispatching function.
        else:
            for veh, a_sel,a_exe, opt,ctd_opts in zip(dqn_agents,action_select,action_to_execute,opts,contd_options):
                veh.send_to_dispatching_pool(a_sel,a_exe,opt,ctd_opts)

    def vehicle_step_update(self, timestep, tick):
        [m.update_vehicles(timestep, tick) for m in self.match_zone_collection]

    # def match_zone_step_wrapper(self, zone):
    #     '''
    #     This is a wrapper to be fed to the parallel pool in each iteration
    #     '''
    #     tick = self.__t - self.start_time
    #     t1 = time.time()
    #     zone.step(tick)  # call the step function for the matching zone
    #     return time.time() - t1

    def enter_market(self):
        # print('Length of entering queue:', len(self.vehicle_queue))
        while len(self.vehicle_queue) > 0:

            t_enter, vehicle_id, vehicle_hex_id = self.vehicle_queue[0]

            if self.__t >= t_enter:
                self.vehicle_queue.pop(0)  # no longer queueing
                self.populate_vehicle(vehicle_id, vehicle_hex_id)
            else:
                break

    def populate_vehicle(self, vehicle_id, vehicle_hex_id):
        r = 1
        if r == 0 and self.current_dummyV < N_DUMMY_VEHICLES:
            agent_type = agent_codes.dummy_agent
            self.current_dummyV += 1

        # If r = 1 or num of dummy agent satisfied
        elif self.current_dqnV < N_DQN_VEHICLES:
            agent_type = agent_codes.dqn_agent
            self.current_dqnV += 1

        else:
            agent_type = agent_codes.dummy_agent
            self.current_dummyV += 1

        location = (self.hex_zone_collection[vehicle_hex_id].lon, self.hex_zone_collection[
            vehicle_hex_id].lat)  # update its coordinate with the centroid of the hexagon

        # append this new available vehicle to the hexagon zone
        self.hex_zone_collection[vehicle_hex_id].add_veh(Vehicle(
            VehicleState(vehicle_id, location, vehicle_hex_id,agent_type),self.with_option,self.local_matching,self.with_charging,self.__t))

    def __update_time(self):
        self.__t += self.__dt

    def store_transitions_from_veh(self):
        """
        vehicle.dump_transition() returns a list of list. [[s,a,s_next,r]]
        """

        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.all_transitions += vehicle.dump_transitions()

    def store_prime_action_from_veh(self):
        """
        vehicle.dump_transition() returns a list of list. [[s,a,s_next,r]]
        """

        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.prime_transitions += vehicle.dump_prime_transitions()

    def store_f_action_from_veh(self):
        """
        vehicle.dump_transition() returns a list of list. [[s,a,s_next,r]]
        """
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.f_transitions += vehicle.dump_f_transitions()
                self.fo_transitions+=vehicle.dump_fo_transitions()

    def last_step_transactions(self,tick):
        #store all the transactions from the vehicles:
        for hex in self.hex_zone_collection.values():
            for vehicle in hex.vehicles.values():
                self.all_transitions += vehicle.dump_last(tick)

    def dump_transition_to_dqn(self):
        """
        convert transitions to batches of state, action, next_state, and off-duty flag.
        :return:
        """
        state, action, next_state, reward, terminate_flag, time_steps, num_valid_relos_ = None, None, None, None, None, None, None
        all_transitions = np.array(self.all_transitions, dtype=object)
        if len(all_transitions) > 0:
            # print('First row of Transitions are:',all_transitions[1])
            [state, action, next_state, reward, terminate_flag, time_steps] = [all_transitions[:, i] for i in range(6)]
        if state is not None:
            num_valid_relos_ = [len([0] + self.hex_zone_collection[state_i[1]].neighbor_hex_id) for state_i in
                                next_state]  # valid relocation for next state.
        return state, action, next_state, reward, terminate_flag, time_steps, num_valid_relos_

    def dump_prime_action_to_dqn(self):
        """
        convert transitions to batches of state, action, next_state, and off-duty flag.
        :return:
        """
        state, action, next_state,trip_flag, time_steps, num_valid_relos_ = None, None, None, None, None, None
        all_transitions = np.array(self.prime_transitions, dtype=object)
        if len(all_transitions) > 0:
            # print('First row of Transitions are:',all_transitions[1])
            [state, action, next_state,trip_flag, time_steps] = [all_transitions[:, i] for i in range(5)]
        if state is not None:
            num_valid_relos_ = [len([0] + self.hex_zone_collection[state_i[1]].neighbor_hex_id) for state_i in
                                next_state]  # valid relocation for next state.
        return state, action, next_state, trip_flag, time_steps, num_valid_relos_

    def dump_f_transitions(self):
        state,next_state,on_opt=None,None,None
        all_transitions=np.array(self.f_transitions,dtype=object)
        if len(all_transitions)>0:
            [state,next_state,on_opt]=[all_transitions[:,i] for i in range(3)]
        return state,next_state,on_opt

    def dump_fo_transitions(self):
        state,next_state=None,None
        all_transitions=np.array(self.fo_transitions,dtype=object)
        if len(all_transitions)>0:
            [state,next_state]=[all_transitions[:,i] for i in range(2)]
        return state,next_state


    def get_current_time(self):
        return self.__t

    def summarize_metrics(self,demand_supply_gap_file, charging_od_file, cruising_od_file, matching_od_file):

        all_vehicles = [vehicle for hex in self.hex_zone_collection.values() for vehicle in hex.vehicles.values() if
                        vehicle.state.agent_type == agent_codes.dqn_agent]
        num_idle = sum([veh.state.status == status_codes.V_IDLE for veh in all_vehicles])  # self.total_idled_vehicles
        num_serving = sum([veh.state.status == status_codes.V_OCCUPIED for veh in all_vehicles])
        num_cruising = sum([veh.state.status == status_codes.V_CRUISING for veh in all_vehicles])
        num_assigned = sum([veh.state.status == status_codes.V_ASSIGNED for veh in all_vehicles])
        num_offduty = sum([veh.state.status == status_codes.V_OFF_DUTY for veh in all_vehicles])
        num_tobedisptached = sum([veh.state.status == status_codes.V_TOBEDISPATCHED for veh in all_vehicles])
        num_waitpile = sum([len(cs.queue) for cs in self.charging_station_collections])
        average_reduced_SOC = np.mean([veh.mileage_per_charge_cycle for veh in all_vehicles if veh.state.status == status_codes.V_WAITPILE])
        num_charging = sum(
            [sum([1 for p in cs.piles if p.occupied == True]) for cs in self.charging_station_collections])
        n_matches = self.num_match
        average_cumulated_earning = np.mean([veh.total_earnings for veh in all_vehicles])
        average_ttreward=np.mean([veh.cum_reward for veh in all_vehicles])
        average_idle_dist = np.mean([veh.idle_dist for veh in all_vehicles])
        total_num_arrivals = self.total_num_arrivals
        total_removed_passengers = self.total_num_longwait_pass + self.total_num_served_pass
        total_num_longwait_pass = self.total_num_longwait_pass
        total_num_served_pass = self.total_num_served_pass
        if int(self.__t % 300) ==0:
            print('current tick is {}, recording data!!!!!!!!!!!'.format(self.__t//60))
            [demand_supply_gap_file.writelines('{},{},{}\n'.format(self.__t,hex.hex_id, (hex.arrivals[-1] - len(hex.vehicles.keys())) )) for hex in self.hex_zone_collection.values()]
            [charging_od_file.writelines('{},{},{}\n'.format(od[0], od[1], od[2])) for veh in all_vehicles for od in veh.charging_od_pairs]
            [cruising_od_file.writelines('{},{},{}\n'.format(od[0], od[1], od[2])) for veh in all_vehicles for od in veh.repositioning_od_pairs]
            [matching_od_file.writelines('{},{},{}\n'.format(od[0], od[1], od[2])) for veh in all_vehicles for od in veh.matching_od_pairs]
        [veh.reset_od_pairs() for veh in all_vehicles]

        return num_idle, num_serving, num_charging, num_cruising, n_matches, average_idle_dist, \
               total_removed_passengers, num_assigned, num_waitpile, num_tobedisptached, average_ttreward, \
               average_reduced_SOC, total_num_longwait_pass, total_num_served_pass, average_cumulated_earning

    def reset_storage(self):
        self.all_transitions = []
        self.prime_transitions=[]
        self.f_transitions=[]
        self.fo_transitions=[]
        self.global_state_buffer=dict()


    def reset(self,start_time,timestep,seed=None):
        #restart the simulator, but assuming the time of previous end time
        self.reset_time(start_time, timestep)
        self.last_vehicle_id = 1
        self.vehicle_queue = []
        # DQN for getting actions and dumping transitions
        self.all_transitions = [] #clear the memories
        self.num_match = 0
        self.total_num_arrivals = 0
        self.total_num_removed_pass = 0
        self.total_num_served_pass = 0
        self.total_num_longwait_pass = 0
        self.total_idled_vehicles = 0
        self.current_dummyV = 0
        self.current_dqnV = 0

        #reset random seed
        if seed is None:
           local_rand_seed=np.random.randint(4)
        else:
            local_rand_seed=seed


        old=random.getstate()

        filename = "../data/daily_trips/{}.csv".format(local_rand_seed)
        n = sum(1 for line in open(filename)) - 1  # number of records in file (excludes header)
        s = int(n*0.3)  # desired sample size
        random.seed(10)
        skip = sorted(
            random.sample(range(1, n + 1), n - s))  # the 0-indexed header will not be included in the skip list
        random.setstate(old)

        alltrips = pd.read_csv('../data/daily_trips/{}.csv'.format(local_rand_seed),skiprows=skip, dtype=int).groupby('origin_hid')

        #reset hex zones
        for hex in self.hex_zone_collection.values():
            h_idx=hex.hex_id
            try:
                demand_df=alltrips.get_group(int(h_idx))
            except KeyError:
                demand_df=pd.DataFrame({'tick':[]})
            hex.all_trips=demand_df.groupby('tick')
            hex.seed=local_rand_seed
            hex.reset()
        #reset charging stations
        for cs in self.charging_station_collections:
            cs.reset()

        hex_ids=[hex.hex_id for hex in self.hex_zone_collection.values()]
        # init entering-market vehicle queue
        vehicle_hex_ids = [hex_ids[i] for i in
                           np.random.choice(len(hex_ids), size=N_VEHICLES)]  # , p=p)]

        n_vehicles = len(vehicle_hex_ids)
        vehicle_ids = range(self.last_vehicle_id, self.last_vehicle_id + n_vehicles)
        self.last_vehicle_id += n_vehicles
        entering_time = np.random.uniform(self.__t, self.__t + ENTERING_TIME_BUFFER, n_vehicles).tolist()
        q = sorted(zip(entering_time, vehicle_ids, vehicle_hex_ids))
        self.vehicle_queue = q

        #initialize a random seed
        random.seed()
        print('simulator reset complete')