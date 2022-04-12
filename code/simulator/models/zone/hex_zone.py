# from simulator.services.routing_service import RoutingEngine
from config.hex_setting import OFF_DURATION, RELOCATION_DIM
from novelties import status_codes
from simulator.models.customer.customer import Customer
from simulator.models.customer.request import request
from collections import defaultdict
import numpy as np
import contextlib
from novelties.pricing.price_calculator import calculate_price
def weighted_random(w, n):
    cumsum = np.cumsum(w)
    rdm_unif = np.random.rand(n)
    return np.searchsorted(cumsum, rdm_unif)


@contextlib.contextmanager
def local_seed(seed):
    # this defines a local random seed funciton, and let the simulator to resume previous random seed
    state = np.random.get_state()
    np.random.seed(seed)  # set seed
    try:
        yield
    finally:
        np.random.set_state(state)  # put the state back on

class hex_zone:

    def __init__(self, hex_id, coord,row_col_coord, coord_list, match_zone, neighbors, charging_station_ids, charging_hexes,
                 charging_coords, od_split, trip_time, t_unit, epoch_length,seed,demand_df):
        """
        hex_id: id of the hexagon zone in the shapefile
        coord: lon and lat values
        arrival_rate: number of arrivals per tick
        neighbors: adjacent hexagons' ids
        charging_station_ids: nearest 5 charging station ids
        charging_coords: list of coordinates of the 5 charging stations
        epoch_length: total ticks per epoch of simulation: 60 * 24 * SIM_DAYS
        """
        self.seed=seed
        self.hex_id = hex_id
        self.match_zone_id = match_zone
        self.lon, self.lat = coord
        self.row_id, self.col_id = row_col_coord
        self.coord_list = coord_list  # this is the list for all the lon lat coordinates of the hexagons
        od_split = np.reshape(od_split, (od_split.shape[0], od_split.shape[-1]))
        trip_time = np.reshape(trip_time, (trip_time.shape[0], trip_time.shape[-1]))  # remove one of the dimension
        self.arrival_rate = np.sum(od_split,
                                   axis=-1).flatten() / t_unit  # now this becomes a  hour by 1 array,and we convert this to each tick of demand!
        self.next_arrivals = None
        # 1 by N matrix
        self.od_ratio = od_split
        self.trip_time = trip_time

        # the following two defines the actions
        self.neighbor_hex_id = neighbors  # length may vary
        self.nearest_cs = charging_station_ids
        self.charging_hexes = charging_hexes
        self.charging_station_loc = charging_coords
        self.n_charges=0 #number of charging stations
        self.passengers = defaultdict()
        self.vehicles = defaultdict()
        self.served_num = 0
        self.removed_passengers = 0
        self.served_pass = 0
        self.total_served=0
        self.longwait_pass = 0
        self.veh_waiting_time = 0
        self.total_pass = 0  # this also servers as passenger id

        self.t_unit = t_unit  # number of ticks per hour
        self.epoch_length = epoch_length
        self.q_network = None
        self.narrivals = 0
        self.next_narrivals = 0
        self.all_trips=demand_df.groupby('tick')
        # initialize the demand for each hexagon zone
        self.init_demand()

    def reset(self):
        #reinitialize the status of the hex zones
        self.passengers.clear()
        self.vehicles.clear()
        self.served_num = 0
        self.removed_passengers = 0
        self.served_pass = 0
        self.longwait_pass = 0
        self.veh_waiting_time = 0
        self.narrivals = 0
        self.next_narrivals = 0
        # initialize the demand for each hexagon zone
        self.init_demand()

    def init_demand(self):
        '''
        todo: generate all the initial demand for each hour. Fix a local random generator to reduce randomness
        :return:
        '''
        # copy the arrival rate list multiple times!
        self.arrivals=[]
        self.destinations=[]
        for tick in range(1440):
            try:
                td=self.all_trips.get_group(tick)['destination_hid'].tolist()
            except KeyError:
                td=[]
            self.arrivals.append(len(td))
            self.destinations.append(td)
        self.arrivals.append(0)
        self.destinations.append([])
        self.arrivals.reverse()
        self.destinations.reverse()
        # high=[i+np.ceil(0.1*i) for i in low]
        # with local_seed(self.seed):
        #     self.arrivals = np.random.poisson(list(self.arrival_rate) * int(
        #         max(1, np.ceil(self.epoch_length / len(self.arrival_rate) / self.t_unit))),
        #                                       size=(self.t_unit, len(self.arrival_rate) * int(max(1, np.ceil(
        #                                           self.epoch_length / len(self.arrival_rate) / self.t_unit)))))
        #
        #     self.arrivals = self.arrivals.flatten('F')  # flatten by columns-major
        #     self.arrivals = list(self.arrivals)#inverse
        #     self.arrivals.reverse()
        #     self.arrivals.append(0)  #for output plotting purposes.
            # self.next_arrivals = list(self.arrivals[1:] + [self.arrivals[0]])

    def add_veh(self, veh):  # vehicle is an object
        '''
        add and remove vehicles by its id
        id contained in veh.state
        :param veh:
        :return:
        '''
        self.vehicles[veh.state.vehicle_id] = veh

    def remove_veh(self, veh):
        self.vehicles.pop(veh.state.vehicle_id)  # remove the vehicle from the list

    def demand_generation(self,tick):
        destinations=self.destinations.pop()
        hour = tick // (
                self.t_unit * 60) % 24
        self.arrivals.pop()
        for i in range(len(destinations)):
            # r={'id':self.total_pass,'origin_id':self.hex_id, 'origin_lat':self.lat, 'origin_lon':self.lon, \
            #    'destination_id':destinations[i], 'destination_lat':self.coord_list[destinations[i]][1], 'destination_lon':self.coord_list[destinations[i]][0], \
            #        'trip_time':self.trip_time[hour,destinations[i]],'request_time':tick}
            # r=request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)
            self.passengers[(self.hex_id, self.total_pass)] = Customer(
                request(self.total_pass, self.hex_id, (self.lon, self.lat), destinations[i],
                        self.coord_list[destinations[i]], self.trip_time[hour, destinations[i]],
                        tick))  # hex_id and pass_id create a unique passenger identifier
            self.total_pass += 1

    # def demand_generation(self, tick):  # the arrival of passenger demand
    #     '''
    #     todo: pop n_arrivals in the next hour
    #     :param tick: current time
    #     :return:
    #     '''
    #     with local_seed(int(self.seed*1e4)+(tick//60)%(24*60)):  # fix the random seed
    #         hour = tick // (
    #                 self.t_unit * 60) % 24  # convert into the corresponding hours. Tick are seconds and is incremeted by 60 seconds in each iteration
    #         # print('hour {}  tick{}'.format(hour, tick))
    #         narrivals = self.arrivals.pop()  # number of arrivals
    #
    #         self.narrivals = narrivals
    #         destination_rate = self.od_ratio[hour, :]
    #         if narrivals > 0 and sum(destination_rate) > 0:
    #             # print('Tick {} hour {} and tunit{}'.format(tick,hour,self.t_unit))
    #             destination_rate = destination_rate / sum(destination_rate)  # normalize to sum =1
    #             # destinations = np.random.choice(destination_rate.shape[-1], p=destination_rate,
    #             #                                 size=narrivals)  # choose the destinations
    #             destinations=weighted_random(destination_rate,narrivals)
    #             for i in range(narrivals):
    #                 # r={'id':self.total_pass,'origin_id':self.hex_id, 'origin_lat':self.lat, 'origin_lon':self.lon, \
    #                 #    'destination_id':destinations[i], 'destination_lat':self.coord_list[destinations[i]][1], 'destination_lon':self.coord_list[destinations[i]][0], \
    #                 #        'trip_time':self.trip_time[hour,destinations[i]],'request_time':tick}
    #                 # r=request(self.total_pass, self.hex_id, (self.lon,self.lat,), destinations[i], self.coord_list[destinations[i]],self.trip_time[hour,destinations[i]],tick)
    #                 self.passengers[(self.hex_id, self.total_pass)] = Customer(
    #                     request(self.total_pass, self.hex_id, (self.lon, self.lat), destinations[i],
    #                             self.coord_list[destinations[i]], self.trip_time[hour, destinations[i]],
    #                             tick))  # hex_id and pass_id create a unique passenger identifier
    #                 self.total_pass += 1
    #
    #     return

    def remove_pass(self, pids):  # remove passengers
        '''
        Remove passengers by key_id
        :return:
        '''
        [self.passengers.pop(pid) for pid in pids]

    def update_passengers(self):
        """
        code for updating the passenger status / or remove them if picked up
        """
        remove_ids = []
        self.longwait_pass=0
        self.served_pass=0
        for pid in self.passengers.keys():
            if self.passengers[pid].matched==True:
                self.served_pass +=1
                remove_ids.append(pid)
            elif self.passengers[pid].waiting_time >= self.passengers[pid].max_tolerate_delay:  # remove passengers after 10 ticks.
                self.longwait_pass +=1
                remove_ids.append(pid)
            else:
                self.passengers[pid].waiting_time += self.t_unit  # update waiting time

        self.removed_passengers += len(remove_ids)
        self.remove_pass(remove_ids)

    def remove_matched(self):
        remove_ids = []
        self.longwait_pass=0
        self.served_pass=0
        for pid in self.passengers.keys():
            if self.passengers[pid].matched==True:
                self.served_pass +=1

        self.removed_passengers += len(remove_ids)
        self.remove_pass(remove_ids)

    def vehicle_dispatch(self, tick):
        """
        Dispatch the vehicles. This step follows from matching step
        :param tick:
        :return:
        """
        if len(self.vehicles.keys()) == 0:
            # no vehicle to dispatch
            return
        tbd_vehicles = {key: vehicle for key, vehicle in self.vehicles.items() if
                        vehicle.state.status == status_codes.V_TOBEDISPATCHED}
        self.dispatch(tbd_vehicles, tick)

    def dispatch(self, vehicles, current_time):
        '''
        todo: fulfill OFF_DUTY cycle: specify when to trigger OFF_Duty status 
        :vehicles: is dict with key and values
        '''
        for vehicle in vehicles.values():
            action_id = vehicle.state.dispatch_action_id
            offduty = 0  # actions are attached before implementing dispatch
            if offduty:
                off_duration = np.random.randint(OFF_DURATION / 2, OFF_DURATION * 3 / 2)
                # self.sample_off_duration()   #Rand time to rest
                vehicle.take_rest(off_duration)
            else:
                # Get target destination and key to cache
                target, charge_flag, target_hex_id, cid = self.convert_action_to_destination(vehicle, action_id)
                vehicle.state.destination_hex = target_hex_id
                vehicle.state.origin_hex = vehicle.state.hex_id
                vehicle.state.need_route = True
                if charge_flag == 0:
                    if isinstance(vehicle.state.current_hex, list): print('relo index wrong')
                    vehicle.cruise(target, action_id, current_time,target_hex_id)
                    if action_id>0 and target_hex_id==vehicle.state.hex_id:
                        print('We are having mistakes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        print('action={}, hex={}, neighbors={}, from={}, to={}'.format(action_id, self.hex_id, self.neighbor_hex_id,vehicle.state.hex_id,target_hex_id))
                else:
                    if isinstance(vehicle.state.current_hex, list): print('charging index wrong')
                    vehicle.head_for_charging_station(cid, target, current_time, action_id)
                    # print('The charing station ID for vehicle {} is {}'.format(vehicle.state.vehicle_id, cid))

    def convert_action_to_destination(self, vehicle, action_id):
        '''
        vehicle: objects
        action_id: action ids from 0-11, pre-derived from DQN
        '''
        cid = None # charging_station_id
        valid_relocation_space = self.neighbor_hex_id
        try:
            target_hex_id = valid_relocation_space[action_id]
            lon, lat = self.coord_list[target_hex_id]
            charge_flag = 0
        except IndexError:
            print(vehicle.state.hex_id,self.neighbor_hex_id,action_id)
            cid = self.nearest_cs[action_id - RELOCATION_DIM]
            target_hex_id = self.charging_hexes[cid]
            lon, lat = self.charging_station_loc[self.nearest_cs[action_id - RELOCATION_DIM]]
            charge_flag = 1
        target = (lon, lat)
        return target, charge_flag, target_hex_id, cid

    def get_num_arrivals(self):
        return self.narrivals

    def get_next_num_arrivals(self):
        return self.next_narrivals

    def get_num_removed_pass(self):
        return self.removed_passengers

    def get_num_served_pass(self):
        return self.served_pass

    def get_num_longwait_pass(self):
        return self.longwait_pass

    def get_stay_idle_veh_num(self):
        idle_vehicles = {key: vehicle for key, vehicle in self.vehicles.items() if
                        vehicle.state.status in [status_codes.V_STAY, status_codes.V_IDLE]}
        return len(idle_vehicles)

    def get_passenger_num(self):
        passenger_dict = {key: passengers for key, passengers in self.passengers.items()}
        return len(passenger_dict)


