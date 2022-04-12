from novelties import status_codes
from .vehicle_state import VehicleState
from .vehicle_behavior import Occupied, Cruising, Idle, Assigned, OffDuty, Waytocharge, Waitpile, Charging, \
    Tobedispatched, Stay, Tobecruised
from logging import getLogger
from collections import deque
import numpy as np
import random
from config.hex_setting import SOC_PENALTY, MILE_PER_METER, SIM_ACCELERATOR, BETA_CHARGE_COST, BETA_EARNING, BETA_DIST, \
    BETA_TIME, IDLE_DURATION, QUICK_END_CHARGE_PENALTY, BETA_RANGE_ANXIETY, PER_TICK_DISCOUNT_FACTOR, OPTION_DIM


# from simulator.services.osrm_engine import OSRMEngine
class Vehicle(object):
    behavior_models = {
        status_codes.V_IDLE: Idle(),
        status_codes.V_CRUISING: Cruising(),
        status_codes.V_OCCUPIED: Occupied(),
        status_codes.V_ASSIGNED: Assigned(),
        status_codes.V_OFF_DUTY: OffDuty(),
        status_codes.V_WAYTOCHARGE: Waytocharge(),
        status_codes.V_CHARGING: Charging(),
        status_codes.V_WAITPILE: Waitpile(),
        status_codes.V_TOBEDISPATCHED: Tobedispatched(),
        status_codes.V_STAY: Stay(),
        status_codes.V_TOBECRUISED: Tobecruised()
    }
    def __init__(self, vehicle_state,with_option,local_matching,with_charging,tick):
        if not isinstance(vehicle_state, VehicleState):
            raise ValueError
        self.state = vehicle_state
        self.customer = None  # the passenger matched and to be picked up
        self.__behavior = self.behavior_models[vehicle_state.status]
        self.__customers = []  # A vehicle can have a list of cusotmers
        self.__customers_ids = []
        self.__charging_station = []
        self.reward = 0
        self.cum_reward=0
        self.on_random=1
        self.flag = 0
        self.option_cruise=0 #ticks cruising so far
        self.time_ticks = [0]
        self.start_tick = tick
        self.end_tick = tick
        self.start_serve_tick = tick
        self.start_wait_tick = tick
        self.mileage_per_charge_cycle = 0
        self.pickup_distance = 0
        self.dropoff_distance = 0
        self.pickup_duration = 0
        self.dropoff_duration = 0
        self.non_charging_mask = 1 if with_charging else 0  # mask out SOC consumption
        self.local_matching = 1 if local_matching else 0
        self.with_option = 1 if with_option else 0
        self.non_included_od_pair = []
        self.require_online_query = False
        self.matching_od_pairs = []
        self.charging_od_pairs = []
        self.repositioning_od_pairs = []
        self.discount_factor = PER_TICK_DISCOUNT_FACTOR  # 0.99
        self.decay_lambda = 1.0
        self.charging_dicounted_reward = 0
        self.assigned_option = -1 #no option assigned at the beginning
        self.option_contd=0 #1 if continuing from previous option, 1 otherwise
        self.recent_transitions = []  # deque(maxlen=10), store options
        self.prime_transition=[] #store primitive action pairs
        self.f_transition=[] #store transitions for calculating f
        self.fo_transition=[] #store transitions for occupied trips
        self.cruise_trajectory=[]
        self.trajectory_transition=[]
        self.total_earnings = 0
        self.working_time = 0
        self.idle_time=0
        self.idle_dist=0


        # self.first_dispatched = 0
        # self.pickup_time = 0
        self.q_action_dict = {}
        # self.duration = [[0] for _ in len(self.behavior_models)]     # Duration for each state
        self.charging_wait = 0
        self.start_state = None
        self.end_state = None  # [0, self.state.hex_id, self.state.SOC]
        self.action = 0 #this is the action to be executed
        self.assigned_action=0 #this can be the same as assigned policy, or different

        # update the information of vehicles, e.g. change the current hex zone location, this is not paralleled
    def update_info(self, hex_zones, routes, hex_coords_list, tick):
        # self.working_time += timestep
        if self.state.need_route:
            # print('OD hexes are:',self.state.hex_id,self.state.current_hex)
            try:
                self.state.route = routes[(self.state.hex_id, self.state.destination_hex)]['route']
                self.state.time_to_destination = routes[(self.state.hex_id, self.state.destination_hex)]['travel_time']
                self.state.travel_dist = routes[(self.state.hex_id, self.state.destination_hex)]['distance']
                # if sum(self.state.travel_dist)==0 and self.state.hex_id!=self.state.destination_hex:
                #     print('no distance, origin={},destination={}'.format(self.state.hex_id,self.state.destination_hex))
            except:  # in case the route is non-existing...we make the vehicle to stay!
                print('Issue processing trips',self.state.hex_id, self.state.destination_hex,self.state.status)

            self.state.need_route = False
            self.state.need_interpolate = True  # ask the remote to interpolate the coords
        if self.state.hex_id != self.state.current_hex:
            self.update_veh_hex(hex_zones, tick)
            # update each vehicle if the new location (current_hex) is different from its current one (hex_id)

    def update_veh_hex(self, hex_zones, tick):
        # record od pairs first
        # if self.state.status == status_codes.V_ASSIGNED:
        #     self.matching_od_pairs.append([tick, self.state.hex_id, self.state.destination_hex])
        # if self.state.status == status_codes.V_WAYTOCHARGE:
        #     self.charging_od_pairs.append([tick, self.state.hex_id, self.state.destination_hex])
        # if self.state.status in [status_codes.V_CRUISING, status_codes.V_STAY]:
        #     self.repositioning_od_pairs.append([tick, self.state.hex_id, self.state.destination_hex])
        hex_zones[self.state.hex_id].remove_veh(self)
        hex_zones[self.state.current_hex].add_veh(self)
        self.state.hex_id = self.state.current_hex

    def reset_od_pairs(self):
        self.matching_od_pairs = []
        self.charging_od_pairs = []
        self.repositioning_od_pairs = []


    def step(self, timestep, timetick,hex_zones):
        """
        step function for vehicle
        :param timetick:
        :param timestep: 60 sec
        :return:
        """
        if self.state.need_interpolate:  # make interpolation of the ticks, distance and coordinates
            self.location_interp(timestep)
            self.state.need_interpolate = False

        try:
            self.__behavior.step(self, timetick,hex_zones)
        except:
            print('Unable to perform behavior step!')
            logger = getLogger(__name__)
            logger.error(self.state.to_msg())
            raise
        # self.log_info(timetick)
    def dump_states(self, tick):
        state_rep = [tick, self.state.hex_id, self.state.SOC]
        return state_rep

    def dump_transitions(self):
        exported_trans = []
        if len(self.recent_transitions) > 2:
            exported_trans = self.recent_transitions[:-2]
            self.recent_transitions = [self.recent_transitions[-1],self.recent_transitions[-2]]
        return exported_trans

    def dump_trajectories(self):
        exported_trajectory=[]
        if len(self.trajectory_transition)>1:
            exported_trajectory=self.trajectory_transition[:-1]
            self.trajectory_transition=[self.trajectory_transition[-1]]
        return exported_trajectory

    def dump_prime_transitions(self):
        #store all the prime transitions from the vehicle
        exported_trans = []
        if len(self.prime_transition) > 2:
            exported_trans = self.prime_transition[:-2]
            self.prime_transition=[self.prime_transition[-1],self.prime_transition[-2]]
        return exported_trans


    def dump_f_transitions(self):
        #store all the prime transitions from the vehicle
        exported_trans = []
        if len(self.f_transition) > 2:
            exported_trans = self.f_transition[:-2]
            self.f_transition=[self.f_transition[-1],self.f_transition[-2]]
        return exported_trans

    def dump_fo_transitions(self):
        #store all the prime transitions from the vehicle
        exported_trans = []
        if len(self.fo_transition) > 1:
            exported_trans = self.fo_transition[:-1]
            self.fo_transition=[self.fo_transition[-1]]
        return exported_trans

    def get_mile_of_range(self):
        return self.state.set_range()

    def get_SOC(self):
        return self.state.SOC

    def get_target_SOC(self):
        return self.state.target_SOC

    def send_to_dispatching_pool(self, action_select,action_execute,opts=-1,ctd_opt=0):
        self.state.dispatch_action_id = action_execute
        self.action=action_execute
        self.assigned_action=action_select
        self.assigned_option=opts
        self.option_contd=ctd_opt
        self.__change_to_tobedispatched()

    def location_interp(self, t_unit=60):
        """
        Interpolate the per tick travel distance and location based on the corresponding time unit
        Input: self.state.route: a list of coordinates
        self.state.time_to_destination: a list segment travel time
        self.state.travel_dist: a list of segment travel distance
        :return:
        """
        ## two possible cases
        # case 1. Relocation to different locations
        if isinstance(self.state.time_to_destination, list):
            if self.state.time_to_destination[0]>0: #valid trajectories
                #this is for parsed trajectories
                self.state.per_tick_coords=list(self.state.route)
                self.state.per_tick_dist=list(self.state.travel_dist)
                self.time_ticks=list(self.state.time_to_destination)
                # total_tt = sum(self.state.time_to_destination)
                # cum_time = np.cumsum(self.state.time_to_destination)
                # cum_dist = np.cumsum(self.state.travel_dist)
                # time_ticks = [i * t_unit for i in
                #               range(1, int(total_tt // t_unit + 1))]  # the time steps to query from per simulation tick
                # self.time_ticks = [t_unit for _ in range(len(time_ticks))]
                # if total_tt % t_unit > 0:
                #     time_ticks.append(total_tt)  # add the final step
                #     self.time_ticks.append(total_tt % t_unit)
                #
                # per_tick_dist = np.interp(time_ticks, cum_time, cum_dist)
                # # if len(per_tick_dist)>=0:
                # try:
                #     per_tick_dist = [per_tick_dist[0]] + np.diff(per_tick_dist).tolist()
                # except IndexError:
                #     print('tick dist:', self.state.travel_dist)
                #     print('tick time:', self.state.time_to_destination)
                #
                # lons = [self.state.route[0][0][0]] + [coord[1][0] for coord in self.state.route]
                # lats = [self.state.route[0][0][1]] + [coord[1][1] for coord in self.state.route]
                #
                # cum_time = cum_time.tolist()
                # per_tick_lon = np.interp(time_ticks, [0] + cum_time, lons)
                # per_tick_lat = np.interp(time_ticks, [0] + cum_time, lats)
                # per_tick_lon = per_tick_lon.tolist()
                # per_tick_lat = per_tick_lat.tolist()
                #
                # per_tick_coords = [[lon, lat] for lon, lat in zip(per_tick_lon, per_tick_lat)]

                # self.state.per_tick_coords = per_tick_coords
                # self.state.per_tick_dist = per_tick_dist

            # empty routes (not feasible)
            else:
                # we keep the vehicle to stay for a few ticks
                '''
                todo: make the K value a predefined parameter from the config file
                '''
                # print('Vehicle status for generating routes:',self.state.status,self.state.hex_id,self.state.destination_hex)
                if self.state.status<=1 or self.state.status==9:  #9 is the stay
                    K = 1 #IDLE_DURATION
                else:
                    K=1 #no wait , for charging, dispatching, assigned status
                self.time_ticks = [t_unit for _ in range(K)]
                # print('The vehicle is in stay state, the status code is {}, the time to destination is {}'.format(
                #     self.state.status, self.time_ticks))
                # print('Trip origin is {}, Tri destination is, the route is {}'.format(
                #     self.state.hex_id, self.state.destination_hex,self.state.route))
                self.state.per_tick_dist = [300 for _ in range(K)] #500 is the cruising distance per minutes
                self.state.per_tick_coords = [[self.state.route[0][0][0], self.state.route[0][0][1]] for _ in range(K)]

            #check if correct
            if self.time_ticks[0]==0:
                asdasdasdasd
        #
        # # case 2: stay or NO ROUTE IS AVAILABLE !
        # else:
        #     # we keep the vehicle to stay for a few ticks
        #     '''
        #     todo: make the K value a predefined parameter from the config file
        #     '''
        #     K = IDLE_DURATION
        #     self.time_ticks = [t_unit for _ in range(K)]
        #     self.state.per_tick_dist = [0 for _ in range(K)]
        #     self.state.per_tick_coords = [[self.state.lon, self.state.lat] for _ in range(K)]

        assert len(self.state.per_tick_coords) == len(self.state.per_tick_dist)
        assert len(self.state.per_tick_dist) == len(self.time_ticks)

    def cruise(self, target_hex_coord, action, tick,target_hex_id):
        '''
        :param target_hex_coord:
        :param tick: used by park to store states.
        '''

        self.cruise_dest=target_hex_id
        if self.state.id==1:
            print('Start cruising from {} to {}'.format(self.state.hex_id,target_hex_id))
        assert self.__behavior.available
        self.prime_start_tick = tick

        self.prime_start_state = [self.prime_start_tick, self.state.hex_id, self.state.SOC]
        self.action=action
        self.f_start_state = [tick, self.state.hex_id, self.state.SOC]

        # only dump after one full transition is finished (and the new reward for matching is attached)
        # SOC follows exponential function:
        if self.assigned_option==-1: #no option assigned
            self.reward = - BETA_RANGE_ANXIETY * self.compute_mileage_anxiety(self.state.SOC) if self.state.SOC<=0.4 else 0
            self.cum_reward+=self.reward
            self.decay_lambda = 1.0
            self.start_tick = tick
            self.option_cruise=0
            self.start_state = [self.start_tick, self.state.hex_id, self.state.SOC]
        else:
            if self.option_contd == 0: #option assigned but started a new option
                self.reward = - BETA_RANGE_ANXIETY * self.compute_mileage_anxiety(
                    self.state.SOC) if self.state.SOC <= 0.4 else 0
                self.cum_reward+=self.reward
                self.option_cruise=1
                self.decay_lambda = 1.0
                self.start_tick = tick
                self.start_state = [self.start_tick, self.state.hex_id, self.state.SOC]
            else: #option assigned, and continue flag is active
                self.reward += - BETA_RANGE_ANXIETY * self.compute_mileage_anxiety(self.state.SOC) if self.state.SOC<=0.4 else 0
                self.cum_reward+=self.compute_mileage_anxiety(self.state.SOC) if self.state.SOC<=0.4 else 0
                self.option_cruise+=1

        self.opt_start_state = [tick, self.state.hex_id, self.state.SOC]
        self.opt_reward = self.reward  # record current reward
        self.opt_start_tick = tick

        self.dist_to_destination=sum(self.state.per_tick_dist)
        self.time_to_destination=sum(self.time_ticks)

        self.__set_destination(target_hex_coord)
        if action == 0:
            self.__change_to_stay()
        else:
            self.__change_to_cruising()

    def save_terminal(self,tick):
        #for vehicle in terminal states, use the save function
        # need to check if the vehicle is at the terminal state
        if self.assigned_option>=0:
            #save terminal with probability
                if (tick+60)//60%1440==0:
                    terminate=True
                else:
                    terminate=False
                self.start_serve_tick = tick  # start serve tick is for full service cycle if being matched next
                self.end_tick = tick
                self.end_state = [self.end_tick, self.state.hex_id, self.state.SOC]
                self.f_end_state=[tick, self.state.hex_id,self.state.SOC]
                # self.f_transition.append([self.opt_start_state, self.f_end_state])
                steps = (self.end_tick - self.start_tick) // 60
                rate = (self.discount_factor ** steps - 1) / (steps * (self.discount_factor - 1))

                # opt_reward = sum([r * (self.discount_factor ** s) for r, s in zip(self.decay_reward, self.decay_coeff)])
                self.recent_transitions.append(
                        [self.start_state, self.assigned_action, self.end_state, rate*self.reward, terminate,
                         (self.end_tick - self.start_tick) // 60 ])
                # print('Line 351 Intermediate recording, current action is {}, Recorded action is {}, selected option is {}, option flag is {}'.format((self.action,self.state.dispatch_action_id),action, self.assigned_action,self.assigned_option))
                # self.f_transition.append([self.opt_start_state, self.f_end_state])
                self.assigned_option = -1
                self.option_contd = 0
                self.option_cruise=0 #reset number of option steps


    def park(self, tick,hex_zones):
        ''':param
        '''
        #update the hex zones

        hex_zones[self.state.hex_id].remove_veh(self)
        hex_zones[self.cruise_dest].add_veh(self)
        self.state.hex_id = self.cruise_dest
        self.start_serve_tick = tick  # start serve tick is for full service cycle if being matched next

        if (tick+60)//60%1440==0:
            terminate=True
            self.end_tick=tick
        else:
            terminate=False
            self.end_tick = tick + 60

        self.repositioning_od_pairs.append([self.opt_start_state[0], self.opt_start_state[1], self.cruise_dest, self.dist_to_destination,self.time_to_destination,
                                            self.assigned_action])  # relocation start time, relocation from, relocation to, travel dist, travel time, use option or not and which option (if used)

        self.end_state = [self.end_tick, self.state.hex_id, self.state.SOC]
        self.f_end_state=[self.end_tick, self.state.hex_id,self.state.SOC]
        trip_flag=False
        if self.prime_start_state is not None:
            self.prime_transition.append([self.prime_start_state,self.action,self.end_state,trip_flag, (self.end_tick - self.prime_start_tick)//60])
        else:
            print('Start state is None and I can not pop in new rewards')

        self.cruise_trajectory.append([self.f_start_state,self.f_end_state,(self.end_tick - self.prime_start_tick)//60,self.action,False]) #no trip
        if len(self.cruise_trajectory)>9:
            self.trajectory_transition.append(self.cruise_trajectory)
            self.cruise_trajectory=[]


        if self.state.SOC < 0:
            self.reward -= self.decay_lambda*SOC_PENALTY
            self.cum_reward-=SOC_PENALTY
        if self.assigned_option==-1: #no assigned policy
            if self.on_random: self.f_transition.append([self.f_start_state, self.f_end_state,0])
            steps = max((self.end_tick - self.start_tick) // 60,1)
            rate = (self.discount_factor ** steps - 1) / (steps * (self.discount_factor - 1))
            self.recent_transitions.append([self.start_state, self.action, self.end_state, self.reward*rate,terminate, (self.end_tick - self.start_tick)//60])
            if self.state.id == 1:
                print('Non---APT----PARK: Park recorded,  action: {}'.format(self.recent_transitions[-1]))
            # print('Line 331 End of option recording, current action is {}, Recorded action is {}, selected option is {}, option flag is {}'.format(
            #     (self.action,self.state.dispatch_action_id),self.assigned_action, self.assigned_action, self.assigned_option))
        else:
                #record the primitive transition followed by the options
                steps = (self.end_tick - self.start_tick) // 60 + 1
                rate = (self.discount_factor ** steps - 1) / (steps * (self.discount_factor - 1))
                self.recent_transitions.append([self.start_state, self.assigned_action, self.end_state, self.reward*rate, terminate,
                                                (self.end_tick - self.start_tick) // 60+1])
                temp_reward=self.reward-self.opt_reward
                steps = max((self.end_tick - self.opt_start_tick) // 60,1)
                rate = (self.discount_factor ** steps - 1) / (steps * (self.discount_factor - 1))

                # self.recent_transitions.append([self.opt_start_state, self.action, self.end_state, temp_reward*rate, terminate,
                #                                 (self.end_tick - self.opt_start_tick) // 60])
                # print('Line 361 Intermediate recording, current aciton is {}, Recorded action is {}, selected option is {}, option flag is {}'.format((self.action,self.state.dispatch_action_id), action, self.assigned_action,self.assigned_option))
                if self.on_random: self.f_transition.append([self.opt_start_state, self.f_end_state,1]) #following options

                # if self.state.id == 1:
                #     print('Y---APT----PARK: Park opt recorded,  action: {}'.format(self.recent_transitions[-1]))

        self.__reset_plan()
        self.__change_to_idle()



    def dump_last(self,tick): #store the transaction and retrive terminal flag
        #dump all remaining transactions to DQN
        self.end_tick = tick
        self.end_state = [self.end_tick, self.state.hex_id, self.state.SOC]
        # update the reward
        terminate=True
        self.recent_transitions.append([self.start_state, self.assigned_action, self.end_state, self.reward,
                                                   terminate, (self.end_tick - self.start_tick) // 60])
                # elif self.start_state is not None:
                #     self.recent_transitions.append([self.start_state, self.action, self.end_state, self.reward, self.terminate_flag, (self.end_tick - self.start_tick)//60])
        exported_trans = self.recent_transitions
        return exported_trans



    def head_for_charging_station(self, cs_id, cs_coord, tick, action_id):
        assert self.__behavior.available
        self.reward = 0  # - BETA_RANGE_ANXIETY * self.compute_mileage_anxiety(self.state.SOC)
        self.decay_lambda = 1.0 # reset decaying lambda, updated per minute
        self.start_tick = tick
        self.start_state = [self.start_tick, self.state.hex_id, self.state.SOC]
        self.action = action_id
        # self.__reset_plan()
        self.__set_destination(cs_coord)
        self.state.assigned_charging_station_id = cs_id
        self.__charging_station.append(cs_id)
        self.__change_to_waytocharge()

    def take_rest(self, duration):
        assert self.__behavior.available
        # self.__reset_plan()
        self.state.idle_duration = 0
        # self.__set_return_time(self.get_location(), duration)
        self.__change_to_off_duty()
        # self.__log()

    def head_for_customer(self, destination, customer_hex_location,tick):
        '''
        :destination: lon, lat
        '''
        assert self.__behavior.available
        # if self.state.status==0: #idle state, set prime start condition
        #     self.prime_start_tick = tick
        #     self.prime_start_state = [self.prime_start_tick, self.state.hex_id, self.state.SOC]

        self.__set_destination(destination)
        self.state.origin_hex=self.state.hex_id
        self.state.destination_hex=customer_hex_location

        self.dist_to_destination=sum(self.state.per_tick_dist)
        self.time_to_destination=sum(self.time_ticks)


        #return tick, from, to, travel distance, travel time, and waiting time of the passenger (and the action that leads to current trip)


        trip_flag=True
        # self.f_start_state = [tick, self.state.hex_id,
        #                       self.state.SOC]  # if not cruising ,meaning idle or stay, start a new transition for f
        if self.prime_start_state is not None:
            self.prime_transition.append([self.prime_start_state,self.action,[tick+60,self.state.hex_id,self.state.SOC],trip_flag, (tick - self.prime_start_tick)//60+1])
        else:
            print('Start state is None and I can not pop in new rewards')

        if len(self.cruise_trajectory)>0:
            #store the trajectory
            self.cruise_trajectory[-1][-1]=True #trip made
            self.trajectory_transition.append(self.cruise_trajectory)
            self.cruise_trajectory=[]
        # else:
        #     self.trajectory_transition.append([[self.f_start_state,[tick+60,self.state.hex_id,self.state.SOC], self.action,True]])

        self.state.need_route=True
        self.state.assigned_customer_id = customer_hex_location

        self.pick_start_tick=tick
        self.pick_start_state=[tick, self.state.hex_id, self.state.SOC]

        self.__customers_ids.append(customer_hex_location)
        # self.f_start_state = [tick, self.state.hex_id, self.state.SOC]
        # if self.state.status!= status_codes.V_CRUISING :
        #     if self.assigned_option==-1:
        #          #if not cruising ,meaning idle or stay, start a new transition for f
        # self.option_contd=0
        #get matched to a passenger, dump the transition

        self.__change_to_assigned()

    def pickup(self,tick):
        # assert self.get_location() == self.customer.get_origin_lonlat()
        #record for pick up trips
        self.matching_od_pairs.append([self.pick_start_tick,self.state.origin_hex,self.state.destination_hex,self.dist_to_destination,self.time_to_destination,self.customer.waiting_time,self.assigned_action])


        self.state.destination_hex = self.customer.get_destination()
        self.state.origin_hex=self.state.hex_id
        self.state.need_route=True
        self.trip_dest=self.state.destination_hex
        self.customer.ride_on()

        if (tick + 60) // 60 % 1440 == 0:
            self.end_tick=tick
        else:
            self.end_tick=tick+60

        self.f_end_state = [self.end_tick, self.state.hex_id,
                              self.state.SOC]  # if not cruising ,meaning idle or stay, start a new transition for f
        # # #
        # self.f_transition.append([self.f_start_state, self.f_end_state,0])
        # if self.state.id==1:
        #     print(self.f_start_state,self.f_end_state)

        self.__customers.append(self.customer)
        customer_id = self.customer.get_id()
        # self.__reset_plan() # For now we don't consider routes of occupied trip
        self.state.assigned_customer_id = customer_id
        self.__set_destination(self.customer.get_destination_lonlat())
        self.__change_to_occupied()
        self.customer = None  # only take the id
        # self.__log()

    def dropoff(self, tick,hex_zones):
        """
        self.end_state inherit directly from the self.park() ot self.start_waitpile(), which are the next steps of crusing and head for charging station.
         since the matching considers the parked idled or charged ones.
        :param tick:
        :return:

        """

        hex_zones[self.state.hex_id].remove_veh(self)
        hex_zones[self.trip_dest].add_veh(self)
        self.state.hex_id = self.trip_dest


        assert len(self.__customers) > 0
        customer = self.__customers.pop(0)
        customer.get_off()
        # total_trip_dist, total_drip_duration, customer's waiting time
        customer_payment = customer.make_payment(self.dropoff_distance,
                                                 self.dropoff_duration, self.pickup_duration)
        # customer_payment=5

        if (tick + 60) // 60 % 1440 == 0:
            terminate = True
            self.end_tick=tick
        else:
            terminate = False
            self.end_tick = tick + 60

        trip_flag=True
        self.f_end_state = [self.end_tick, self.state.hex_id, self.state.SOC]
        # self.f_transition.append([self.f_start_state,self.f_end_state])



        self.total_earnings += customer_payment
        #distribute the payment over different time steps
        # self.reward += self.decay_lambda*BETA_EARNING * customer_payment
        self.reward +=  customer_payment
        self.cum_reward+= customer_payment
        self.end_state = [self.end_tick, self.state.hex_id, self.state.SOC]
        trip_flag=True
        if self.state.SOC < 0:
            self.reward -= self.decay_lambda * SOC_PENALTY
            self.cum_reward-=SOC_PENALTY
        # update the reward

            #     # update the reward, add previous SOC penalty back to avoid double count on mileage anxiety.
            #     last_step_SOC = self.recent_transitions[-1][0][-1]


        if self.assigned_option>=0:

            temp_reward = self.reward - self.opt_reward
            steps = max((self.end_tick - self.opt_start_tick) // 60,1)
            rate = (self.discount_factor ** steps - 1) / (steps * (self.discount_factor - 1)) #discounted rate
            action = self.action
            #
            # self.recent_transitions.append([self.opt_start_state, action, self.end_state, temp_reward*rate, terminate,
            #                                 (self.end_tick - self.opt_start_tick) // 60])


            steps = max((self.end_tick - self.start_tick) // 60,1)
            rate = (self.discount_factor ** steps - 1) / (steps * (self.discount_factor - 1)) #discounted rate


            self.recent_transitions.append(
                [self.start_state, self.assigned_action, self.end_state, self.reward*rate, terminate,
                 (self.end_tick - self.start_tick) // 60 ])

            # if self.state.id==1:
            #     print('Y---APT----PARK: Dropoff recorded, for action: {}, and for opt: {}'.format(self.recent_transitions[-2],self.recent_transitions[-1]))

        # print('Line 492 Trip End of option recording, current action is {}, Recorded action is {}, selected option is {}, option flag is {}'.format((self.action,self.state.dispatch_action_id),
        #     self.assigned_action, self.assigned_action, self.assigned_option))

        if self.assigned_option==-1:
            temp_reward = self.reward - self.opt_reward
            steps = max((self.end_tick - self.opt_start_tick) // 60,1)
            rate = (self.discount_factor ** steps - 1) / (steps * (self.discount_factor - 1)) #discounted rate
            action=self.assigned_action

            self.recent_transitions.append([self.opt_start_state, action, self.end_state, temp_reward*rate, terminate,
                                            (self.end_tick - self.opt_start_tick) // 60 ])

            if self.state.id==1:
                print('Non---APT----PARK: Dropoff recorded, for non-apt action: {}'.format(self.recent_transitions[-1]))


            # print('Line 502 Trip Intermediate option recording, current action is {}, Recorded action is {}, selected option is {}, option flag is {}'.format((self.action,self.state.dispatch_action_id),
            #     action, self.assigned_action, self.assigned_option))

        # elif self.start_state is not None:
        #     self.recent_transitions.append([self.start_state, self.action, self.end_state, self.reward, self.terminate_flag, (self.end_tick - self.start_tick)//60])
        self.reset_dist_and_time_per_trip()
        self.assigned_option=-1
        self.option_contd=0
        self.state.current_capacity = 0
        self.__customers_ids = []
        self.__change_to_idle()
        # self.__reset_plan()
        # self.__log()

    def start_waitpile(self, tick):
        self.start_charge_SOC = self.state.SOC
        self.start_wait_tick = tick
        self.__change_to_waitpile()
        # self.__log()

    def start_charge(self):
        self.__change_to_charging()

    def end_charge(self, tick, unit_time_price):
        """
        :param tick: current tick, i*60 sec
        :param unit_time_price: unit_time charging price by charging type
        :return:
        """
        self.state.current_capacity = 0  # current passenger on vehicle
        self.__customers_ids = []
        self.state.SOC = self.state.target_SOC  # 90% charge
        self.end_tick = tick
        self.end_state = [self.end_tick, self.state.hex_id, self.state.SOC]
        cumulative_discounted_reward = self.decay_lambda * np.sum([self.discount_factor**(t+1) for t in range((self.end_tick-self.start_wait_tick)//60)])  # 60 is t_unit
        self.decay_lambda *= self.discount_factor**((self.end_tick-self.start_wait_tick)//60)  # update lambda, the last update was when it arrived at charging station
        self.reward += - BETA_TIME * 60*cumulative_discounted_reward - BETA_CHARGE_COST * unit_time_price * self.decay_lambda
        self.cum_reward+=- BETA_TIME * 60*cumulative_discounted_reward - BETA_CHARGE_COST * unit_time_price *self.decay_lambda
        terminate_flag=False
        self.recent_transitions.append([self.start_state, self.assigned_action, self.end_state, terminate_flag, self.reward, (self.end_tick - self.start_tick)//60])
        self.mileage_per_charge_cycle = 0
        self.__change_to_idle()
        self.__reset_plan()
        # self.reward += - BETA_TIME * (self.end_tick - self.start_wait_tick) - \
        #                BETA_CHARGE_COST * unit_time_price * (self.end_tick - self.start_wait_tick)

    def quick_end_charge(self):
        self.state.current_capacity = 0  # current passenger on vehicle
        self.__customers_ids = []
        self.state.SOC = self.state.target_SOC  # 90% charge
        self.end_tick = self.start_wait_tick
        self.end_state = [self.end_tick, self.state.hex_id, self.state.SOC]
        #  self.reward += - BETA_TIME * (self.end_tick - self.start_wait_tick) - self.decay_lambda* QUICK_END_CHARGE_PENALTY
        self.reward -= self.decay_lambda * QUICK_END_CHARGE_PENALTY
        self.cum_reward-=QUICK_END_CHARGE_PENALTY
        self.decay_lambda *= self.discount_factor
        terminate_flag=False
        self.recent_transitions.append([self.start_state, self.action, self.end_state, self.reward, terminate_flag, (self.end_tick - self.start_tick)//60])
        self.mileage_per_charge_cycle = 0
        self.__change_to_idle()
        self.__reset_plan()

    # def update_location(self, location):
    #     self.state.lon, self.state.lat = location

    def update_customers(self, customer):
        # customer.ride_on()
        self.__customers.append(customer)

    def update_time_to_destination(self):
        '''
        :return if arrived or not
        :update: per tick location, SOC change, record total travel distance
        todo: check if the following code is correct
        '''
        if len(self.time_ticks) > 0:
            current_coords = self.state.per_tick_coords.pop(0)
            current_dist = self.state.per_tick_dist.pop(0)

            current_duration = self.time_ticks.pop(0)

            if self.state.status!=status_codes.V_OCCUPIED:
                self.idle_dist+=current_dist*MILE_PER_METER
                self.idle_time += current_duration/60


            self.decay_lambda *= self.discount_factor
            # self.reward += self.decay_lambda*(-BETA_DIST * current_dist * MILE_PER_METER - BETA_TIME * current_duration)
            self.reward+=-BETA_DIST * current_dist * MILE_PER_METER - BETA_TIME * current_duration
            self.cum_reward+=-BETA_DIST * current_dist * MILE_PER_METER - BETA_TIME * current_duration
            # if not self.no_charge_trial:
            self.state.SOC -= (current_dist * MILE_PER_METER / self.state.mile_of_range) * SIM_ACCELERATOR * self.non_charging_mask

            self.mileage_per_charge_cycle += current_dist * MILE_PER_METER / self.state.mile_of_range
            self.state.real_time_location = current_coords
            # record pick-up and drop-off distance
            if self.state.status == status_codes.V_ASSIGNED:
                self.pickup_distance += current_dist * MILE_PER_METER
                self.pickup_duration += current_duration  # in sec
            if self.state.status == status_codes.V_OCCUPIED:
                self.dropoff_distance += current_dist * MILE_PER_METER
                self.dropoff_duration += current_duration  # in sec
            if len(self.time_ticks) <= 0:
                self.state.time_to_destination = {}
                self.state.lat = current_coords[1]  # destination lat and lon were appended with destination info.
                self.state.lon = current_coords[0]
                return True
            else:
                return False

    # some getter methods
    def get_id(self):
        vehicle_id = self.state.vehicle_id
        return vehicle_id

    def get_hex_id(self):
        return self.state.hex_id

    def get_customers_ids(self):
        return self.__customers_ids

    def get_destination(self):
        destination = self.state.destination_lon, self.state.destination_lat
        return destination

    def get_speed(self):
        speed = self.state.speed
        return speed

    def get_agent_type(self):
        return self.state.agent_type

    def get_price_rates(self):
        return [self.state.price_per_travel_m, self.state.price_per_wait_min]

    def reachedCapacity(self):
        if self.state.current_capacity == self.state.max_capacity:
            return True
        else:
            return False

    def get_assigned_customer_id(self):
        customer_id = self.state.assigned_customer_id
        return customer_id

    def get_assigned_cs_id(self):
        return self.state.assigned_charging_station_id

    def to_string(self):
        s = str(getattr(self.state, 'id')) + " Capacity: " + str(self.state.current_capacity)
        return s

    def get_status(self):
        return self.state.status

    def get_state(self):
        state = []
        for attr in self.state.__slots__:
            state.append(getattr(self.state, attr))
        return state

    def get_num_cust(self):
        return self.state.current_capacity

    def get_vehicle(self, id):
        if id == self.state.vehicle_id:
            return self

    def exit_market(self):
        return False

    def __reset_plan(self):
        self.state.reset_plan()

    def __set_destination(self, destination):
        self.state.destination_lon, self.state.destination_lat = destination

    def __change_to_idle(self):
        self.__change_behavior_model(status_codes.V_IDLE)

    def __change_to_cruising(self):
        self.__change_behavior_model(status_codes.V_CRUISING)

    def __change_to_stay(self):
        self.__change_behavior_model(status_codes.V_STAY)

    def __change_to_assigned(self):
        self.__change_behavior_model(status_codes.V_ASSIGNED)

    def __change_to_occupied(self):
        self.__change_behavior_model(status_codes.V_OCCUPIED)

    def __change_to_off_duty(self):
        self.__change_behavior_model(status_codes.V_OFF_DUTY)

    def __change_to_waytocharge(self):
        self.__change_behavior_model(status_codes.V_WAYTOCHARGE)

    def __change_to_charging(self):
        self.__change_behavior_model(status_codes.V_CHARGING)

    def __change_to_waitpile(self):
        self.__change_behavior_model(status_codes.V_WAITPILE)

    def __change_to_tobedispatched(self):
        self.__change_behavior_model(status_codes.V_TOBEDISPATCHED)

    def __change_to_tobecruised(self):
        self.__change_behavior_model(status_codes.V_TOBECRUISED)

    def __change_behavior_model(self, status):
        self.__behavior = self.behavior_models[status]
        self.state.status = status

    def compute_mileage_anxiety(self, SOC, a = 1.093, b = 6.380, c = -0.090):
        """
        we use exponential function to describe the mileage anxiety
        :param SOC: soc [0,1]
        :param a: 1.09329753
        :param b: 6.3796224
        :param c: -0.09034716
        :return: mileage anxiety in range of [0,1]
        """
        return np.max([(a * np.exp(-b * SOC) + c), 0])

    def reset_dist_and_time_per_trip(self):
        self.pickup_distance = 0
        self.dropoff_distance = 0
        self.pickup_duration = 0
        self.dropoff_duration = 0
    # def log_info(self,tick):
    #     if self.state.id in [1,100,1000] and tick < 3000*60:
    #         with open('logs/vehicle_track/vehicle_track_{}.log'.format(self.state.id), 'a') as f:
    #             f.write('tick:{},Veh_ID:{},Coord:{},SOC:{},Mileage:{},Total Payment:{},Reward:{}\n'.format(tick//60,self.state.id,self.state.real_time_location, self.state.SOC,self.mileage_per_charge_cycle*self.state.mile_of_range, self.total_earnings,self.reward))
