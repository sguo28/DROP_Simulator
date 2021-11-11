import numpy as np
from novelties import status_codes
from collections import defaultdict
from config.hex_setting import REJECT_TIME, SEC_PER_MIN
import ray


@ray.remote
class matching_zone(object):
    def __init__(self, m_id, hex_zones, time_od):
        """
        m_id: matching zone id
        hex_zones: the list of hex zone objects
        terminal_states: the dictionary for terminating states
        """
        self.matching_zone_id = m_id
        self.hex_zones = hex_zones
        self.reject_wait_time = REJECT_TIME * SEC_PER_MIN  # sec
        self.local_hex_collection = {hex.hex_id: hex for hex in hex_zones}  # create a a local hex
        self.num_matches = 0
        self.time_od = time_od

    def get_local_collection(self):
        return self.local_hex_collection

    def get_info(self):
        print('Match zone id: {}, number of hexs:{}'.format(self.matching_zone_id, len(self.hex_zones)))

    def dispatch(self, tick):
        '''
        Call dispatch for each hex zones
        :param tick:
        :return:
        '''
        [h.vehicle_dispatch(tick) for h in self.hex_zones]

    def update_passengers(self):
        '''
        Call update passenger in each hex zones
        :return:
        '''
        [h.update_passengers() for h in self.hex_zones]

    def update_vehicles(self,timestep,timetick):
        '''
        call step function for each vehicles
        :return:
        '''
        for hex in self.hex_zones:
            for veh in hex.vehicles.values():
                veh.step(timestep,timetick)

    def async_demand_gen(self, tick):
        # do the demand generation for all hex zones in the matching zone
        [h.demand_generation(tick) for h in self.hex_zones]
        return True

    def get_vehicles_by_hex(self):
        '''
        return: list of vehicle_dict per hex
        '''
        veh_dict = [hex.vehicles for hex in self.hex_zones]
        return veh_dict

    def get_vehicles_by_hex_list(self):
        """
        :return:
        """
        veh_dict = [hex.vehicles.values() for hex in self.hex_zones]
        return veh_dict

    def set_vehicles_by_hex(self, new_veh,timestep,timetick):
        # reset the new collection of vehicles for each hex areas in the matching zone
        # update the vehicle at the same time as push
        # make sure the order in new_veh is the same as the hex zone orders in each matching zone
        for i in range(len(new_veh)):
            self.hex_zones[i].vehicles = new_veh[i]
            for veh in self.hex_zones[i].vehicles.values():
                veh.step(timestep,timetick)
        # print('total vehicles deployed to zone {} is {}'.format(self.matching_zone_id,nvs))

    def get_arrivals_length(self):
        return len(self.hex_zones[0].arrivals)

    def get_all_veh(self):
        '''
        :return: all vehicles in the hex areas inside the matching zone
        '''
        all_vehs = defaultdict()
        for hex_zone in self.hex_zones:
            all_vehs.update(hex_zone.vehicles)

        return all_vehs

    def get_all_passenger(self):
        '''
        :return: all available passengers in the list
        todo: consider sorting the passengers based on their time of arrival?
        '''
        available_pass = defaultdict()
        for hex_zone in self.hex_zones:
            local_availables = {key: value for (key, value) in hex_zone.passengers.items() if value.matched == False}
            available_pass.update(local_availables)
        return available_pass

    def get_served_num(self):
        return sum([h.served_num for h in self.hex_zones])

    def get_veh_waiting_time(self):
        '''
        todo: this function makes no sense [# this is the waiting time for a charging pile]
        :return:
        '''
        return sum([h.veh_waiting_time for h in self.hex_zones])

    def match(self):
        '''
        Perform the matching here.
        :return:
        '''
        # get all vehicles and passengers first
        all_pass = self.get_all_passenger()
        all_veh = self.get_all_veh()
        self.matching_algorithms(all_pass, all_veh)

    def matching_algorithms(self, passengers, vehicles):
        '''
        todo: complete the matching algorithm here
        todo: change the following 0.1 to some threshold
        passengers: the set of available Customer objects
        vehicles: the set of vehicle objects
        match available vehicles with passengers
        Change the status for passengers and vehicles
        :return: 
        no return here. We will change the mark for each passengers and drivers as they are matched
        '''
        # get only available vehicles
        # print(
        #     'Current matching zone={}, Total matched passengers={}, Number of passengers={}, Number of drivers={}'.format(
        #         self.matching_zone_id, self.num_matches, len(passengers.keys()), len(vehicles.keys())))
        if len(passengers.keys()) > 0 and len(vehicles.keys()) > 0:
            vehicles = {key: value for key, value in vehicles.items() if value.state.status == status_codes.V_IDLE and value.state.SOC>0.1}
            if len(vehicles.keys()) > 0:
                self.num_matches += self.match_requests(vehicles, passengers)

    ##### match requests to vehicles ######
    def match_requests(self, vehicles, passengers):
        """
        :param vehicles:
        :param passengers:
        :return:
        """
        v_hex_id = [veh.state.hex_id for veh in vehicles.values()]
        vehs = [veh for veh in vehicles.values()]

        r_hex_id = [customer.request.origin_id for customer in passengers.values()]
        requests = [customer for customer in passengers.values()]

        od_mat = self.time_od[v_hex_id,:][:,r_hex_id]
        assignments = self.assign_nearest_vehicle(vehs, od_mat)

        for [v_id, r_id] in assignments:
            # if vehicle is None or customer is None:
            #     continue
            vehicle = vehs[v_id]
            customer = requests[r_id]
            customer.matched = True
            vehicle.customer = customer  # add matched passenger to the current on
            vehicle.state.need_route = True
            vehicle.state.current_hex = customer.get_origin()
            vehicle.head_for_customer(customer.get_origin_lonlat(),customer.get_origin())

        return len(assignments)  # record nums of getting matched


    # Returns list of assignments
    def assign_nearest_vehicle(self, vehicles, od_mat):
        """
        :param:vehicles: vehicles in idled status
        :param: od_tensor: a V by R time matrix (vehicle and request)
        :return: matched od pairs: vid, rid (row and col numbers of od_mat)
        """
        assignments = []
        time = od_mat
        for vid, veh in enumerate(vehicles):
            if vid > time.shape[1]:
                break
            rid = time[vid].argmin()
            # if tt > self.reject_wait_time:
            #     continue
            time[:,rid] = 1e9
            assignments.append([vid, rid])  # o-d's hex_id, trip time, and distance
        return assignments

    def get_metrics(self):
        num_arrivals = sum([h.get_num_arrivals() for h in self.hex_zones])
        num_removed_pass = sum([h.get_num_removed_pass() for h in self.hex_zones])
        return [self.num_matches, num_arrivals, num_removed_pass]
