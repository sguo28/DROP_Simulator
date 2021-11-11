from numpy.core.fromnumeric import mean
from novelties import status_codes
from collections import deque
from config.hex_setting import CHARGE_ACCELERATOR, QUEUE_LENGTH_THRESHOLD, PER_TICK_DISCOUNT_FACTOR


class chargingpile:
    def __init__(self, type, location, hex_id, hex):
        """

        :param type: Level 2 or DC-fast
        :param location: lon lat
        :param hex_id: hex_id
        :param hex: the object hex

        metrics:
            rate: charging speed
            unit_time price by type: $1.5/hr or 0.30 per min, Source: EVgo in NYC https://account.evgo.com/signUp
        """
        self.occupied = False
        self.time_to_finish = 0
        self.assigned_vehicle = None  # vehicle agent
        self.type = type
        self.location = location
        self.hex_id = hex_id
        self.hex = hex
        self.served_num = 0
        if self.type == status_codes.SP_DCFC:
            self.rate = CHARGE_ACCELERATOR * 240 / (60 * 60)  # mile per sec
            self.unit_time_price = 18 / (60*60)  # 18 USD per hour
        else:
            self.rate = CHARGE_ACCELERATOR * 25 / (60 * 60)  # mile per sec
            self.unit_time_price = 1.5 / (60 * 60)  # 1.5 USD per hour

    def reset(self):
        self.occupied = False
        self.time_to_finish = 0
        self.assigned_vehicle = None  # vehicle agent
        self.served_num = 0

    def assign_vehicle(self, veh):
        self.occupied = True
        self.time_to_finish = (veh.get_target_SOC() - veh.get_SOC()) * veh.get_mile_of_range() / self.rate
        self.assigned_vehicle = veh
        self.assigned_vehicle.start_charge()

    def get_cp_location(self):
        return self.location

    def get_cp_hex_id(self):
        return self.hex_id

    def step(self, time_step, tick):
        """
        :param time_step: 60 seconds per minute
        :param tick: current tick
        :return:
        """

        if self.time_to_finish > 0:
            self.time_to_finish -= time_step
            if self.time_to_finish <= 0:
                # charging has been completed!
                self.occupied = False
                self.time_to_finish = 0
                self.assigned_vehicle.end_charge(tick, self.unit_time_price) # cost per unit time by charging type
                self.hex.add_veh(self.assigned_vehicle)  # add the vehicle back
                # print("####VEH {} END CHARGE#### AT LOC {}".format(self.assigned_vehicle.state.vehicle_id ,self.get_cp_hex_id()))
                self.assigned_vehicle = None
                self.served_num += 1


class charging_station:
    def __init__(self, n_l2=1, n_dcfast=1, lat=None, lon=None, hex_id=None, hex=None, row_col_coord=None):
        self.location = float(lon), float(lat)
        # initial the charging piles for the charging station
        self.piles = [chargingpile(type=status_codes.SP_LEVEL2, location=self.location, hex_id=hex_id, hex=hex) for _ in
                      range(n_l2)] + \
                     [chargingpile(type=status_codes.SP_DCFC, location=self.location, hex_id=hex_id, hex=hex) for _ in
                      range(n_dcfast)]
        # self.available_piles=self.piles
        self.waiting_time = []
        self.charging_time = []
        self.queue = deque()  # waiting queue for vehicle
        self.virtual_queue = []
        self.time_to_cs = []
        self.hex_id = hex_id
        self.hex = hex
        self.row_id, self.col_id = row_col_coord
        self.num_l2_pile = n_l2
        self.num_dcfc_pile = n_dcfast

    def reset(self):
        self.waiting_time = []
        self.charging_time = []
        self.queue = deque()  # waiting queue for vehicle
        self.virtual_queue = []
        self.time_to_cs = []
        [p.reset() for p in self.piles] #reset the status of all charging piles

    def get_cs_location(self):
        return self.location

    def get_cs_hex_id(self):
        return self.hex_id

    def get_available(self):
        # set the list of available charging piles
        self.available_piles = [p for p in self.piles if p.occupied == False]

    def step(self, time_step, tick):
        '''
        First update each pile, then find available charging piles, then match, then update queue
        :return:
        '''
        # update the status
        [p.step(time_step, tick) for p in self.piles]  # update the status of each charging pile
        self.get_available()  # update available piles

        # update waiting time of each vehicle in the queue
        # assign waiting vehicles to each pile
        # must have both vehicle and pile available to proceed

        while len(self.queue) > 0 and len(self.available_piles) > 0:
            veh = self.queue.popleft()
            pile = self.available_piles.pop()
            pile.assign_vehicle(veh)

            self.waiting_time.append(veh.charging_wait)  # total waiting time of the vehicle
            self.charging_time.append(pile.time_to_finish)  # total charging time

        # for unmatched vehicles, update the waiting time of vehicles in the charging queue
        for v in self.queue:
            v.charging_wait += 1

    def add_arrival_veh(self, veh):
        '''
        :param veh: vehicle object (class)
        '''
        if len(self.queue)/(self.num_dcfc_pile+self.num_l2_pile) <= QUEUE_LENGTH_THRESHOLD: # we seperate dcfc and l2, so either of the number must be 0.
            self.queue.append(veh)
            veh.charging_wait = 0  # no wait at the beginning
            # print(veh.state.origin_hex,veh.state.current_hex, veh.state.destination_hex, veh.state.per_tick_coords, self.hex_id,self.location)
            self.hex.remove_veh(veh)  # remove this vehicle from the hex zone
        else:
            veh.quick_end_charge()  # quick pop out the vehicle if the queue is long.

    def get_queue_length(self):
        return len(self.queue)

    def get_average_waiting_time(self):
        if len(self.waiting_time) < 1:
            return 0.0
    def get_served_num(self):
        return sum([cp.served_num for cp in self.piles])

#### in your funciton, you can define your charging repository as a list of length N, N = total number of stations
