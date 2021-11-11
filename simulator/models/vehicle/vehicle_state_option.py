from novelties import status_codes
from novelties import vehicle_types
from config.setting import TARGET_SOC, TARGET_SOC_STD
import numpy as np


class VehicleState(object):
    # State Vector for the vehicle
    __slots__ = [
        'id', 'lat', 'lon', 'speed', 'status', 'destination_lat', 'destination_lon', 'type',
        'travel_dist', 'price_per_travel_m', 'price_per_wait_min', 'assigned_mid',  'charge_flag', # full_charge_price',
        'assigned_customer_id', 'assigned_charging_station_id', 'time_to_destination', 'idle_duration',
        'queueing_duration', 'current_capacity', 'max_capacity', 'driver_base_per_trip', 'mileage',
        'mile_of_range', 'target_SOC', 'SOC', 'agent_type', 'charging_threshold', 'hex_id', 'current_hex',
        'vehicle_id', 'dispatch_action_id', 'need_route','route', 'real_time_location', 'need_interpolate',
        'per_tick_dist', 'per_tick_coords','total_travel_distance','duration_by_status','require_dump_transition',
        'destination_hex','origin_hex', 'converted_action_id']

    def __init__(self, id, location, hex_id, agent_type):
        """
        todo: delete gas-related metrics
        """
        self.id = id
        self.vehicle_id = id
        self.lon, self.lat = location
        self.hex_id = hex_id
        self.assigned_mid = None
        self.charge_flag = 0
        self.current_hex = hex_id  # when initialized, the current location is the same as hex_id
        self.agent_type = agent_type
        self.speed = 0
        self.status = status_codes.V_IDLE
        self.destination_lon, self.destination_lat = None, None
        self.assigned_customer_id = None
        self.assigned_charging_station_id = None
        self.time_to_destination = 0
        self.idle_duration = 0
        # self.total_idle = 0
        self.queueing_duration = 0
        self.current_capacity = 0
        self.total_travel_distance = 0
        self.driver_base_per_trip = 0
        self.price_per_travel_m = 0
        self.price_per_wait_min = 0
        self.type = self.selectVehicleType()
        self.max_capacity = self.setCapacity()
        # self.mileage = self.set_mileage()
        self.mile_of_range = self.set_range()
        self.target_SOC = round(min(1, max(0, np.random.normal(0.90, 0.02))),3)
        self.set_price_rates()
        self.SOC = round(min(1, max(0, np.random.normal(TARGET_SOC, TARGET_SOC_STD))),4)  # SOC~N(90%,2%)
        self.charging_threshold = 0.2
        self.dispatch_action_id = 0
        self.converted_action_id = 0
        self.need_route = False
        self.route = {}
        self.need_interpolate = False  ## this is the flag to say we need to interpolate the distance and coordinate of each vehicle
        self.per_tick_dist = []
        self.per_tick_coords = []
        self.real_time_location = location # first use initialized location
        self.destination_hex = None
        self.origin_hex=None
        self.duration_by_status = [[] for _ in range(7)] # 7 status codes in total
        self.require_dump_transition = False  # this is a flag to show if the vehicle has a full cycle

    def selectVehicleType(self):
        r = 1  # randrange(4)
        if r == 0:
            return vehicle_types.hatch_back
        elif r == 1:
            return vehicle_types.sedan
        elif r == 2:
            return vehicle_types.luxury
        elif r == 3:
            return vehicle_types.SUV
        return r

    def setCapacity(self):
        if self.type == vehicle_types.hatch_back:
            return 3
        if self.type == vehicle_types.sedan:
            return 4
        if self.type == vehicle_types.luxury:
            return 4
        if self.type == vehicle_types.SUV:
            return 5

    # def set_mileage(self): # mile/gallon
    #     if self.type == vehicle_types.sedan:
    #         self.mileage = float(30)
    #         return self.mileage
    #     if self.type == vehicle_types.hatch_back:
    #         self.mileage = float(35)
    #         return self.mileage
    #     if self.type == vehicle_types.luxury:
    #         self.mileage = float(25)
    #         return self.mileage
    #     if self.type == vehicle_types.SUV:
    #         self.mileage = float(15)
    #         return self.mileage

    def set_range(self):  # miles
        if self.type == vehicle_types.sedan:  # model 3
            self.mileage = float(220)
            return self.mileage
        if self.type == vehicle_types.hatch_back:  # model 2 (Nov 2021)
            self.mileage = float(254)
            return self.mileage
        if self.type == vehicle_types.luxury:  # model y
            self.mileage = float(326)
            return self.mileage
        if self.type == vehicle_types.SUV:  # model x
            self.mileage = float(340)
            return self.mileage

    def set_price_rates(self):
        if self.type == vehicle_types.sedan:
            self.price_per_travel_m = float(375) / 1000.0  # Per meter
            self.price_per_wait_min = float(0.05 / 3600)  # Per hour
            self.driver_base_per_trip = float(1000)

    # When trip is over or cancelled
    def reset_plan(self):
        self.destination_lon, self.destination_lat = None, None
        self.speed = 0
        self.assigned_customer_id = None
        self.assigned_charging_station_id = None
        # self.time_to_destination = 0

    def to_msg(self):
        state = [str(getattr(self, name)).format(":.2f") for name in self.__slots__]
        return ','.join(state)
