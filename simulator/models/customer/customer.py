from novelties import status_codes, customer_preferences, vehicle_types
from config.hex_setting import SPEED, WAIT_COST,MILE_PER_METER, TOTAL_COST_PER_MILE, DRIVER_TIME_VALUE, \
    SERVICE_PRICE_PER_MILE, SERVICE_PRICE_PER_MIN, MAX_WAIT_TIME
from random import randrange


class Customer(object):
    def __init__(self, request):
        self.request = request
        self.status = status_codes.C_CALLING
        self.waiting_time = 0
        self.car_preference = 0
        self.RS_preference = 0
        self.time_preference = 0
        self.max_tolerate_delay = 0
        self.price_threshold = 0
        self.go_to_nxt_timestep = 0
        self.accepted_price = 0
        self.matched=False
        self.set_preferences()

    def set_preferences(self):
        '''
        :return: passenger types are randomly assigned
        todo: This part is very slow. Consider modifying this to speed up the simulation.
        '''
        i = randrange(2)
        i=0
        if i == 0:
            self.car_preference = customer_preferences.C_any_car
            self.price_threshold = 20
        else:
            self.car_preference = customer_preferences.C_luxury_car
            self.price_threshold = 30
        i = randrange(2)
        if i == 0:
            self.time_preference = customer_preferences.C_not_inHurry
            self.max_tolerate_delay = float(MAX_WAIT_TIME)
        else:
            self.time_preference = customer_preferences.C_inHurry
            self.max_tolerate_delay = float(MAX_WAIT_TIME)
        i = randrange(2)
        if i == 0:
            self.RS_preference = customer_preferences.C_not_rideShare
        else:
            self.car_preference = customer_preferences.C_rideShare

    def step(self, timestep):
        if self.status == status_codes.C_CALLING and not self.go_to_nxt_timestep:
            self.disappear()

    def to_string(self):
        s = str(self.request.id) + " " + str(self.status) + " " + str(self.waiting_time)
        return s

    def print_customer(self):
        print("\n Customer Info")
        # print(self.state)
        print(self.request.id)
        print(self.request.origin_id)
        print(self.request.destination_id, self.request.trip_time)
        print(self.status, self.waiting_time)

    def get_id(self):
        return self.request.id

    def get_origin(self):
        # The initial location for the request
        return self.request.origin_id

    def get_origin_lonlat(self):
        return self.request.origin_lon, self.request.origin_lat

    def get_destination_lonlat(self):
        return self.request.destination_lon, self.request.destination_lat

    def get_destination(self):
        return self.request.destination_id

    def get_trip_duration(self):
        return self.request.trip_time

    def get_request(self):
        return self.request

    # Customer utility function goes here
    def accpet_reject_ride(self, initial_price, assigned_vehicle_status, time_till_pickup):
        accept_response = 0
        capacity = assigned_vehicle_status.current_capacity
        threshold = 0
        # if self.RS_preference == customer_preferences.C_not_rideShare and capacity > 0:
        #     return 0
        #
        # if time_till_pickup > self.max_tolerate_delay and self.time_preference == customer_preferences.C_inHurry:
        #     return 0

        # response = randrange(2)
        # utility = 1/initial_price
        # if utility >= 1/self.price_threshold:
        #     accept_response = 1
        if time_till_pickup <= 0:
            time_till_pickup = 3600.0
        utility = (15.0/(capacity+1)) + (4.0*assigned_vehicle_status.type) + (3600.0/time_till_pickup)
        # print("P: ", initial_price, "P - 10: ", initial_price-10)
        if assigned_vehicle_status.type == vehicle_types.hatch_back:
            accept_response = 1
            return accept_response
        elif assigned_vehicle_status.type == vehicle_types.sedan:
            threshold = float(9)
        elif assigned_vehicle_status.type == vehicle_types.SUV:
            threshold = float(13)
        else:
            threshold = float(17)
        if utility > (initial_price - threshold):
            accept_response = 1
            self.accepted_price = initial_price
        else:
            self.accepted_price = 0
        return accept_response

    def wait_for_vehicle(self, waiting_time):
        self.waiting_time = waiting_time
        self.status = status_codes.C_WAITING

    def ride_on(self):
        self.status = status_codes.C_IN_VEHICLE
        # self.__log()

    def get_off(self):
        self.status = status_codes.C_ARRIVED
        # self.__log()

    def disappear(self):
        self.status = status_codes.C_DISAPPEARED
        # self.__log()

    def is_arrived(self):
        return self.status == status_codes.C_ARRIVED

    def is_disappeared(self):
        return self.status == status_codes.C_DISAPPEARED

    def make_payment(self, total_trip_distance,total_trip_duration, customer_wait_time):
        '''
        :param: total_trip_distance = pick-up dist + drop-off dist, unit: meter
        customer wait time: unit sec.
        TRIP_TIME: seconds
        SPEED: 5 m/s
        PRICE_PER_TRAVEL_M: USD/m
        FULL_CHARGE_PRICE: USD
        MILE_OF_RANGE: 220 mile
        WAIT_COST: 0.05 USD/min
        TOTAL_COST_PER_MILE = operation and maintenance
        '''
        # initial_price = (full_service_duration * SPEED * MILE_PER_METER * TOTAL_COST_PER_MILE) - (
        #             WAIT_COST * customer_wait_time) + DRIVER_TIME_VALUE * full_service_duration
        # if customer_wait_time<300: #waiting time less than 5 minutes
        #     waiting_penalty=customer_wait_time*WAIT_COST
        # elif customer_wait_time>=300 and customer_wait_time<600:
        #     waiting_penalty=300*WAIT_COST+(customer_wait_time-300)*2*WAIT_COST #additional penalty for extra waiting tiem
        # else:
        #     waiting_penalty = 900 * WAIT_COST+(customer_wait_time - 600) * 4 * WAIT_COST  # additional penalty for extra waiting tiem

        price = (total_trip_distance *  SERVICE_PRICE_PER_MILE) + (total_trip_duration * SERVICE_PRICE_PER_MIN)
        price = price - 0.5*(customer_wait_time+self.waiting_time)/60

        # print(
        #     'My waiting time {}, my trip duration {} and length {}, my waiting for pickup time {}, my payment={}'.format(
        #         self.waiting_time, total_trip_duration, total_trip_distance, customer_wait_time,price))
        # price=max(price,1)
        return price

        # SERVICE_REWARD = RIDE_REWARD*num_pass + TRIP_REWARD * trip_time - WAIT_COST * wait_time
        # RIDE_REWARD = 10.0 #5,6,7,8,9,10
        # TRIP_REWARD = 1.0
        # WAIT_COST = 0.05
        # HOP_REWARD = 3.5
        # MIN_TRIPTIME = 1.0 # in meters
        # ASSIGNMENT_SPEED = 50 # km/h (grand circle distance)

        # if not FLAGS.enable_pooling:
        #     return self.accepted_price
        # # print("C: ", self.accepted_price, base)
        # else:
        #     if self.accepted_price/cap < base:
        #         return base
        #     else:
        #         return self.accepted_price/cap

    def calculate_price(self, dist, wait_time, mile_of_range, price_per_travel_m, price_per_wait_min, full_charge_price,
                        driver_base):
        # price = trip_time*self.price_per_travel_min + wait_time*self.price_per_wait_min + dist*mileage
        # print(dist, price_per_travel_m, mileage, wait_time, price_per_wait_min)
        # We can use the fare associated with each request as the base fare
        # print("Dist: ", dist, "Dist Cost: ", (dist*price_per_travel_m), "Gas Price: ", (dist*(gas_price/(mileage*1000.0))), "Base: ", price_per_travel_m, "Wait time: ", wait_time, "Wait Cost: ",(price_per_wait_min/wait_time))

        if ((dist * price_per_travel_m) + (dist * (full_charge_price / (mile_of_range))) < (
                price_per_wait_min * wait_time)):
            print("ERR", )
            print("Dist: ", dist, "Dist Cost: ", (dist * price_per_travel_m), "Gas Price: ",
                  (dist * (full_charge_price / (mile_of_range * 1000.0))), "Base: ", price_per_travel_m, "Wait time: ",
                  wait_time, "Wait Cost: ", (price_per_wait_min * wait_time))

        if wait_time <= 0:
            wait_time = 3600
        price = (dist * price_per_travel_m) + (dist * (full_charge_price / (mile_of_range))) - (
                    price_per_wait_min * wait_time)
        # print("Final Price: ", round(driver_base+price, 2)/100.0)

        return (driver_base + price) / 100.0

    def __log(self):
        pass
        msg = ','.join(map(str, [self.request.id, self.status, self.waiting_time]))
        # sim_logger.log_customer_event(msg)
