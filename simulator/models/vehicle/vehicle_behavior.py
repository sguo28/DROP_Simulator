class VehicleBehavior(object):
    available = True

    def step(self, vehicle, tick, hex_zones):
        pass


class Waytocharge(VehicleBehavior):
    '''
    todo: vehicle.state.status == V_WAITPILE; charging_station_repo[vehicle.get_assigned_cs()].add_arrival_veh(vehicle)
    '''
    available = False
    # Updated remaining time to destination
    def step(self, vehicle, tick, hex_zones):
        arrived = vehicle.update_time_to_destination()
        if arrived: # arrive at charing station
            vehicle.start_waitpile(tick)
            
class Charging(VehicleBehavior):
    available = False

class Waitpile(VehicleBehavior):
    available = False

class Idle(VehicleBehavior):

    pass

class Tobedispatched(VehicleBehavior):

    pass

class Tobecruised(VehicleBehavior):
    pass

class Cruising(VehicleBehavior):
    # Updated remaining time to destination, if arrived states changes to parking
    def step(self, vehicle, tick, hex_zones):
        arrived = vehicle.update_time_to_destination()
        if arrived:
            vehicle.park(tick, hex_zones) # arrived and be idle.
            return
class Stay(VehicleBehavior):
    def step(self, vehicle, tick, hex_zones):
        arrived = vehicle.update_time_to_destination()
        if arrived:
            vehicle.park(tick, hex_zones) # arrived and be idle.
            return

    # def drive(self, vehicle, timestep):
    #     route = vehicle.get_route()      # Sequence of (lon, lat)
    #     speed = vehicle.get_speed()
    #     dist_left = timestep * speed    # Remaining Distance
    #     rlats, rlons = zip(*([vehicle.get_location()] + route)) # New vehicle location after driving this route
    #     step_dist = geoutils.great_circle_distance(rlats[:-1], rlons[:-1], rlats[1:], rlons[1:])    # Get distcnace in meters
    #     for i, d in enumerate(step_dist): # update location per time step
    #         if dist_left < d:
    #             bearing = geoutils.bearing(rlats[i], rlons[i], rlats[i + 1], rlons[i + 1])      # Calculate angle of motion
    #             next_location = geoutils.end_location(rlats[i], rlons[i], dist_left, bearing)   # Calculate nxt location
    #             vehicle.update_location(next_location, route[i + 1:])           # Updating location based on route's nxt (lon, lat)
    #             return
    #         dist_left -= d
    #
    #     if len(route) > 0:
    #         vehicle.update_location(route[-1], [])  # Go the last step


class Occupied(VehicleBehavior):
    available = False
    # Updated remaining time to destination, if arrived customer gets off
    def step(self, vehicle, tick, hex_zones):
        arrived = vehicle.update_time_to_destination()
        if arrived:
            vehicle.dropoff(tick,hex_zones)

class Assigned(VehicleBehavior):
    available = False
    # Updated remaining time to destination, if arrived, update customer ID and picks him up
    def step(self, vehicle, tick, hex_zones):
        arrived = vehicle.update_time_to_destination()
        if arrived:
            vehicle.state.need_route = True
            vehicle.pickup(tick)

class OffDuty(VehicleBehavior):
    available = False
    # Updated remaining time to destination, if returned state changes to parking
    def step(self, vehicle, tick, hex_zones):
        returned = vehicle.update_time_to_destination()
        if returned:
            vehicle.park()
