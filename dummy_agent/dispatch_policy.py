class DispatchPolicy(object):
    def __init__(self):
        # self.demand_predictor = DemandPredictionService()
        self.updated_at = {}

    def dispatch(self, current_time, vehicles,f):
        # self.update_state(current_time, vehicles)
        # print("Len: ", len(tbd_vehicles))
        if len(vehicles) == 0:
            return []
        commands = self.get_dispatch_decisions(vehicles,current_time)
        return commands

    def get_dispatch_decisions(self, vehicles,current_time):
        return []

    # Return best action for this vehicle given its state, and returns whether it will be Offduty or not
    # pseudo prediction (generate random dispatching actions)
    def predict_best_action(self, vehicle_id, vehicle_state,current_time):
        pass

    # Store time of dispatch for each vehicle
    def record_dispatch(self, vehicle_ids, current_time):
        for vehicle_id in vehicle_ids:
            self.updated_at[vehicle_id] = current_time

    # Creating Dispatch dictionary associated with each vehicle ID, it could be decided for that vehicle to be
    # Offduty, or be assigned a destination to head to, or have a cache key
    def create_dispatch_dict(self, vehicle_id, destination=None, offduty=False, cache_key=None,action = None):
        pass
        dispatch_dict = {}
        dispatch_dict["vehicle_id"] = vehicle_id
        dispatch_dict["action"] =action
        if offduty:
            dispatch_dict["offduty"] = True
        elif cache_key is not None:
            dispatch_dict["cache"] = cache_key
        else:
            dispatch_dict["destination"] = destination
        return dispatch_dict
    # Get the destination from dispatched vehicles
    
    # def convert_action_to_destination(self, vehicle_state, a):
    #     pass
    #     cache_key = None
    #     target = None
    #     ax, ay = a  # Action from action space matrix
    #     x, y = mesh.convert_lonlat_to_xy(vehicle_state.lon, vehicle_state.lat)
    #     lon, lat = mesh.convert_xy_to_lonlat(x + ax, y + ay)
    #     if lon == vehicle_state.lon and lat == vehicle_state.lat: # not move
    #         pass
    #     elif FLAGS.use_osrm and mesh.convert_xy_to_lonlat(x, y) == (lon, lat):
    #         cache_key = ((x, y), (ax, ay))  # Create cache key with location associated with action
    #     else:
    #         target = (lat, lon)

    #     return target, cache_key

class Dummy_DispatchPolicy(DispatchPolicy):
    def __init__(self):
        super().__init__()