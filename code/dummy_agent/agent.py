from simulator.settings import FLAGS, INITIAL_MEMORY_SIZE
from simulator.models.vehicle.vehicle_repository import VehicleRepository


class Dummy_Agent(object):

    def __init__(self, dispatch_policy):
        self.dispatch_policy = dispatch_policy

    def get_dispatch_commands(self, current_time, vehicles):

        # dispatch_commands = self.dispatch_policy.dispatch(current_time, vehicles)
        # return dispatch_commands
        return []

class DQN_Agent(Dummy_Agent):

    def get_dispatch_commands(self, current_time, vehicles):
        dispatch_commands = self.dispatch_policy.dispatch(current_time, vehicles)
        return dispatch_commands



