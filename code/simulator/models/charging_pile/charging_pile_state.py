from novelties import status_codes
from simulator.settings import SIM_ACCELERATOR
class ChargingPileState(object):
    # state vector for charging piles
    __slots__ = ['id','flag','type','assigned_vehicle_id','idle_duration',
    'charging_rate','time_to_available', 'avg_waiting_time','incentive']

    def __init__(self,type): # id, location, Level 2/supercharge
        # self.id = id
        self.type = type
        self.charging_rate = self.set_charging_rate()
        self.idle_duration = 0
        self.time_to_available = 0
        self.avg_waiting_time = 0
        self.incentive = 0

    def reset(self): # when charging pile is on-duty
        self.time_to_available = 0
    def set_charging_rate(self):
        if self.type == status_codes.SP_DCFC:
            return SIM_ACCELERATOR* 80/20 # mile per min
        if self.type == status_codes.SP_LEVEL2:
            return SIM_ACCELERATOR* 25/60 # mile per min




