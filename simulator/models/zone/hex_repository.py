from .hex_zone import hex_zone
from common import zipped_hex

class hex_repository():

    @classmethod
    def init(cls):
        cls.hex_zones = [hex_zone(hex_id,coord,neighbors, charging_stations, arrival_rate) \
            for hex_id,coord,neighbors, charging_stations, arrival_rate in zipped_hex.get_hex_info()]

    @classmethod
    def get_all(cls):
        return cls.hex_zones

    @classmethod
    def get_hex(cls,hex_ids):
        hex_id_list, _,_,_,_ =zipped_hex.get_hex_info()
        is_selected = [hex_id_list[i] in hex_ids for i in range(len(hex_id_list))]
        cls.hex_zones[is_selected]