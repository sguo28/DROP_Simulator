from .charging_pile import charging_station
from config.hex_setting import charging_station_data_path


class ChargingRepository():
    '''
    make it infinity:
    '''
    # charging_stations ={}
    charging_repo = []
    @classmethod
    def init(cls):
        with open(charging_station_data_path,'r') as f:
            next(f)
            for lines in f:
                line = lines.strip().split(',')
                num_l2, num_dc, ilat, ilon,hex_id = line
                cls.charging_repo.append(charging_station(n_l2=int(float(num_l2)),n_dcfast=int(float(num_dc)),lat = ilat, lon=ilon,hex_id=hex_id))

    @classmethod
    def get_all(cls):
        return cls.charging_repo
    @classmethod
    def get_charging_station(cls,cid):
        return cls.charging_repo[cid]
        


'''
    # State Vector for charging station
    request_column_names = [
        'id', # unique ID
        'c_lon', # lon for grid
        'c_lat', # lat for grid
        'flag', # 0: available; 1: occupied 
        'charging_type', # level 2, DCFC
        'incentive'   # based on real-time energy use
    ]
    charging_piles = {}
    new_available_piles = []

    @classmethod
    def init(cls):
        cls.charging_piles = {}
        cls.new_available_piles = []

    @classmethod
    # Creating customer dictionary with their associated IDs
    def update_charging_piles(cls, charging_piles):
        cls.new_available_piles = charging_piles
        for charging_pile in charging_piles:
            cls.charging_piles[charging_pile.id] = charging_pile

    @classmethod
    def get(cls, charging_pile_id):
        return cls.charging_piles.get(charging_pile_id, None)

    @classmethod
    def get_all(cls):
        return list(cls.charging_piles.values())

    @classmethod
    # # Get the new requests asociated with the new customers list
    def get_new_requests(cls):
        # print("C: ", len(cls.new_customers))
        all_charging_piles = [customer.get_request() for customer in cls.charging_piles]
        # print(cls.request_column_names)
        #  Creating a DF with all the request columns
        df = pd.DataFrame.from_records(all_charging_piles, columns=cls.request_column_names)
        # print(df.columns.values)
        return df

    @classmethod
    #  Delete a specfic customer ID
    def delete(cls, charging_pile_id):
        cls.charging_piles.pop(charging_pile_id)
'''