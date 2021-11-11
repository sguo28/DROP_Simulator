class request:
    '''
    a wrapper to make it easier to read information for each generated passenger request
    '''
    __slots__ = ['id','origin_id','origin_lat','origin_lon','destination_id','destination_lat','destination_lon', 'request_time','trip_time','matched','picked_up', 'initial_price']
    def __init__(self,id, oid,ocoord, did,dcoord,triptime, tick):
        '''
        :param args: a dictionary of attributes for the request class , having the following information
        todo: see if additional attributes are required in the request class
        '''
        # Note hex_id + id gives a unique identifier for each passenger
        self.id=id
        self.origin_id=oid
        self.destination_id=did
        self.origin_lon, self.origin_lat=ocoord
        self.destination_lon, self.destination_lat=dcoord
        self.request_time=tick
        self.trip_time = triptime
        self.matched=False
        self.picked_up=False
        self.initial_price = 0
        # self.waiting_time = 0
    
