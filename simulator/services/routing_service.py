from simulator.services.osrm_engine import OSRMEngine
from simulator.services.fastroute_engine import FastRoutingEngine
from config.hex_setting import FLAGS


class RoutingEngine(object):
    engine = None

    @classmethod
    def create_engine(cls):
        if cls.engine is None:
            if FLAGS.use_osrm:
                cls.engine = OSRMEngine()
            else:
                cls.engine = FastRoutingEngine()
        return cls.engine

