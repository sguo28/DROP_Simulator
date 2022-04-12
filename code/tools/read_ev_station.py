import pandas as pd
import sys

sys.path.insert(0, '~/RS_Pricing')
def get_processed_charging_piles():
    pass
    chargingpiles = pd.read_json("./data/processd_cp.json")
    return chargingpiles