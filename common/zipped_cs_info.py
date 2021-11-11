from collections import defaultdict
import csv

def get_cs_info():
    columns = defaultdict(list)
    with open('../data/processed_cs.csv','r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for rid,r in row.items():
                columns[rid].append(r)

    list_l2 = [int(float(i)) for i in columns['EV_Level2']]
    list_dc = [int(float(i)) for i in columns['EV_DC_Fast']]
    return list_l2, list_dc,columns['Latitude'], columns['Longitude']

