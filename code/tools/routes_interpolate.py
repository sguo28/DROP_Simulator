#interpolate the pkl file containing routes between all od pairs.
import numpy as np
import pickle
def routes_interp(hex_id,destination_hex,routes,t_unit=60):
    route = routes[(hex_id, destination_hex)]['route']
    time_to_destination = routes[(hex_id, destination_hex)]['travel_time']
    travel_dist = routes[(hex_id, destination_hex)]['distance']
    if time_to_destination[0] > 0:  # valid trajectories
        total_tt = sum(time_to_destination)
        cum_time = np.cumsum(time_to_destination)
        cum_dist = np.cumsum(travel_dist)
        time_ticks = [i * t_unit for i in
                      range(1, int(total_tt // t_unit + 1))]  # the time steps to query from per simulation tick
        step_time_ticks = [t_unit for _ in range(len(time_ticks))]
        if total_tt % t_unit > 0:
            time_ticks.append(total_tt)  # add the final step
            step_time_ticks.append(total_tt % t_unit)

        per_tick_dist = np.interp(time_ticks, cum_time, cum_dist)
        # if len(per_tick_dist)>=0:
        try:
            per_tick_dist = [per_tick_dist[0]] + np.diff(per_tick_dist).tolist()
        except IndexError:
            print('tick dist:', travel_dist)
            print('tick time:', time_to_destination)

        lons = [route[0][0][0]] + [coord[1][0] for coord in route]
        lats = [route[0][0][1]] + [coord[1][1] for coord in route]

        cum_time = cum_time.tolist()
        per_tick_lon = np.interp(time_ticks, [0] + cum_time, lons)
        per_tick_lat = np.interp(time_ticks, [0] + cum_time, lats)
        per_tick_lon = per_tick_lon.tolist()
        per_tick_lat = per_tick_lat.tolist()

        per_tick_coords = [[round(lon,5), round(lat,5)] for lon, lat in zip(per_tick_lon, per_tick_lat)]
        per_tick_time=step_time_ticks
        per_tick_dist=[round(i) for i in per_tick_dist]
    else:
        per_tick_coords=route
        per_tick_dist=travel_dist
        per_tick_time=time_to_destination

    return per_tick_coords,per_tick_dist,per_tick_time

new_routes=dict()
with open('../../data/all_routes.pkl', 'rb') as f:
    # self.hex_routes = pickle.load(f)
    hex_routes=pickle.load(f)
    id=0
    for key in hex_routes.keys():
        oid,did=key
        route,dist,time=routes_interp(oid,did,hex_routes)
        new_routes[key]=dict()
        new_routes[key]['route']=route
        new_routes[key]['travel_time']=time
        new_routes[key]['distance']=dist
        id+=1
        if len(time)==0:
            print(id,time,key,route,hex_routes[key]['route'],hex_routes[key]['travel_time'])
    with open('../../data/parsed_routes.pkl','wb') as f:
        pickle.dump(new_routes,f)


# with open('../../data/parsed_routes.pkl','rb') as f:
#     hex_routes = pickle.load(f)
#     print('file loaded!')