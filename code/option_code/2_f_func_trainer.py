import numpy as np
from collections import defaultdict
from dqn_option_agent.f_approx_agent import F_Agent
from dqn_option_agent.f_approx_network import F_Network
from config.setting import NUM_REACHABLE_HEX, MAP_WIDTH, MAP_HEIGHT, F_AGENT_SAVE_PATH
import pickle
import geopandas as gpd
import torch

NUM_OPTIONS = 0  # train the f_approx under one option running: the traj is collected with one option running.

def load_data():
    od_by_hour = defaultdict(list)
    with open('logs/vehicle_track/od_trajs_%d.csv'%NUM_OPTIONS, 'r') as f:
        next(f)
        for lines in f:
            line = lines.strip().split(',')
            h, o, d = line  # hour, oridin, dest, trip_time/num of trip
            od_by_hour[int(h)].append([o,d])
    return od_by_hour

def get_hex_diffusions(xy_coords):
    with open('../data/hex_diffusion.pkl', "rb") as f:
        hex_diffusions = pickle.load(f)  # with key: hex_id
    mat = np.zeros((NUM_REACHABLE_HEX, MAP_WIDTH, MAP_HEIGHT))

    for key_id, diffusions in hex_diffusions.items():
        for hex_id, diff in enumerate(diffusions):
            x, y = xy_coords[hex_id]
            mat[key_id, x, y] = diff
    return mat

def load_f_func_approx_by_hour(hex_diffusion,epi):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f_func_approx_list = defaultdict()
    for hr in range(24):
        f_approx = F_Network()
        checkpoint = torch.load(F_AGENT_SAVE_PATH + 'f_network_o%d_h%d_e%d.pkl' % (NUM_OPTIONS,hr,epi))
        f_approx.load_state_dict(checkpoint['net'], False)
        f_func_approx_list[hr] = f_approx.to(device)
        print('Successfully load saved network for hour {}!'.format(hr))

    f_dict = defaultdict()
    for hr in range(24):
        f_dict[hr] = (f_func_approx_list[hr].forward(
            torch.from_numpy(np.array(hex_diffusion)).to(dtype=torch.float32,device=device))).cpu().detach().numpy()
    with open('logs/hex_p_value_%d.csv'%NUM_OPTIONS, 'w') as p_file:
        for hr in range(24):
            for hex_id, p_value in enumerate(f_dict[hr]):
                p_file.writelines('{},{},{}\n'.format(hr, hex_id, p_value[0]))
            print('finished processing data in hour {}'.format(hr))

if __name__ == '__main__':
    # load training set, it includes hour, origin_hex_id, and destination_hex_id
    training_set = load_data()
    f_func_agents = defaultdict()
    df = gpd.read_file('../data/NYC_shapefiles/snapped_clustered_hex.shp')  # tagged_cluster_hex

    xy_coords= df[['col_id', 'row_id']].to_numpy()
    hex_diffusions = get_hex_diffusions(xy_coords)

    num_episode = 3
    # initial F networks
    #
    for hr in range(24):
        f_func_agents[hr] = F_Agent(hex_diffusions,NUM_OPTIONS)

    [f_approx.add_od_pair(training_set[hr]) for hr, f_approx in f_func_agents.items()]
    # record the training history

    with open('logs/f_func_training_hist_%d.csv' % (NUM_OPTIONS), 'w') as f:
        for episode in range(num_episode):  # we train 10 episode
            print('start training.......')
            [f_approx.train_f_function(hr,f,episode) for hr, f_approx in f_func_agents.items()]
            print('Finish episode {}'.format(episode))

    load_f_func_approx_by_hour(hex_diffusions,num_episode-1)
