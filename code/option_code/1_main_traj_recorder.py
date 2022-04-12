import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import time
from common.time_utils import get_local_datetime
from config.setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS, START_TIME
from simulator.simulator_cnn import Simulator
from dqn_option_agent.f_approx_agent import F_Agent
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent_fh import DeepQNetworkAgent
from dqn_option_agent.h_agent import H_Agent
# ---------------MAIN FILE---------------

NUM_OPTION = 0

if __name__ == '__main__':
    """
    This is a trajectory recorder for f_func approximator training.
    The trajectory is recorded based on the random dispatching policy of higher level DQN and fully-H-oriented options.
    
    The trajectories include:
    (1) with option: hex_id of initial and terminal state and 
    (2) without option: per step movement  
    """
    arg = argparse.ArgumentParser("Start running")

    arg.add_argument("--islocal", "-l", default=1, type=bool, help="choose local matching instead of global matching")
    arg.add_argument("--isoption", "-o", default=1, type=bool, help="choose covering option or not")
    arg.add_argument("--ischarging", "-c", default=0, type=bool, help="choose charging option or not")
    arg.add_argument("--num_option", "-option", default=NUM_OPTION, type=int, help="number of options to append")
    args = arg.parse_args()
    if SIM_DAYS > 0:
        start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)  # start_time = 0
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))
        islocal = "l" if args.islocal else "nl"
        isoption = "o" if args.isoption else "no"
        ischarging = "c" if args.ischarging else "nc"
        simulator = Simulator(start_time, TIMESTEP,args.isoption,args.islocal,args.ischarging)
        simulator.init(HEX_SHP_PATH, CS_SHP_PATH, TRIP_FILE, TRAVEL_TIME_FILE, NUM_NEAREST_CS)
        dqn_agent = DeepQNetworkAgent(simulator.hex_diffusions, args.num_option, args.isoption, args.islocal, args.ischarging)
        n_steps = int(3600 * 24 / TIMESTEP)  # number of time ticks per day
        with open('logs/vehicle_track/od_trajs_%d.csv'% args.num_option, 'w') as traj:
            for episode in range(10):
                #reinitialize the status of the simulator
                simulator.reset(start_time=episode*(end_time-start_time),timestep=TIMESTEP)
                for day in range(SIM_DAYS):
                    print("############################ DAY {} SUMMARY ################################".format(day))
                    for i in range(n_steps):
                        tick = simulator.get_current_time()
                        start_tick = time.time()
                        simulator.step()
                        t1 = time.time() - start_tick
                        local_state_batches, num_valid_relos, assigned_option_ids = simulator.get_local_states()
                        # t2 = time.time() - start_tick
                        global_state = simulator.get_global_state()
                        # t3 = time.time() - start_tick
                        # if tick >0 and np.sum(global_state) == 0: # check if just reset
                        #     global_state = global_state_slice
                        if len(local_state_batches) > 0:
                            converted_action_set, assigned_options = dqn_agent.get_actions(local_state_batches, num_valid_relos,assigned_option_ids, global_state)
                            simulator.attach_actions_to_vehs(converted_action_set, assigned_options)
                        simulator.update()  # update time, get metrics.

                        if tick % (1440*60) == 0:
                            simulator.store_transitions_from_veh()
                            prime_start_states,current_hexes, end_states, ticks = simulator.dump_prime_action_to_dqn()
                            if prime_start_states is not None:
                                for state,next_state in zip(prime_start_states,end_states):
                                    traj.writelines('{},{},{}\n'.format(state[0]//60%24,state[1],next_state[1]))
                            # now reset transition and global state
                            simulator.reset_storage()
