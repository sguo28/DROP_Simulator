import argparse
import time
import numpy as np
from common.time_utils import get_local_datetime
from config.hex_setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS,  START_TIME, TRAINING_CYCLE, STORE_TRANSITION_CYCLE, H_TRAIN_EPISODE
from dqn_option_agent.h_agent import H_Agent
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent import DeepQNetworkAgent
from simulator.simulator_cnn import Simulator


# ---------------MAIN FILE---------------
NUM_OPTION = 0
print('OPTION IS {}'.format(NUM_OPTION))
if __name__ == '__main__':
    """
    todo: dont forget to remove the epsilon=1.0 and SOC threshold = -10. o.w. all action is randomly selected.
    """
    arg = argparse.ArgumentParser("Start running")

    arg.add_argument("--islocal", "-l", default=1, type=bool, help="local matching or global matching")
    arg.add_argument("--isoption", "-o", default=1, type=bool, help="covering option or not")
    arg.add_argument("--ischarging", "-c", default=0, type=bool, help="charging or not")
    args = arg.parse_args()
    if SIM_DAYS > 0:
        start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)  # start_time = 0
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))
        islocal = "l" if args.islocal else "nl"
        isoption = "o" if args.isoption else "no"
        ischarging = "c" if args.ischarging else "nc"

        simulator = Simulator(start_time, TIMESTEP, args.isoption, args.islocal, args.ischarging)
        simulator.init(HEX_SHP_PATH, CS_SHP_PATH, TRIP_FILE, TRAVEL_TIME_FILE, NUM_NEAREST_CS)
        dqn_agent = DeepQNetworkAgent(simulator.hex_diffusions, args.isoption, args.islocal, args.ischarging)
        h_agent = H_Agent(simulator.hex_diffusions, simulator.xy_coords,NUM_OPTION, args.isoption, args.islocal, args.ischarging)

        n_steps = int(3600 * 24 / TIMESTEP)  # number of time ticks per day
        with open('logs/parsed_results_%s_%s_%s.csv' % (isoption, islocal, ischarging), 'w') as f, open(
                'logs/target_charging_stations_%s_%s_%s.csv' % (isoption, islocal, ischarging), 'w') as g, open(
                'logs/h_test_training_hist_%s_%s_%s_%d.csv' % (isoption, islocal, ischarging,NUM_OPTION), 'w') as h, open(
                'logs/demand_supply_gap_%s_%s_%s.csv' % (isoption, islocal, ischarging), 'w') as l1, open(
                'logs/cruising_od_%s_%s_%s.csv' % (isoption, islocal, ischarging), 'w') as m1, open(
                'logs/matching_od_%s_%s_%s.csv' % (isoption, islocal, ischarging), 'w') as n1, open('logs/vehicle_track/dummy_od_trajs_%d.csv'%NUM_OPTION, 'w') as traj:
            for episode in range(H_TRAIN_EPISODE): # we need at least 2e4 steps = ~ 14 days.
                # reinitialize the status of the simulator
                simulator.reset(start_time=episode * (end_time - start_time), timestep=TIMESTEP)
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
                            action_set, converted_action_set = dqn_agent.get_actions(local_state_batches, num_valid_relos,assigned_option_ids, global_state)
                            print('max_converted_action_id:{}'.format(np.max(converted_action_set)))
                            simulator.attach_actions_to_vehs(action_set, converted_action_set)
                        simulator.update()  # update time, get metrics.

                        # simulator.summarize_metrics(f, l1, g, m1, n1)

                        # dump transitions to OPTION module
                        if tick % STORE_TRANSITION_CYCLE == 0:
                            simulator.store_transitions_from_veh()
                            states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_ = simulator.dump_transition_to_dqn()
                            if states is not None:
                                [h_agent.add_transition(states, actions, next_states, rewards, terminate_flag,
                                                          time_steps, valid_action_num_) for
                                 states, actions, next_states, rewards, terminate_flag, time_steps, valid_action_num_ in
                                 zip(states, actions, next_states, rewards, terminate_flags, time_steps,
                                     valid_action_nums_)]
                                # print('For episode {}, tick {}, average reward is {}'.format(episode, tick / 60,
                                #                                                              np.mean(rewards)))
                            gstates = simulator.dump_global()
                            h_agent.add_global_state_dict(gstates)  # a 4-dim np array
                            # now reset transition and global state
                            simulator.reset_storage()

                        t6 = time.time() - start_tick
                        t_start = time.time()
                        if tick % TRAINING_CYCLE == 0:
                            h_agent.train(h)
                            h_agent.soft_target_update(1e-3)

                        # if tick % (500*60) == 0: # we do a hard update of target network every 500 ticks

                        t7 = time.time() - t_start

                # last step transaction dump
                tick = simulator.get_current_time()
                simulator.store_global_states()
                simulator.last_step_transactions(tick)
                states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_ = simulator.dump_transition_to_dqn()
                if states is not None:
                    for states, actions, next_states, rewards, terminate_flag, time_steps, valid_action_num_ in zip(
                            states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_):
                        h_agent.add_transition(states, actions, next_states, rewards, terminate_flag, time_steps,
                                               valid_action_num_)

                    print('For tick {}, average reward is {}'.format(tick / 60, np.mean(rewards)))
                gstates = simulator.dump_global()
                h_agent.add_global_state_dict(gstates)  # a 4-dim np array


