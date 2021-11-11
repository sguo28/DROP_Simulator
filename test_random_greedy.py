import argparse
import glob
import time

from common.time_utils import get_local_datetime
from config.hex_setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS, N_EPISODE, START_TIME, OPTION_DIM, LEARNING_RATE, TEST_START
# from dqn_option_agent.dqn_option_agent import DeepQNetworkOptionAgent
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent_test import DeepQNetworkAgent
from simulator.simulator_cnn import Simulator

# ---------------The code is to use existing ones and perform the testing

if __name__ == '__main__':
    arg = argparse.ArgumentParser("Start running")
    arg.add_argument("--islocal", "-l", default=1, type=bool, help="choose local matching instead of global matching")
    arg.add_argument("--isoption", "-o", default=0, type=bool, help="set number of options")
    arg.add_argument("--ischarging", "-c", default=0, type=bool, help="choose charging option or not")
    args = arg.parse_args()
    start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)  # start_time = 0
    print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
    end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
    print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))
    islocal = "l" if args.islocal else "nl"
    isoption = "o" if args.isoption else "no"
    ischarging = "c" if args.ischarging else "nc"
    simulator = Simulator(start_time, TIMESTEP, args.isoption, args.islocal, args.ischarging)
    simulator.init(HEX_SHP_PATH, CS_SHP_PATH, TRIP_FILE, TRAVEL_TIME_FILE, NUM_NEAREST_CS)
    dqn_agent = DeepQNetworkAgent(simulator.hex_diffusions, OPTION_DIM, args.isoption, args.islocal,
                                  args.ischarging)
    dqn_agent.neighbor_id = simulator.all_neighbors  # 1347 by 7 matrix
    dqn_agent.od_time = simulator.od_time / 60  # convert into minutes
    #use 140000 for random policy and 15000 for greedy policy
    #11000 for DQN, 12000 for online DRDQN, 13000 for random DRDQN, 18000 for DRDQN

    use_random=0
    use_greedy=1
    dqn_agent.final_epsilon = 1
    t_trail=16000

    for s in range(1):
        if 1:  # 10 runs
            trial = t_trail+s
            print(
                'Current exeuction trail {} with setting charging ={} option={} local matching={}'.format(trial, ischarging,
                                                                                                          isoption,
                                                                                                          islocal))
            n_steps = int(3600 * 24 / TIMESTEP)  # number of time ticks per day
            with open(
                    'logs/test_results/parsed_results_{}_{}_{}_{}.csv'.format(OPTION_DIM, LEARNING_RATE, ischarging, trial),
                    'w') as f, \
                    open('logs/test_results/target_charging_stations_{}_{}_{}_{}.csv'.format(OPTION_DIM, LEARNING_RATE,
                                                                                             ischarging, trial), 'w') as g, \
                    open('logs/test_results/training_hist_{}_{}_{}_{}.csv'.format(OPTION_DIM, LEARNING_RATE, ischarging,
                                                                                  trial), 'w') as h, \
                    open('logs/test_results/demand_supply_gap_{}_{}_{}_{}.csv'.format(OPTION_DIM, LEARNING_RATE, ischarging,
                                                                                      trial), 'w') as l1, \
                    open('logs/test_results/cruising_od_{}_{}_{}_{}.csv'.format(OPTION_DIM, LEARNING_RATE, ischarging,
                                                                                trial), 'w') as m1, \
                    open('logs/test_results/matching_od_{}_{}_{}_{}.csv'.format(OPTION_DIM, LEARNING_RATE, ischarging,
                                                                                trial), 'w') as n1:
                f.writelines(
                    '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format("time", "num_idle", "num_serving",
                                                                               "num_charging",
                                                                               "num_cruising", "num_assigned",
                                                                               "num_waitpile",
                                                                               "num_tobedisptached", "average_idle_time",
                                                                               "num_matches", "average_idle_dist",
                                                                               "longwait_pass",
                                                                               "served_pass", "removed_pass",
                                                                               "consumed_SOC_per_cycle",
                                                                               "total_system_revenue"))
                g.writelines('{},{},{}\n'.format("tick", "cs_id", "destination_cs_id"))
                h.writelines(
                    '{},{},{},{},{},{}\n'.format("step", "loss", "reward", "learning_rate", "sample_reward", "sample_SOC"))
                l1.writelines('{},{},{}\n'.format("step", "hex_zone_id", "demand_supply_gap"))
                m1.writelines('{},{},{}\n'.format("step", "origin_hex", "destination_hex"))
                n1.writelines('{},{},{}\n'.format("step", "origin_hex", "destination_hex"))

                for episode in range(9):
                    if episode+16 in [0,6,7,13,14,20,21,27,28]:
                        continue
                    simulator.reset(start_time=episode * (end_time - start_time), timestep=TIMESTEP, seed=episode + 1+TEST_START)
                    for day in range(SIM_DAYS):
                        print("############################ DAY {} SUMMARY ################################".format(day))
                        for i in range(n_steps):
                            tick = simulator.get_current_time()
                            start_tick = time.time()
                            global_state = simulator.get_global_state()
                            local_state_batches, num_valid_relos, assigned_option_ids = simulator.get_local_states()
                            nidle = [veh for hx in simulator.hex_zone_collection.values() for veh in hx.vehicles.values() if
                                     veh.state.status == 0]  # idle vehicles

                            v_idx = []
                            opt_idx = []
                            if len(local_state_batches) > 0:
                                # dump terminal transitions for those vehicle
                                local_state_batches, num_valid_relos, assigned_option_ids = simulator.get_local_states()
                                action_selected, action_to_execute, assigned_opts, contd_opts, status_code = dqn_agent.get_actions(
                                    local_state_batches, num_valid_relos, global_state, assigned_option_ids, v_idx, opt_idx,
                                    tick)

                                if OPTION_DIM == 0:
                                    simulator.attach_actions_to_vehs(action_selected, action_to_execute)

                                else:
                                    simulator.attach_actions_to_vehs(action_selected, action_to_execute, assigned_opts,
                                                                     contd_opts)
                            simulator.step()
                            t1 = time.time() - start_tick
                            hrs = (i // 60) % 24
                            simulator.update()  # update time, get metrics.
                            (num_idle, num_serving, num_charging, num_cruising, n_matches, total_num_arrivals,
                             total_removed_passengers, num_assigned, num_waitpile, num_tobedisptached, num_offduty,
                             average_reduced_SOC, total_num_longwait_pass, total_num_served_pass,
                             average_cumulated_earning) = simulator.summarize_metrics(l1, g, m1, n1)

                            f.writelines(
                                '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(tick, num_idle, num_serving,
                                                                                           num_charging,
                                                                                           num_cruising, num_assigned,
                                                                                           num_waitpile,
                                                                                           num_tobedisptached, num_offduty,
                                                                                           n_matches,
                                                                                           total_num_arrivals,
                                                                                           total_num_longwait_pass,
                                                                                           total_num_served_pass,
                                                                                           total_removed_passengers,
                                                                                           average_reduced_SOC,
                                                                                           average_cumulated_earning))
                            # dump transitions to DQN module
                            if 1:
                                simulator.store_transitions_from_veh()
                                simulator.store_f_action_from_veh()
                                simulator.store_prime_action_from_veh()
                                simulator.store_trajectory_from_veh()
                                gstates = simulator.dump_global()
                                dqn_agent.add_global_state_dict(gstates)  # a 4-dim np array
                                states, next_states, on_opts = simulator.dump_f_transitions()
                                simulator.reset_storage()

                    tick = simulator.get_current_time()
                    simulator.store_global_states()
                    gstates = simulator.dump_global()
                    dqn_agent.add_global_state_dict(gstates)  # a 4-dim np array
                    simulator.reset_storage()

