import argparse
import time
import glob,os
import shutil
import numpy as np
import pandas as pd
from common.time_utils import get_local_datetime
from config.hex_setting import HEX_SHP_PATH, CS_SHP_PATH, NUM_NEAREST_CS, TRIP_FILE, TRAVEL_TIME_FILE, \
    TIMESTEP, START_OFFSET, SIM_DAYS, N_EPISODE, START_TIME, TRAINING_CYCLE, UPDATE_CYCLE, STORE_TRANSITION_CYCLE, CNN_RESUME, OPTION_DIM,LEARNING_RATE
# from dqn_option_agent.dqn_option_agent import DeepQNetworkOptionAgent
from dqn_agent.dqn_agent_with_cnn.cnn_dqn_agent import DeepQNetworkAgent
from simulator.simulator_cnn import Simulator
from torch.utils.tensorboard import SummaryWriter
# ---------------MAIN FILE---------------

if __name__ == '__main__':
    arg = argparse.ArgumentParser("Start running")

    arg.add_argument("--islocal", "-l", default=1, type=bool, help="choose local matching instead of global matching")
    arg.add_argument("--isoption", "-o", default=0, type=bool, help="set number of options")
    arg.add_argument("--ischarging", "-c", default=0, type=bool, help="choose charging option or not")
    arg.add_argument("--trial", "-t", default=0, type=int, help="sequence of running")
    args = arg.parse_args()
    if SIM_DAYS > 0:
        start_time = START_TIME + int(60 * 60 * 24 * START_OFFSET)  # start_time = 0
        print("Simulate Episode Start Datetime: {}".format(get_local_datetime(start_time)))
        end_time = start_time + int(60 * 60 * 24 * SIM_DAYS)
        print("Simulate Episode End Datetime : {}".format(get_local_datetime(end_time)))
        islocal = "l" if args.islocal else "nl"
        isoption = "o" if args.isoption else "no"
        ischarging = "c" if args.ischarging else "nc"
        ttrial=int(args.trial)


        #clear the log files
        # try:
        #     shutil.rmtree('logs/results_log_tfboard_{}_{}_{}'.format(LEARNING_RATE, OPTION_DIM,trial))
        # except Exception as e:
        #     print('Failed to delete. Reason: %s' % (e))


        simulator = Simulator(start_time, TIMESTEP,args.isoption,args.islocal,args.ischarging)
        simulator.init(HEX_SHP_PATH, CS_SHP_PATH, TRIP_FILE, TRAVEL_TIME_FILE, NUM_NEAREST_CS)

        for seeds in range(5): #10 runs
            trial=ttrial+seeds
            print('Current exeuction trail {} with setting charging ={} option={} local matching={}'.format(trial,
                                                                                                            ischarging,
                                                                                                            isoption,
                                                                                                            islocal))
            writer = SummaryWriter('logs/results_log_tfboard_{}_{}_{}'.format(LEARNING_RATE, OPTION_DIM, trial))
            dqn_agent = DeepQNetworkAgent(simulator.hex_diffusions,OPTION_DIM, args.isoption,args.islocal,args.ischarging,writer)
            dqn_agent.neighbor_id=simulator.all_neighbors # 1347 by 7 matrix

            dqn_agent.od_time=simulator.od_time/60 #convert into minutes

            # DeepQNetworkOptionAgent(simulator.hex_diffusions,isoption,LEARNING_RATE,ischarging)
            n_steps = int(3600 * 24 / TIMESTEP)  # number of time ticks per day


            with open('logs/parsed_results_{}_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging,trial), 'w') as f,\
                    open('logs/target_charging_stations_{}_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging,trial), 'w') as g, \
                    open('logs/training_hist_{}_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging,trial), 'w') as h, \
                    open('logs/demand_supply_gap_{}_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging,trial), 'w') as l1,\
                    open('logs/cruising_od_{}_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging,trial), 'w') as m1, \
                    open('logs/matching_od_{}_{}_{}_{}.csv'.format(OPTION_DIM,LEARNING_RATE,ischarging,trial), 'w') as n1:
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
                g.writelines('{},{},{}\n'.format("tick", "cs_id","destination_cs_id"))
                h.writelines('{},{},{},{},{},{}\n'.format("step", "loss", "reward", "learning_rate","sample_reward","sample_SOC"))
                l1.writelines('{},{},{}\n'.format("step", "hex_zone_id", "demand_supply_gap"))
                m1.writelines('{},{},{}\n'.format("step","origin_hex","destination_hex"))
                n1.writelines('{},{},{}\n'.format("step", "origin_hex", "destination_hex"))
                current_epsilon = dqn_agent.decayed_epsilon
                epi_count=0
                for episode in range(20):
                    if episode in [0,6,7,13,14,20,21,27,28]:
                        continue
                    #reinitialize the status of the simulator
                    simulator.reset(start_time=episode*(end_time-start_time),timestep=TIMESTEP,seed=episode+1)
                    for day in range(SIM_DAYS):
                        print("############################ DAY {} SUMMARY ################################".format(day))
                        for i in range(n_steps):
                            tick = simulator.get_current_time()
                            start_tick = time.time()
                            global_state = simulator.get_global_state()
                            local_state_batches, num_valid_relos,assigned_option_ids = simulator.get_local_states()
                            nidle=[veh for hx in simulator.hex_zone_collection.values() for veh in hx.vehicles.values() if veh.state.status==0] #idle vehicles

                            #set f
                            # if OPTION_DIM>0 and dqn_agent.f_train_step>0:
                            #     dqn_agent.get_local_f(global_state,tick)
                            v_idx = []
                            opt_idx = []
                            if len(local_state_batches) > 0:
                                # dump terminal transitions for those vehicle
                                if OPTION_DIM>0:
                                    terminal_flag, v_idx, opt_idx= dqn_agent.is_terminal(local_state_batches,tick) #return terminal states and not choosing options
                                    # terminal_flag=[1 if veh.option_cruise>2 else 0 for veh in nidle]
                                    terminal_veh=0
                                    for veh in range(len(nidle)):
                                        if nidle[veh].option_cruise>4 or veh in v_idx: #terminal
                                            nidle[veh].save_terminal(tick); terminal_veh+=1
                                else:
                                    v_idx=[];opt_idx=[]

                                local_state_batches, num_valid_relos, assigned_option_ids = simulator.get_local_states()
                                action_selected,action_to_execute,assigned_opts,contd_opts,status_code=dqn_agent.get_actions(local_state_batches, num_valid_relos, global_state,assigned_option_ids, v_idx, opt_idx,tick)

                                #set the status of vehicle (if random or not)
                                for v,s in zip(nidle, status_code):
                                    v.on_random=s

                                # print('Action selected 0={}, 1={},2={},3={}'.format(np.mean(act1),np.mean(act2),np.mean(act3),np.mean(act4)))

                                if OPTION_DIM==0:
                                    simulator.attach_actions_to_vehs(action_selected,action_to_execute)
                                    stay_select = sum([1 for a in action_selected if a == 0]) / len(action_selected)
                                    dqn_agent.writer.add_scalar('main_dqn/stay_percent', stay_select,
                                                                dqn_agent.train_step)
                                else:
                                    simulator.attach_actions_to_vehs(action_selected,action_to_execute,assigned_opts,contd_opts)
                                    if len(action_selected)>0:
                                        opt_select = sum([1 for a in action_selected if a> 6]) / len(action_selected)
                                        stay_select = sum([1 for a in action_selected if a == 0]) / len(action_selected)
                                        contd_prop = sum([1 for a in contd_opts if a == True]) / len(contd_opts + 0.00001)
                                        # print(
                                        #     'Propotion of option selected={:.2f}, percent of continued option={:.2f},propotion of stay selected={:.2f}'.format(
                                        #         opt_select, contd_prop, stay_select))
                                        dqn_agent.writer.add_scalar('main_dqn/stay_percent', stay_select,
                                                                    dqn_agent.train_step)
                                        dqn_agent.writer.add_scalar('main_dqn/option_percent', opt_select,
                                                                    dqn_agent.train_step)
                                        for options in range(7,16):
                                            count=sum([1 for a in action_selected if a==options])/len(action_selected)
                                            dqn_agent.writer.add_scalar('main_dqn/option_{}_percent'.format(options), count,
                                                                            dqn_agent.train_step)


                            simulator.step()
                            t1 = time.time() - start_tick
                            hrs=(i//60)%24
                            # t2 = time.time() - start_tick

                            # t3 = time.time() - start_tick
                            # if tick >0 and np.sum(global_state) == 0: # check if just reset
                            #     global_state = global_state_slice

                            # t4 = time.time() - start_tick
                            simulator.update()  # update time, get metrics.
                            # t5 = time.time() - start_tick
                            (num_idle, num_serving, num_charging, num_cruising, n_matches, total_num_arrivals,
                             total_removed_passengers, num_assigned, num_waitpile, num_tobedisptached, num_offduty,
                             average_reduced_SOC, total_num_longwait_pass, total_num_served_pass, average_cumulated_earning) = simulator.summarize_metrics(l1, g, m1, n1)

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
                                states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_ = simulator.dump_transition_to_dqn()
                                if states is not None:
                                    [dqn_agent.add_transition(states, actions, next_states, rewards, terminate_flag, time_steps,  valid_action_num_) for
                                     states, actions, next_states, rewards, terminate_flag, time_steps, valid_action_num_ in zip(states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_)]
                                    # print('For episode {}, tick {}, average reward is {},replay buffer={}'.format(episode, tick/60,np.mean(rewards),len(dqn_agent.memory.memory)))
                                gstates=simulator.dump_global()
                                # print('now adding global state...size=',len(gstates.keys()))
                                dqn_agent.add_global_state_dict(gstates)  # a 4-dim np array

                                t1=time.time()
                                states, next_states,on_opts = simulator.dump_f_transitions()
                                hrs=0
                                if states is not None:
                                    [dqn_agent.add_f_transition([state, next_state,on_opt],hrs) for
                                     state, next_state,on_opt in
                                     zip(states, next_states,on_opts)]

                                trajectory=simulator.dump_trajectories()
                                if trajectory is not None:
                                    [dqn_agent.add_trajectories(traj) for traj in trajectory]
                                print('Trajectory record length:',len(dqn_agent.trajectory_memory.memory))
                                # print('Replay buffer size:',len(dqn_agent.memory.memory[0].memory),len(dqn_agent.memory.memory[-1].memory))
                                # print('Add f_transition cost={}'.format(time.time()-t1))

                                # t1=time.time()
                                # states, next_states = simulator.dump_fo_transitions()
                                # if states is not None:
                                #     [dqn_agent.add_fo_transition([state, next_state]) for
                                #      state, next_state in
                                #      zip(states, next_states)]
                                # # print('Add fo_transition cost={}'.format(time.time()-t1))
                                # t1=time.time()


                                states, actions, next_states, trip_flags, time_steps, valid_action_nums_ = simulator.dump_prime_action_to_dqn()
                                if states is not None:
                                    [dqn_agent.add_H_transition(state, action, next_state, flag, time, valid_loc) for
                                     state, action, next_state, flag, time, valid_loc in
                                     zip(states, actions, next_states, trip_flags, time_steps, valid_action_nums_)]
                                # now reset transition and global state
                                simulator.reset_storage()

                            t6 = time.time() - start_tick
                            t_start = time.time()
                            if tick % TRAINING_CYCLE == 0 and epi_count>0:
                                    dqn_agent.train(h) #

                            if OPTION_DIM>0:
                                # if epi_count>0:dqn_agent.train_f_online(5)
                                if dqn_agent.n_f_nets>0:
                                    dqn_agent.train_h_trajectory_add(1)
                                    # dqn_agent.train_f()
                                    # dqn_agent.soft_target_update(1e-3)
                            if tick % UPDATE_CYCLE == 0:
                                    dqn_agent.copy_parameter()
                            if dqn_agent.h_train_step%100==0:
                                    if dqn_agent.n_f_nets > 0:
                                        dqn_agent.copy_H_parameter()
                                    print('----------------------------------------Update Target Network Now -----------------------------------\n\n\n')

                            t7 = time.time() - t_start

                    if OPTION_DIM>0 and epi_count in [0,3,10] and dqn_agent.n_f_nets<3:
                        dqn_agent.train_f(10000)

                    #last step transaction dump
                    tick=simulator.get_current_time()
                    simulator.store_global_states()
                    simulator.last_step_transactions(tick)
                    states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_ = simulator.dump_transition_to_dqn()
                    if states is not None:
                            [dqn_agent.add_transition(states, actions, next_states, rewards, terminate_flag, time_steps,
                                                      valid_action_num_) for
                             states, actions, next_states, rewards, terminate_flag, time_steps, valid_action_num_ in
                             zip(states, actions, next_states, rewards, terminate_flags, time_steps, valid_action_nums_)]
                            # print('For tick {}, average reward is {}'.format(tick / 60, np.mean(rewards)))

                    states, next_states,on_opts = simulator.dump_f_transitions()
                    if states is not None:
                        [dqn_agent.add_f_transition([state, next_state,on_opt],0) for
                         state, next_state,on_opt in
                         zip(states, next_states,on_opts)]

                    t1 = time.time()
                    states, actions, next_states, trip_flags, time_steps, valid_action_nums_ = simulator.dump_prime_action_to_dqn()
                    if states is not None:
                        [dqn_agent.add_H_transition(state, action, next_state, flag, time, valid_loc) for
                         state, action, next_state, flag, time, valid_loc in
                         zip(states, actions, next_states, trip_flags, time_steps, valid_action_nums_)]

                    # now reset transition and global state


                    gstates = simulator.dump_global()
                    dqn_agent.add_global_state_dict(gstates)  # a 4-dim np array
                    simulator.reset_storage()
                    if epi_count>10:
                        dqn_agent.save_parameter(trial)
                    epi_count+=1



