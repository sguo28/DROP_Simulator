import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

type=['DRDQN-A','ODRQN','RDRDQN-1','DRDQN-inf','DQN','DRDQN-0','DRDQN-1','Random','Greedy']
metrics=['peak_reward_mean','peak_reward_std','op_reward_mean','op_reward_std','night_reward_mean','night_reward_std','overall_reward_mean','overall_reward_std','peak_reject','op_reject','night_reject','overall_reject']

metrics_dict=dict()
for m in metrics:
    metrics_dict[m]=[]
seed=[196,185,140,121,112,122,192,150,160]
option=[3,3,3,3,0,3,3,3,0]
lr=0.005
n_veh=500

#summarize reward
summary=dict()
for t in type:
    summary[t]=copy.deepcopy(metrics_dict)


for s,o,r in zip(seed,option,type):
    total_reward=[]
    k=5
    # if s==191 or s==192:
    #     k=2
    # else:
    #     k=5

    for i in range(k):
        # if i==0 and s==120:
        #     print('skip results for ssed 120')
        #     continue
        # if i==0 and s==191:
        #     print('skip results for ssed 120')
        #     continue

        files='../logs/test_results/parsed_results_{}_{}_nc_{}.csv'.format(o,lr,s*100+i)
        try:
            df=pd.read_csv(files,na_values=0)
            df.fillna(0, inplace=True)
            print(s, o, r, i)
            # df['total_system_reveue']=df['total_system_revenue']*n_veh
            # for i in range(len(df)):
            #     df[i] = df[i].apply(pd.to_numeric, errors='coerce')
            #     df[i] = df[i].fillna(0)
            # print(df.head(-5))
            # print(df.columns)
        except:
            print('Not available')
            continue

        for t in range(0,df.shape[0],1440):
            print(df.shape[0])
            if t==4*1440: continue
            # if t//1440+16 in [0,1,14,15,21,22,28,29]:
            #     continue
            total_reward=(df.loc[t+1439,['total_system_revenue']])/24
            night_reward=(df.loc[t+300,['total_system_revenue']]-df.loc[t+120,['total_system_revenue']])/3
            peak_reward=(1*(df.loc[t+9*60,['total_system_revenue']]-df.loc[t+7*60,['total_system_revenue']])+1*(df.loc[t+19*60,['total_system_revenue']]-df.loc[t+17*60,['total_system_revenue']]))/4
            op_reward = (df.loc[t + 14*60, ['total_system_revenue']] - df.loc[t + 9*60, ['total_system_revenue']])/5
            summary[r]['overall_reward_mean'].append(total_reward)
            summary[r]['peak_reward_mean'].append(peak_reward)
            summary[r]['op_reward_mean'].append(op_reward)
            summary[r]['night_reward_mean'].append(night_reward)
            overall_reject=np.sum(df.loc[t+60:t+1439,['longwait_pass']].to_numpy())/np.sum(df.loc[t+60:t+1439,['removed_pass']].to_numpy())
            #
            # plt.plot(df.loc[t+60:t+1438,['longwait_pass']])
            # plt.plot(df.loc[t +60:t + 1438, ['removed_pass']])
            # plt.show()
            night_reject=np.sum(df.loc[t+120:t+300,['longwait_pass']].to_numpy())/np.sum(df.loc[t+120:t+300,['removed_pass']].to_numpy())
            op_reject = np.sum(df.loc[t + 9*60:t + 14*60, ['longwait_pass']].to_numpy()) / np.sum(
                df.loc[t + 9*60:t + 14*60, ['removed_pass']].to_numpy())
            peak_reject=(1*np.sum(df.loc[t + 7*60:t + 9*60, ['longwait_pass']].to_numpy())+np.sum(df.loc[t + 17*60:t + 19*60, ['longwait_pass']].to_numpy())) / (1*np.sum(
                df.loc[t + 7*60:t + 9*60, ['removed_pass']].to_numpy())+np.sum(
                df.loc[t + 17*60:t + 19*60, ['removed_pass']].to_numpy()))
            summary[r]['overall_reject'].append(overall_reject)
            summary[r]['peak_reject'].append(peak_reject)
            summary[r]['op_reject'].append(op_reject)
            summary[r]['night_reject'].append(night_reject)



for type in summary.keys():
    for key in summary[type].keys():
        if key[-4:]=='mean' and summary[type][key]:
            mean_key=key
            std_key=key[:-4]+'std'
            summary[type][std_key]=np.std(summary[type][mean_key])
            summary[type][mean_key] = np.mean(summary[type][mean_key])
            print(type,mean_key[:-4],summary[type][mean_key],summary[type][std_key] )
        if key[-6:]=='reject' and summary[type][key]:
            summary[type][key] = np.mean(summary[type][key])
            print(type,key,summary[type][key])


# def process_info()