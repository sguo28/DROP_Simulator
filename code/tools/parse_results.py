import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def auto_save_metric_plots(df, time_step, rolling_over,lr,opt,seed):
    for i in range(len(df)):
        df[i] = df[i].apply(pd.to_numeric, errors='coerce')
        df[i] = df[i].fillna(0)
    ROLLING_OVER = rolling_over  # per week

    # time=df['time'].max()/ROLLING_OVER
    time = min([d['time'].max() for d in df]) / 60
    tmin=0
    tmax=16
    # time = 100 * 6
    # df_pass[df_pass[["num_matches", "pass_arrivals", "longwait_pass", "served_pass"]]<-1000]=0

    labels=[]


    colors=[]
    for l,o,s in zip(lr,opt,seed):
        print(l,o)
        if o>0:
            if abs(s-12000)<100:
                labels.append('inf')
                colors.append('g')
            elif s//1000>=10:
                labels.append('DRDQN-{}'.format(o))
                colors.append('r')
            elif s//1000==25:
                labels.append('1-DRDQN-{}'.format(o))
                colors.append('k')
            elif s//1000==22:
                labels.append('SDRDQN-{}'.format(o))
                colors.append('g')
        else:
            labels.append('0 option')
            colors.append('b')
    diff_label=['total_reward']#['average_idle_time']
    [d.rename(columns={'average_idle_time':'total_reward'},inplace=True) for d in df]
    for d in df:
        d['removed_pass']+=d['served_pass']
    plot_labels=['num_idle','longwait_pass','num_serving','num_assigned','num_cruising','total_reward','removed_pass','num_matches']
    fig, axes = plt.subplots(nrows=len(plot_labels), ncols=1, figsize=(10, 6*len(plot_labels)))
    axe = axes.ravel()


    for d,label,color in zip(df,labels,colors):
        ax_id = 0
        for id, col in enumerate(d.columns.tolist()[1:]):
            # v=moving_average(d["%s" % col],rolling_over)
            if col in diff_label:
                diff_v=diff_value(d["%s" % col].to_numpy(),1440)
                diff_v= diff_v [:tmax*rolling_over]
                v=group_average(diff_v,rolling_over)
            elif col in plot_labels:
                v=group_average(d["%s" % col].to_numpy()[:tmax*rolling_over],rolling_over)
            else:
                continue

            v,upper,lower=moving_average(v,1)
            axe[ax_id].plot(v,'-.*', label=label,lw=0.5,ms=2,color=color)
            axe[ax_id].fill_between(np.arange(len(v)), lower, upper ,alpha=.1)
            #d["%s" % col].rolling(window=rolling_over).mean()[:-1].plot(ax=axe[ax_id], style='-.', label=label)
            # d["%s" % col].groupby(d.index // ROLLING_OVER).mean()[:-1].plot(ax=axe[ax_id], style='-.', label=label)
            axe[ax_id].set_ylabel(col)
            axe[ax_id].set_title(col)
            axe[ax_id].set_xlabel('Episode')
            axe[ax_id].set_xlim([tmin-1, tmax+1])
            axe[ax_id].legend(loc=0, prop={'size': 6})
            ax_id+=1

    # train_df = pd.DataFrame(train_name)

    plt.savefig('cnn_results.pdf')


def moving_average(x, w):
    #return np.convolve(x, np.ones(w), 'valid') / w
    return np.array([np.mean(x[i:i+w]) for i in range(len(x))]),np.array([np.percentile(x[i:i+w],75) for i in range(len(x))]),np.array([np.percentile(x[i:i+w],25) for i in range(len(x))])

def group_average(x,w):
    # return np.array([np.mean(x[i + w-1]) for i in range(0, len(x)-w, w)])
    return np.array([np.mean(x[i:i+w]) for i in range(0,len(x)-w,w)])

def diff_value(x,epi_len):
    vals=[]
    for i in range(0,len(x),epi_len):
        v=np.diff(x[i:i+epi_len])
        v=[0]+list(v)
        vals+=v

    return vals

if __name__ == "__main__":
    df=[]

    # options=[1]
    # lr='0.0001'
    # df=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(i,lr)) for i in options]

    # options=[3]
    # lr='0.001'
    # df+=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(i,lr)) for i in options]
    #
    # options=[0]
    # lr='0.001'
    # df+=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(i,lr)) for i in options]
                                     
    # # options=[1]
    # # lr='0.00012'
    # # df+=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(i,lr)) for i in options]
    # options=[0]
    # lr='0.0012'
    # df+=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(i,lr)) for i in options]

    # options=[1]
    # lr='0.0011'
    # df+=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(i,lr)) for i in options]
    #
    # options=[0]
    # lr='0.000149'
    # df+=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(i,lr)) for i in options]


    tseed=[12200,12201,12000,12001,12002,12003,12004]#,12002,12003,12004,12000]
    tlearning_rate=['0.005' for _ in range(len(tseed))]
    toptions=[3 for _ in range(len(tseed))]
    # tseed=[12100,11201,19600]#,12002,12003,12004,12000]
    # tlearning_rate=['0.005' for _ in range(len(tseed))]
    # toptions=[3,0,3]

    # tseed1=[800+i for i in range(3)]
    # learning_rate+=['0.001' for _ in range(len(tseed1))]
    # options+=[3 for _ in range(len(tseed1))]
    #
    # # seed+=tseed
    # seed+=tseed1New Folder


    options=[];seed=[];learning_rate=[]
    for option,lr,s in zip(toptions,tlearning_rate,tseed):
        try:
            print(option,lr,s)
            df+=[pd.read_csv('../logs/test_results/parsed_results_{}_{}_nc_{}.csv'.format(option,lr,s))]
            options.append(option)
            learning_rate.append(lr)
            seed.append(s)
        except:
            continue


    # learning_rate+=['0.000145','0.000145']
    # options+=[1]
    #
    # for option,lr in zip(options,learning_rate):
    #     df+=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(option,lr))]

    # #
    # options=[1]
    # lr='0.00015'
    # df+=[pd.read_csv('../logs/parsed_results_{}_{}_nc.csv'.format(i,lr)) for i in options]
    # #
    auto_save_metric_plots(df, 1, 1440,learning_rate,options,seed)
