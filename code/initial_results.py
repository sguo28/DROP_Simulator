import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .simulator.settings import FLAGS
df = pd.read_csv('./logs/output_charge_day0.csv')
# df[df['cs_id']==int(0)]['wait_time'].plot(ylim = (0,50))
# plt.figure()
df['time'] = (df['time']-df['time'][0])//(60)
df = df.set_index('time')
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,4))

df.groupby('cs_id')['wait_time'].plot(ax=axes[0])
axes[0].set_ylabel('waiting time(min)')
axes[0].set_xlabel('time(min)')

df.groupby('cs_id')['queue_len'].plot(ax=axes[1])
axes[1].set_ylabel('number of queueing vehicles')
# axes[1].set_yticks(np.arange(0,24,4))
axes[1].set_xlabel('time(min)')

df.groupby('cs_id')['served_num'].plot(ax=axes[2])
axes[2].set_ylabel('number of served request')
axes[2].set_xlabel('time(min)')


plt.show()