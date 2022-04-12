import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
option=0
range_r=20

df=gpd.read_file('../../data/NYC_shapefiles/clustered_hex.shp')
f_file=pd.read_csv('../saved_f/hex_p_value_1000_{}.csv'.format(option),names=['hr','hex','vals'],skiprows=1)
print(df.shape,f_file.columns,f_file.shape)
print(f_file.head())


for hr in range(24):
    data=f_file[f_file['hr'].astype(int)==hr]['vals']

    median=np.median(data)
    print(np.mean(data),np.median(data), sum(data**2))
    zero_percentile=sum(data<0)/len(data)*100
    zero_percentile=50
    lower=np.percentile(data,max(zero_percentile-range_r,0))
    upper=np.percentile(data,min(zero_percentile+range_r,100))
    newdata=np.zeros(len(data))
    newdata[(data>lower) & (data<upper)] =1


    df['f_vals']=data.to_numpy()
    df['terminal']=newdata

    fig,ax=plt.subplots(1,2,figsize=[14,7])
    df.plot(column='terminal',ax=ax[0],cmap='Pastel1',legend=True)
    df.plot(column='f_vals',ax=ax[1],cmap='bwr',legend=True)
    plt.savefig('results_option_{}_hr_{}.pdf'.format(option,hr))