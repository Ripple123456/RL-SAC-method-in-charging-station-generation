import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取原始数据
charging_s = pd.read_excel('./o_data/charging_stations.xlsx')
ht_sites = pd.read_excel('./o_data/high_traffic_sites.xlsx')
# print(charging_s['LATITUDE'])
X_charging_s = np.array(charging_s['LATITUDE'])
Y_charging_s = np.array(charging_s['LONGITUDE'])
size_c = X_charging_s.shape[0]

X_ht_sites = np.array(ht_sites['LATITUDE'])
Y_ht_sites = np.array(ht_sites['LONGITUDE'])
size_t = X_ht_sites.shape[0]

#拼接所有数据（现有充电桩和交通密集点）
X_all = np.concatenate([X_charging_s,X_ht_sites])
Y_all = np.concatenate([Y_charging_s,Y_ht_sites])


#数据归一化
X_max = np.max(X_all)
X_min = np.min(X_all)
Y_max = np.max(Y_all)
Y_min = np.min(Y_all)

X_c_new = (X_charging_s-X_min)/(X_max-X_min)
Y_c_new = (Y_charging_s-Y_min)/(Y_max-Y_min)

charging_data = np.concatenate([X_c_new.reshape(size_c,1),Y_c_new.reshape(size_c,1)],axis=1)

X_t_new = (X_ht_sites-X_min)/(X_max-X_min)
Y_t_new = (Y_ht_sites-Y_min)/(Y_max-Y_min)

traffic_data = np.concatenate([X_t_new.reshape(size_t,1),Y_t_new.reshape(size_t,1)],axis=1)

#保存归一化数据
np.save('charging_data_minmax.npy',charging_data)
np.save('traffic_data_minmax.npy',traffic_data)


#画图
plt.scatter(X_t_new,Y_t_new,c='g',
            label='high traffic sites')

plt.scatter(X_c_new,Y_c_new,c='r',
            label='charging station',marker='x',s=80)

plt.legend()
plt.show()