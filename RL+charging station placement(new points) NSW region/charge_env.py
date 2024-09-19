import numpy as np
import pandas as pd

class base_env:
    def __init__(self):
        self.car_num = 50  #汽车生成数量
        self.new_charging_n =50  #新充电桩安装数量
        self.obs_dim = self.new_charging_n*2  #状态维度
        self.act_dim = 2 #动作维度
        self.current_c = np.load('charging_data_minmax.npy') #当前充电桩位置信息
        self.high_traffic_sites = np.load('traffic_data_minmax.npy') #高密度交通地点信息
        self.high_traffic_sites_df = pd.DataFrame(self.high_traffic_sites) #高密度交通地点的dataframe，方便采样


    ## 环境初始化函数
    def reset(self):
        self.t = 0  #初始化时间步为0
        self.charging_ = self.current_c #初始化现有充电桩
        self.state = np.zeros(self.obs_dim) #初始化状态

        return self.state

    ##奖励函数
    def reward_fun(self,charging_,cars_):
        reward = 0

        ## 对每个车辆计算距离最近的充电桩地点并计算距离
        for i in range(cars_.shape[0]):
            dis = np.sqrt(np.sum((charging_-cars_[i])**2,axis=1))  #计算欧式距离
            inx = np.argmin(dis) #获取最小值点
            reward-=dis[inx] #奖励累计为-最小距离
            charging_ = np.delete(charging_,inx,0) #删除计算过的充电桩
        return reward

    def step(self,action):
        done = False
        cars_dists = self.high_traffic_sites_df.sample(self.car_num,replace=False)  #按照交通密集分布抽取车辆地点
        state = self.state.copy()
        state[self.t*2:self.t*2+self.act_dim] = action ##更新状态
        self.charging_ = np.concatenate([self.charging_,action.reshape(1,action.shape[0])],axis=0) ##更新现有充电桩

        reward = self.reward_fun(self.charging_,np.array(cars_dists)) #计算奖励

        self.t+=1

        if self.t==self.new_charging_n:
            # print(self.charging_)
            done = True

        self.state = state
        return self.state,reward,done,{}


## 环境测试

if __name__ =='__main__':
    env = base_env()
    state = env.reset()
    action = np.array([0.5,0.88])
    reward = 0
    d = False
    while not d:
        s,r,d,_ = env.step(action)
        print(s)
        reward+=r
    print(reward)