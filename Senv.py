import numpy as np
import scipy.io as sio
import sys

NUM_UE = 50
NUM_Channel = 25
He =  abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel,NUM_UE) + 1j * np.random.randn(NUM_Channel,NUM_UE)))# 边缘
Hc =  0.1*abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel,NUM_UE) + 1j * np.random.randn(NUM_Channel,NUM_UE)))# 云

# # 存储
# save_fn = 'Data_He.mat'
# sio.savemat(save_fn,{'He': He})
# save_fn = 'Data_Hc.mat'
# sio.savemat(save_fn,{'Hc': Hc})
# 读取
# load_He = 'Data_He.mat'
# He = sio.loadmat(load_He)['He']
# load_Hc = 'Data_Hc.mat'
# Hc = sio.loadmat(load_Hc)['Hc']
B = 1000000000/NUM_UE# 每个用户的信道带宽
var_noise = 10**(-5)

class env_network:
    def __init__(self, Task_coef, Pe, Pc, fe, fc, alpha, beta, T_max):
        self.Task_coef = Task_coef
        self.Pe = Pe
        self.Pc = Pc
        self.fe = fe
        self.fc = fc
        self.alpha = alpha
        self.beta = beta
        self.T_max = T_max
        self.length_Pe = len(Pe)
        self.length_Pc = len(Pc)
        self.length_Task_coef = len(Task_coef)
        self.action_space = np.arange(self.length_Pe * self.length_Pc * self.length_Task_coef) # dimension of one user action

    def reset(self):
        pass

    def sample(self):
        x = np.random.choice(self.action_space, size=NUM_UE)
        return x

    # QoS
    def sum_rate(self, UE_Channel_matrix, He, Hc, pe, pc, B, var_noise):
        rate_edge = np.zeros((NUM_Channel,NUM_UE))
        rate_cloud = np.zeros((NUM_Channel,NUM_UE))
        # 边缘网络
        for n in range(NUM_Channel):
            U = np.transpose(np.nonzero(UE_Channel_matrix[n,:]))
            L = len(U)
            for m in range(L):
                if L > 1:
                    Com_H = He[n, U[m]]
                    # 串行干扰消除干扰用户计算
                    I1 = np.zeros(L)
                    I2 = np.zeros(L)
                    for m1 in range(L):  # 依次检验所有用户
                        if Com_H <= He[n, U[m1]]:  # 大信号 包含原信号
                            I1[m1] = He[n, U[m1]] ** 2 * pe[U[m1]]
                        else:
                            I2[m1] = 0.01 * He[n, U[m1]] ** 2 * pe[U[m1]]
                    rate_edge[n,U[m]] = B * np.math.log2(1 + He[n, U[m]] ** 2 * pe[U[m]] / (var_noise + sum(I1[:]) + sum(I2[:]) - He[n, U[m]] ** 2 * pe[U[m]]))
                elif L == 1:
                    rate_edge[n,U[m]] = B * np.math.log2(1 + He[n, U[m]] ** 2 * pe[U[m]] / var_noise)

        # 云网络
        for n in range(NUM_Channel):
            U = np.transpose(np.nonzero(UE_Channel_matrix[n,:]))
            L = len(U)
            for m in range(L):
                if L > 1:
                    Com_H = Hc[n, U[m]]
                    I1 = np.zeros(L)
                    I2 = np.zeros(L)
                    for m1 in range(L):
                        if Com_H <= Hc[n, U[m1]]:
                            I1[m1] = Hc[n, U[m1]] ** 2 * pc[U[m1]]
                        else:
                            I2[m1] = 0.01 * Hc[n, U[m1]] ** 2 * pc[U[m1]]
                    rate_cloud[n,U[m]] =  B * np.math.log2(1 + Hc[n, U[m]] ** 2 * pc[U[m]] / (var_noise + sum(I1[:]) + sum(I2[:]) - Hc[n, U[m]] ** 2 * pc[U[m]]))
                elif L == 1:
                    rate_cloud[n,U[m]] =  B * np.math.log2(1 + Hc[n, U[m]] ** 2 * pc[U[m]] / var_noise)

        # for m in range(NUM_UE):  # 遍历所有的L
        #     Com_H = Hc[m]  # 当前用户的信道增益
        #     # 串行干扰消除干扰用户计算
        #     I1 = np.zeros(NUM_UE)
        #     I2 = np.zeros(NUM_UE)
        #     for m1 in range(NUM_UE):  # 依次检验所有用户
        #         if Com_H <= Hc[m1]:  # 大信号 包含原信号
        #             I1[m1] = Hc[m1] ** 2 * pc[m1]
        #         else:
        #             I2[m1] = 0.1*Hc[m1] ** 2 * pc[m1]
        #     # 用户U[j]所接入WiFi的其他用户
        #     rate_cloud[m] = 100 * B * np.math.log2(1 + Hc[m] ** 2 * pc[m] / (var_noise + sum(I1[:]) + sum(I2[:]) - Hc[m] ** 2 * pc[m]))
        # print(rate_edge)
        # print(rate_cloud)
        return rate_edge, rate_cloud

    def step(self, action, task_current, UE_Channel_matrix):
        print(action)
        obs = []
        task_coef = np.zeros([NUM_UE])
        pe = np.zeros([NUM_UE])
        pc = np.zeros([NUM_UE])
        for j in range(NUM_UE):
            task_coef[j] = self.Task_coef[int(action[j] / (self.length_Pe *self.length_Pc))]
            # task_coef[j] = 1# 完全过载到云端
            # task_coef[j] = 0  # 完全过载到边缘网络
            pe[j] = self.Pe[int(action[j] % (self.length_Pe *self.length_Pc) % self.length_Pe)]  # 余数
            pc[j] = self.Pc[int(action[j] % (self.length_Pe *self.length_Pc) / self.length_Pe)]
        reward, E, T = self.compute_reward(UE_Channel_matrix, task_coef, pe, pc, task_current)
        for j in range(NUM_UE):
            obs.append((reward[j], E[j], T[j]))
        return obs

    def compute_reward(self, UE_Channel_matrix, task_coef, pe, pc, task_current):
        E_off = np.zeros([NUM_UE])
        E_exe = np.zeros([NUM_UE])
        E = np.zeros([NUM_UE])
        T_off = np.zeros([NUM_UE])
        T_exe = np.zeros([NUM_UE])
        T = np.zeros([NUM_UE])
        reward = np.zeros([NUM_UE])

        rate_edge, rate_cloud = self.sum_rate(UE_Channel_matrix, He, Hc, pe, pc, B, var_noise)
        for j in range(NUM_UE):
            i = np.transpose(np.nonzero(UE_Channel_matrix[:,j]))[0][0]
            E_off[j] = (pe[j] * task_coef[j] * task_current[j]) / rate_edge[i,j] + (pc[j] * (1 - task_coef[j]) * task_current[j]) / rate_cloud[i,j]
            T_off[j] = (task_coef[j] * task_current[j]) / rate_edge[i, j] + ((1 - task_coef[j]) * task_current[j]) / rate_cloud[i,j]
            E_exe[j] = self.beta * (self.alpha * task_coef[j] * task_current[j] * self.fe ** 2 + self.alpha * (1- task_coef[j]) * task_current[j]* self.fc ** 2)
            T_exe[j] = (self.alpha * task_coef[j] * task_current[j]) / self.fe + (self.alpha * (1 - task_coef[j]) * task_current[j]) / self.fc
            E[j] = E_off[j] + E_exe[j]
            T[j] = T_off[j] + T_exe[j]
            reward[j] = 1/E[j]

        return reward, E, T

    # It creates a one hot vector of a number as num with size as len
    def one_hot(self, num, len):
        assert num >= 0 and num < len, "error"
        vec = np.zeros([len], np.int32)
        vec[num] = 1
        return vec

    # generates next-state from action and observation
    def state_generator(self, action, obs, task_current):
        input_vector = []
        if action is None:
            print('None')
            sys.exit()
        for user_i in range(action.size): # Traversing each UE' action
            input_vector_i = self.one_hot(action[user_i], self.length_Pe * self.length_Pc * self.length_Task_coef) # action
            Task_alloc = self.one_hot(int(task_current[user_i] / (10 ** 6) - 1), 10)  # lam 5
            # Task_alloc = self.one_hot(int(task_current[user_i] / (10 ** 6) - 2), 10)  # lam 6
            input_vector_i = np.append(input_vector_i, Task_alloc) # 任务
            input_vector.append(input_vector_i)
        return input_vector

    ##################################obtain date from batch##########################
    def get_states_user(self,batch):
        states = []
        for user in range(NUM_UE):
            states_per_user = []
            for each in batch:
                states_per_batch = []
                for step_i in each:

                    try:
                        states_per_step = step_i[0][user]

                    except IndexError:
                        print(step_i)
                        print("-----------")

                        print("eror")

                        '''for i in batch:
                            print i
                            print "**********"'''
                        sys.exit()
                    states_per_batch.append(states_per_step)
                states_per_user.append(states_per_batch)
            states.append(states_per_user)
        states = np.reshape(np.array(states), [-1, np.array(states).shape[2], np.array(states).shape[3]])
        return states

    def get_actions_user(self,batch):
        actions = []
        for user in range(NUM_UE):
            actions_per_user = []
            for each in batch:
                actions_per_batch = []
                for step_i in each:
                    actions_per_step = step_i[1][user]
                    actions_per_batch.append(actions_per_step)
                actions_per_user.append(actions_per_batch)
            actions.append(actions_per_user)
        actions = np.reshape(np.array(actions), [-1, np.array(actions).shape[2]])
        return actions

    def get_rewards_user(self,batch):
        rewards = []
        for user in range(NUM_UE):
            rewards_per_user = []
            for each in batch:
                rewards_per_batch = []
                for step_i in each:
                    rewards_per_step = step_i[2][user]
                    rewards_per_batch.append(rewards_per_step)
                rewards_per_user.append(rewards_per_batch)
            rewards.append(rewards_per_user)
        rewards = np.reshape(np.array(rewards), [-1, np.array(rewards).shape[2]])
        return rewards

    #
    def get_next_states_user(self,batch):
        next_states = []
        for user in range(NUM_UE):
            next_states_per_user = []
            for each in batch:
                next_states_per_batch = []
                for step_i in each:
                    next_states_per_step = step_i[3][user]
                    next_states_per_batch.append(next_states_per_step)
                next_states_per_user.append(next_states_per_batch)
            next_states.append(next_states_per_user)
        next_states = np.reshape(np.array(next_states), [-1, np.array(next_states).shape[2], np.array(next_states).shape[3]])
        return next_states



