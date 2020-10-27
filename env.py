from Senv import env_network
import numpy as np
import random
NUM_UE = 50
NUM_Channel = 25
NEW_TASK_PROB = 0.2
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
DELTA_T = 0.001


class NetworkEnv():
    def __init__(self, Pe, Pc, fe, fc, alpha, beta, T_max):
        # self.Task_coef = Task_coef
        self.Pe = Pe
        self.Pc = Pc
        self.fe = fe
        self.fc = fc
        self.alpha = alpha
        self.beta = beta
        self.T_max = T_max
        # self.length_Pe = len(Pe)
        # self.length_Pc = len(Pc)
        # self.length_Task_coef = len(Task_coef)
        # self.action_space = np.arange(
        # self.length_Pe * self.length_Pc * self.length_Task_coef)  # dimension of one user action

    def reset(self, UE_Channel_matrix):
        self.x = np.zeros(NUM_UE)
        self.n_step = 0
        self.task_remain = np.zeros(NUM_UE)
        self.UE_Channel_matrix = UE_Channel_matrix
        task_num = np.random.poisson(lam=25, size=1)[0]
        new_task_users = np.random.choice(a=50, size=task_num, replace=False, p=None)
        new_task = np.zeros(task_num)
        for i in range(task_num):
            # temp = random.normalvariate(1000, 100)
            new_task[i] += random.normalvariate(1000000, 100)
        for i in range(task_num):
            u = new_task_users[i]
            self.task_remain[u] += new_task[i]

        obs = [[self.task_remain[i], self.Pe, self.x[i]] for i in range(NUM_UE)]
        return obs


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
        return rate_edge.transpose(), rate_cloud.transpose()

    def compute_reward(self, UE_Channel_matrix, task_coef, pe, pc, task_current):
        E_off = np.zeros([NUM_UE])
        E_exe = np.zeros([NUM_UE])
        E = np.zeros([NUM_UE])
        T_off = np.zeros([NUM_UE])
        T_exe = np.zeros([NUM_UE])
        T = np.zeros([NUM_UE])
        reward = np.zeros([NUM_UE])

        rate_edge, rate_cloud = self.sum_rate(UE_Channel_matrix, He, Hc, pe, pc, B, var_noise)
        rate_edge = rate_edge.transpose()
        rate_cloud = rate_cloud.transpose()
        for j in range(NUM_UE):
            i = np.transpose(np.nonzero(UE_Channel_matrix[:,j]))[0][0]
            E_off[j] = (pe[j] * task_coef[j] * task_current[j]) / rate_edge[i,j] + (pc[j] * (1 - task_coef[j]) * task_current[j]) / rate_cloud[i,j]
            T_off[j] = (task_coef[j] * task_current[j]) / rate_edge[i, j] + ((1 - task_coef[j]) * task_current[j]) / rate_cloud[i,j]
            E_exe[j] = self.beta * (self.alpha * task_coef[j] * task_current[j] * self.fe ** 2 + self.alpha * (1- task_coef[j]) * task_current[j]* self.fc ** 2)
            T_exe[j] = (self.alpha * task_coef[j] * task_current[j]) / self.fe + (self.alpha * (1 - task_coef[j]) * task_current[j]) / self.fc
            E[j] = E_off[j] + E_exe[j]
            T[j] = T_off[j] + T_exe[j]
            if E[j] == 0:
                reward[j] = 0
            else:
                reward[j] = 1/E[j]

        return reward, E, T

    def step(self, actions):
        He = abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel, NUM_UE) + 1j * np.random.randn(NUM_Channel, NUM_UE)))  # 边缘
        Hc = 0.1 * abs(1 / np.sqrt(2) * (np.random.randn(NUM_Channel, NUM_UE) + 1j * np.random.randn(NUM_Channel, NUM_UE)))  # 云
        rate_edge, rate_cloud = self.sum_rate(self.UE_Channel_matrix, He, Hc, np.array([0.3 for i in range(NUM_UE)]), np.array([0.4 for i in range(NUM_UE)]), B, var_noise)
        for i in range(NUM_UE):
            offloaded_data = (rate_cloud[i].sum() + rate_edge[i].sum()) * DELTA_T
            self.task_remain[i] -= offloaded_data
            if self.task_remain[i] < 0:
                self.task_remain[i] = 0

        print("task_remain:", self.task_remain)
        # print("channel gain:", He)
        reward, _, _ = self.compute_reward(self.UE_Channel_matrix, actions, np.array([0.3 for i in range(NUM_UE)]), np.array([0.4 for i in range(NUM_UE)]), self.task_remain)
        self.n_step += 1
        obs = [[self.task_remain[i], self.Pe, self.x[i]] for i in range(NUM_UE)]
        return obs, reward, False, None



if __name__ == '__main__':
    UE_Channel_matrix = np.zeros((NUM_Channel, NUM_UE))
    for j in range(NUM_UE):
        if j < NUM_Channel:
            UE_Channel_matrix[j, j] = 1
        else:
            UE_Channel_matrix[int(j - NUM_Channel), j] = 1
    UE_Channel_matrix = np.array(UE_Channel_matrix)
    env = NetworkEnv(0.3, 0.4, 10**14, 10**15, 10**8, 10**(-46), 8)
    env.reset(UE_Channel_matrix)
    for i in range(100):
        action = [random.random() for i in range(NUM_UE)]
        state, reward, done, info = env.step(action)
        # print(reward)

