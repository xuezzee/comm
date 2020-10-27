from Senv import env_network
from SDRL import QNetwork, Memory
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
import scipy.io as sio
import math

## Define resource management model parameters
TIME_SLOTS = 3000
time_step = 0
Task_coef = [round(i / 10.0, 10) for i in range(1, 10)]
Pe = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
Pc = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
NUM_UE = 50 #
NUM_Channel = 25
fe = 10**14
fc = 10**15
alpha = 10**8 # number of CPU cycles one bits 10^8
beta = 10**(-46) # effective capacitance coefficient for each CPU cycle 10^−28~10^−27可根据用户变动暂时不动
T_max = 8
# 初始化用户关联信道
UE_Channel_matrix = np.zeros((NUM_Channel,NUM_UE))
for j in range(NUM_UE):
    if j < NUM_Channel:
        UE_Channel_matrix[j, j] = 1
    else:
        UE_Channel_matrix[int(j - NUM_Channel), j] = 1
UE_Channel_matrix = np.array(UE_Channel_matrix)


# Define DQN network parameters
Test_interval = 20
memory_size = 1000  # size of experience replay deque
batch_size = 6  # Num of batches to train at each time_slot
pretrain_length = batch_size  # this is done to fill the deque up to batch size before training
hidden_size = 128  # Number of hidden neurons
learning_rate = 1.1*10**-4   # learning rate DQN 2*10**-5  DDQN  10**-5 DuelingDQN  1.1*10**-4
explore_start = 0.02  # initial exploration rate
explore_stop = 0.01  # final exploration rate
decay_rate = 0.0001  # rate of exponential decay of exploration
gamma = 0.9  # discount  factor
step_size = 5  #length of history sequence for each datapoint in batch
action_size = len(Task_coef)*len(Pe)*len(Pc) # Action 任务调度系数+边缘功率比例+云功率比例
state_size = action_size + 10
alpha_net = 0 # co-operative fairness constant
beta_net = 10**-3 # 防止Q过大数据溢出

double_q = False
dueling_q = True
Task = []
if double_q == True:
    Total_reward_DDQN = []
    UE_reward_DDQN = []
    Total_E_DDQN = []
    UE_E_DDQN = []
    Loss_DDQN = []
elif dueling_q == True:
    Total_reward_DuelingDQN = []
    UE_reward_DuelingDQN = []
    Total_E_DuelingDQN = []
    UE_E_DuelingDQN = []
    Loss_DuelingDQN = []
else:
    Total_reward_DQN = []
    UE_reward_DQN = []
    Total_E_DQN = []
    Total_T_DQN = []
    UE_E_DQN = []
    Loss_DQN = []

class DQN_Main(object):  # Class definition
    def choose_action(self, state_vector, beta):
        action = np.zeros([NUM_UE], dtype=np.int32)
        # Exploration
        explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * time_step)  # curent exploration probability
        if explore_p > np.random.rand():
            action = env.sample()  # random action sampling
        # choosing action with max probability
        else:  # feeding the input-history-sequence of (t-1) slot for each user seperately
            for each_user in range(NUM_UE):  ############### each user makes decision
                feed = {mainQN.inputs_: state_vector[:, each_user].reshape(1, step_size, state_size)}  # Dimension 1 5 6
                Qs = sess.run(mainQN.output, feed_dict=feed)  # predicting Q-values of state respectively
                # Monte-carlo sampling from Q-values  (Boltzmann distribution)
                prob = ((1 - alpha_net) * np.exp(beta_net * Qs)) / np.sum(np.exp(beta_net * Qs)) + alpha_net / (NUM_UE + 1)  # Normalizing probabilities of each action  with temperature (beta)
                # 高斯分布  2*10**-4
                # prob = 1/(math.sqrt(2* 3.1415)*0.1)*np.exp(-(Qs)**2/(2*0.1**2)) #均值为0，方差为0.1
                # print(prob)
                # plt.plot(Qs, prob, 'o-')
                # plt.show()
                action[each_user] = np.argmax(prob, axis=1)
        return action

    def DQN_MainFunction(self):  ################
        #  Initialize local variables
        beta = 1  # Annealing constant for Monte - Carlo
        memory = Memory(max_size=memory_size)  # this is experience replay buffer(deque) from which each batch will be sampled and fed to the neural network for training
        history_input = deque(maxlen=step_size)  # this is our input buffer which will be used for predicting next Q-values
        task_counter = 0  # 控制测试

        # 存储
        Task = (np.random.poisson(lam=5, size=10000000))
        Task = 10**6*np.clip(Task, 1, 10) #lam = 5
        # Task = (np.random.poisson(lam=6, size=10000000))
        # Task = 10 ** 6 * np.clip(Task, 2, 11) #lam = 6
        # save_fn = 'Data_Task_40.mat'
        # sio.savemat(save_fn,{'Task': Task})
        # 读取
        # load_Task = 'Data_Task_40.mat'
        # Task = sio.loadmat(load_Task)['Task'][0]
        Task_current = Task[int(NUM_UE * task_counter): int(NUM_UE * (task_counter + 1))]
        # Initialize state
        action = env.sample()  # 随机动作采样
        obs = env.step(action, Task_current, UE_Channel_matrix) # 由动作得到观测
        state = env.state_generator(action, obs, Task_current)
        for i in range(pretrain_length * step_size * 5):
            action = env.sample()
            obs = env.step(action, Task_current, UE_Channel_matrix)
            next_state = env.state_generator(action, obs, Task_current)
            reward = [i[0] for i in obs[:NUM_UE]]
            memory.add((state, action, reward, next_state))
            state = next_state
            history_input.append(state)
            task_counter += 1
            Task_current = Task[int(NUM_UE * task_counter): int(NUM_UE * (task_counter + 1))]

        # Main function
        for time_step in range(TIME_SLOTS):

            # Training process
            if time_step % 50 == 0:
                if time_step < 5000:  # When the Annealing constant is 0.9, it reaches fixed value.
                    beta -= 0.001

            action = self.choose_action(np.array(history_input),beta)  # converting input history into numpy array5组状态，每组状态中包括三个用户的状态（三行）feeding the input-history-sequence of (t-1) slot for each user seperately
            obs = env.step(action, Task_current, UE_Channel_matrix)  # obs
            next_state = env.state_generator(action, obs, Task_current)
            reward = [i[0] for i in obs[:NUM_UE]]  # reward for all users given by environment
            memory.add((state, action, reward, next_state))  # add new experiences into the memory buffer as (state, action , reward , next_state) for training
            state = next_state
            history_input.append(state)  # add new experience to generate input-history sequence for next state

            # Testing process
            if time_step % Test_interval == 0:  ###############
                num = 20
                if double_q == True:
                    total_reward_DDQN = 0
                    ue_reward_DDQN = 0
                    total_E_DDQN = 0
                    ue_E_DDQN = 0
                elif dueling_q == True:
                    total_reward_DuelingDQN = 0
                    ue_reward_DuelingDQN = 0
                    total_E_DuelingDQN = 0
                    ue_E_DuelingDQN = 0
                else:
                    total_reward_DQN = 0
                    ue_reward_DQN = 0
                    total_E_DQN = 0
                    total_T_DQN = 0
                    ue_E_DQN = 0

                for ind in range(num):
                    eps = 0
                    ep_r = 0#
                    ep_t = 0
                    ep_e = 0
                    ep_uer = 0
                    ep_uee = 0
                    terminal = False
                    while (terminal == False) and eps < 10:
                        action = self.choose_action(np.array(history_input),beta)
                        obs = env.step(action,Task_current, UE_Channel_matrix)
                        next_state = env.state_generator(action, obs, Task_current)  # Generate next state from action and observation
                        reward = [i[0] for i in obs[:NUM_UE]]
                        E = [i[1] for i in obs[:NUM_UE]]  # 能耗
                        T = [i[2] for i in obs[:NUM_UE]]  # 时间
                        eps += 1
                        if np.all(T) <= T_max:
                            terminal = True
                        # inter = 0
                        # for i in range(NUM_UE):
                        #     if T[i]<T_max:
                        #         inter += 1
                        # # print(T)
                        # # print(inter)
                        # if inter >= NUM_UE-2:
                        #     # print('ning')
                        #     terminal = True
                            # print('ning')
                        if ind == num and terminal == True:
                            memory.add((state, action, reward, next_state))
                        state = next_state
                        history_input.append(state)
                        task_counter += 1
                        Task_current = Task[int(NUM_UE * task_counter): int(NUM_UE * (task_counter + 1))]
                        ep_uer += reward[0]
                        ep_uee += E[0]
                        ep_r += np.sum(reward)
                        ep_e += np.sum(E)
                        ep_t += np.sum(T)
                    if double_q == True:
                        total_reward_DDQN += ep_r
                        total_E_DDQN += ep_e
                        ue_reward_DDQN += ep_uer
                        ue_E_DDQN += ep_uee

                    elif dueling_q == True:
                        total_reward_DuelingDQN += ep_r
                        total_E_DuelingDQN += ep_e
                        ue_reward_DuelingDQN += ep_uer
                        ue_E_DuelingDQN += ep_uee
                    else:
                        total_reward_DQN += ep_r
                        total_E_DQN += ep_e
                        total_T_DQN += ep_t
                        ue_reward_DQN += ep_uer
                        ue_E_DQN += ep_uee


                if double_q == True:
                    total_reward_DDQN = total_reward_DDQN / num
                    Total_reward_DDQN.append(total_reward_DDQN)
                    ue_reward_DDQN = ue_reward_DDQN / num
                    UE_reward_DDQN.append(ue_reward_DDQN)
                    total_E_DDQN = total_E_DDQN / num
                    Total_E_DDQN.append(total_E_DDQN)
                    ue_E_DDQN = ue_E_DDQN / num
                    UE_E_DDQN.append(ue_E_DDQN)
                elif dueling_q == True:
                    total_reward_DuelingDQN = total_reward_DuelingDQN / num
                    Total_reward_DuelingDQN.append(total_reward_DuelingDQN)
                    ue_reward_DuelingDQN = ue_reward_DuelingDQN / num
                    UE_reward_DuelingDQN.append(ue_reward_DuelingDQN)
                    total_E_DuelingDQN = total_E_DuelingDQN / num
                    Total_E_DuelingDQN.append(total_E_DuelingDQN)
                    ue_E_DuelingDQN = ue_E_DuelingDQN / num
                    UE_E_DuelingDQN.append(ue_E_DuelingDQN)
                else:
                    total_reward_DQN = total_reward_DQN / num
                    Total_reward_DQN.append(total_reward_DQN)
                    ue_reward_DQN = ue_reward_DQN / num
                    UE_reward_DQN.append(ue_reward_DQN)
                    total_E_DQN = total_E_DQN / num
                    Total_E_DQN.append(total_E_DQN)
                    total_T_DQN = total_T_DQN / num
                    Total_T_DQN.append(total_T_DQN)
                    ue_E_DQN = ue_E_DQN / num
                    UE_E_DQN.append(ue_E_DQN)

            # Training block starts
            states, actions, rewards, next_states = memory.sample(batch_size, step_size)
            # creating target vector (possible best action)
            q_targnext, q_evalnext = sess.run([mainQN.output, mainQN.output_eval],
                                              # q_evalnext is Q traget, obtaining by Q estimation
                                              feed_dict={mainQN.inputs_: next_states,
                                                         mainQN.inputs: next_states})  # next observation

            # The main difference between DDQN and DQN
            if double_q == True:
                batch_index = np.arange(q_targnext.shape[0], dtype=np.int32)
                max_ind = np.argmax(q_evalnext,axis=1)  # the action that brings the highest value is evaluated by q_eval
                select_target_Qs = q_targnext[batch_index, max_ind]  #############################
            else:  # DuelingDQN与DQN采用相同的式子
                select_target_Qs = np.max(q_targnext, axis=1)

            targets = rewards[:, -1] + gamma * select_target_Qs  # Q_target =  reward + gamma * Q_next
            loss, _ = sess.run([mainQN.loss, mainQN.opt],  # calculating loss and train using Adam  optimizer
                               feed_dict={mainQN.inputs: states,
                                          mainQN.targetQs_: targets,
                                          mainQN.actions_: actions[:, -1]})# 6个用户当前的动作,一个batch
            if double_q == True:
                Loss_DDQN.append(loss)
            elif dueling_q == True:
                Loss_DuelingDQN.append(loss)
            else:
                Loss_DQN.append(loss)


        if double_q == True:
            save_fn = 'Data_DDQN_Channel30try.mat'
            sio.savemat(save_fn,
                            {'Total_reward_DDQN': Total_reward_DDQN, 'UE_reward_DDQN': UE_reward_DDQN, 'Total_E_DDQN': Total_E_DDQN, 'UE_E_DDQN': UE_E_DDQN})
        elif dueling_q == True:
            save_fn = 'Data_DuelingDQN_UE50try.mat'
            sio.savemat(save_fn,
                            {'Total_reward_DuelingDQN': Total_reward_DuelingDQN, 'UE_reward_DuelingDQN': UE_reward_DuelingDQN, 'Total_E_DuelingDQN': Total_E_DuelingDQN, 'UE_E_DuelingDQN': UE_E_DuelingDQN})
        else:
            save_fn = 'Data_DQN_try.mat'
            sio.savemat(save_fn,{'Total_reward_DQN': Total_reward_DQN, 'UE_reward_DQN': UE_reward_DQN, 'Total_E_DQN': Total_E_DQN, 'Total_T_DQN': Total_T_DQN, 'UE_E_DQN': UE_E_DQN})


if __name__ == '__main__':
    # class Instantiation
    tf.reset_default_graph()  # reseting default tensorflow computational graph
    env = env_network(Task_coef, Pe, Pc, fe, fc, alpha, beta, T_max)  # initializing the environment
    mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate, step_size=step_size,
                      state_size=state_size, action_size=action_size)  # initializing deep Q network
    saver = tf.train.Saver()  # saver object to save the checkpoints of the DQN to disk
    sess = tf.Session()  # initializing the session
    sess.run(tf.global_variables_initializer())  # initialing all the tensorflow variables

    # operation main function
    GLOBAL_DQNMain = DQN_Main()
    GLOBAL_DQNMain.DQN_MainFunction()

########################################################################################

if double_q == True:
    plt.plot(20*np.arange(len(Total_reward_DDQN)), Total_reward_DDQN, label=u'DDQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The total reward')
    plt.show()

    plt.plot(20 * np.arange(len(UE_reward_DDQN)), UE_reward_DDQN, label=u'DDQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The single user reward')
    plt.show()

    plt.plot(20 * np.arange(len(Total_E_DDQN)), Total_E_DDQN, label=u'DDQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The total energy consumption')
    plt.show()

    plt.plot(20 * np.arange(len(UE_E_DDQN)), UE_E_DDQN, label=u'DDQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The single user energy consumption')
    plt.show()
elif dueling_q == True:
    plt.plot(20*np.arange(len(Total_reward_DuelingDQN)), Total_reward_DuelingDQN, label=u'DuelingDQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The total reward')
    plt.show()

    plt.plot(20 * np.arange(len(UE_reward_DuelingDQN)), UE_reward_DuelingDQN, label=u'DuelingDQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The single user reward')
    plt.show()

    plt.plot(20 * np.arange(len(Total_E_DuelingDQN)), Total_E_DuelingDQN, label=u'DuelingDQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The total energy consumption')
    plt.show()

    plt.plot(20 * np.arange(len(UE_E_DuelingDQN)), UE_E_DuelingDQN, label=u'DuelingDQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The single user energy consumption')
    plt.show()
else:
    plt.plot(20*np.arange(len(Total_reward_DQN)), Total_reward_DQN, label=u'DQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The total reward')
    plt.show()

    plt.plot(20 * np.arange(len(UE_reward_DQN)), UE_reward_DQN, label=u'DQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The single user reward')
    plt.show()

    plt.plot(20 * np.arange(len(Total_E_DQN)), Total_E_DQN, label=u'DQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The total energy consumption')
    plt.show()

    plt.plot(20 * np.arange(len(UE_E_DQN)), UE_E_DQN, label=u'DQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('The single user energy consumption')
    plt.show()

    plt.plot(np.arange(len(Total_T_DQN)), Total_T_DQN, label=u'DQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Delay')
    foo_fig = plt.gcf()  # 'get current figure'
    foo_fig.savefig('Figure5d.eps', format='eps', dpi=1000)
    plt.show()

    plt.plot(np.arange(len(Loss_DQN)), Loss_DQN, label=u'DQN')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # foo_fig = plt.gcf()  # 'get current figure'
    # foo_fig.savefig('Figure5d.eps', format='eps', dpi=1000)
    plt.show()
