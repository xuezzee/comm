from Senv import env_network
import tensorflow as tf
import numpy as np

Task_coef = [round(i / 10.0, 10) for i in range(1, 10)]
Pe = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
Pc = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
NUM_UE = 50
fe = 10**14
fc = 10**15
alpha = 10**8
beta = 10**(-46)
T_max = 8
dueling_q = True


class QNetwork:
    def __init__(self,learning_rate=1.1*10**-4, action_size=len(Task_coef)*len(Pe)*len(Pc), state_size=len(Task_coef)*len(Pe)*len(Pc) + 10, hidden_size=128, step_size=5, name='QNetwork',
                 ):
        with tf.variable_scope(name):
            #####################build target network############
            self.inputs_ = tf.placeholder(tf.float32, [None, step_size, state_size], name='inputs_')  # each user each step each state, inputs_ [user, step_size, state_size]
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            ##########################################
            self.lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            self.lstm_out, self.state = tf.nn.dynamic_rnn(self.lstm, self.inputs_, dtype=tf.float32)
            self.reduced_out = self.lstm_out[:, -1, :]
            self.reduced_out = tf.reshape(self.reduced_out, shape=[-1, hidden_size])
            ###############################################################
            self.w2 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size]))
            self.b2 = tf.Variable(tf.constant(0.1, shape=[hidden_size]))
            self.h2 = tf.matmul(self.reduced_out, self.w2) + self.b2
            self.h2 = tf.nn.relu(self.h2)
            self.h2 = tf.contrib.layers.layer_norm(self.h2)

            if dueling_q == True:# Dueling DQN
                # 计算Value
                self.w3 = tf.Variable(tf.truncated_normal([hidden_size,1]))
                self.b3 = tf.Variable(tf.constant(0.1, shape=[1]))
                self.V = tf.matmul(self.h2, self.w3) + self.b3
                # 计算Advantage
                self.w3 = tf.Variable(tf.truncated_normal([hidden_size, action_size]))
                self.b3 = tf.Variable(tf.constant(0.1, shape=[action_size]))
                self.A = tf.matmul(self.h2, self.w3) + self.b3
                self.output = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))  # Q = V(s) + A(s,a)
            else:  # DDQN和DQN
                self.w3 = tf.Variable(tf.truncated_normal([hidden_size, action_size]))
                self.b3 = tf.Variable(tf.constant(0.1, shape=[action_size]))
                self.output = tf.matmul(self.h2, self.w3) + self.b3



            ######################build evaluate net ####################
            self.inputs = tf.placeholder(tf.float32, [None, step_size, state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')  # 原始版本未注销
            one_hot_actions = tf.one_hot(self.actions_, action_size)  # Binarization

            self.lstm_out_eval, self.state = tf.nn.dynamic_rnn(self.lstm, self.inputs, dtype=tf.float32)
            self.reduced_out_eval = self.lstm_out_eval[:, -1, :]
            self.reduced_out_eval = tf.reshape(self.reduced_out_eval, shape=[-1, hidden_size])

            self.h2_eval = tf.matmul(self.reduced_out_eval, self.w2) + self.b2
            self.h2_eval = tf.nn.relu(self.h2_eval)
            self.h2_eval = tf.contrib.layers.layer_norm(self.h2_eval)
            self.output_eval = tf.matmul(self.h2_eval, self.w3) + self.b3


            self.Q_eval = tf.reduce_sum(tf.multiply(self.output_eval, one_hot_actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q_eval))  # loss function
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            # self.opt = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)# adopt AdamOptimizer


from collections import deque
class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, step_size):
        env = env_network(Task_coef, Pe, Pc, fe, fc, alpha, beta, T_max)
        idx = np.random.choice(np.arange(len(self.buffer) - step_size),
                               size=batch_size,
                               replace=False)  # Sampling batch_size from buffer [ 26 135  97  30  34 126]
        res = []
        for i in idx:
            temp_buffer = []
            for j in range(step_size):
                temp_buffer.append(self.buffer[i + j])
            res.append(temp_buffer)
        states = env.get_states_user(res)  # matrix of rank 4 shape [NUM_SUSERS,batch_size,step_size,state_size]
        actions = env.get_actions_user(res)  # matrix of rank 3 shape [NUM_SUSERS,batch_size,step_size]
        rewards = env.get_rewards_user(res)  # matrix of rank 3 shape [NUM_SUSERS,batch_size,step_size]
        next_states = env.get_next_states_user(
            res)  # matrix of rank 4 shape [NUM_SUSERS,batch_size,step_size,state_size]
        return states, actions, rewards, next_states

