from Senv import env_network


class NetworkEnv(env_network):
    def __init__(self, Task_coef, Pe, Pc, fe, fc, alpha, beta, T_max):
        super().__init__(Task_coef, Pe, Pc, fe, fc, alpha, beta, T_max)
