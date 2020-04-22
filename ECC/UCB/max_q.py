import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from Agent import Agent


class MAMAB():

    def __init__(self, no_bandits, no_agents, bandits, optimal_bandit_index, reward_vairance, delta, no_iter, n_j=5):
        self.no_bandits = no_bandits
        self.no_agents  = no_agents
        self.bandits = bandits
        self.delta = delta

        self.com_count = 0
        self.n_j = n_j
        self.agent_tot_regret_with_time = [[0] for i in range(self.no_agents)]
        self.self_tot_regret_with_time = [[0] for i in range(self.no_agents)]
        self.com_tot_regret_with_time = [[[0] for j in range(self.no_bandits)]  for i in range(self.no_agents)]


        self.agent_tot_reward_with_time = [[0] for i in range(self.no_agents)]
        self.self_tot_reward_with_time = [[0] for i in range(self.no_agents)]
        self.com_tot_reward_with_time = [[[0] for j in range(self.no_bandits)]  for i in range(self.no_agents)]


        #For computing the ratio F_ik
        self.self_F = [[[0 for i in range(no_iter+1)] for j in range(self.no_bandits)] for k in range(self.no_agents)]
        self.com_F =  [[[0 for i in range(no_iter+1)] for j in range(self.no_bandits)] for k in range(self.no_agents)]

        self.agents = [0 for i in range(no_agents)]
        self.agent_com_data = [(None, 0) for i in range(no_agents)]

        self.agent_Duplicate_count_with_time = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]
        self.duplicate_eliminator = [[0 for _ in range(self.no_bandits)] for __ in range(self.no_agents)]

        self.N_ij_s_T = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]
        self.N_ij_T   = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]

        self.optimal_bandit_index = optimal_bandit_index #only for regret calculation


        for i in range(no_agents):
            self.agents[i] = Agent(bandits=self.bandits, no_agents=self.no_agents, index=i, reward_variance=reward_vairance)

        for i in range(no_agents):
            self.agent_com_data[i] = 0
        self.reward_t = [0 for i in range(self.no_bandits)]

    def Sample(self):
        self.reward_t = [self.bandits[i].sample()[0] for i in range(self.no_bandits)]
        return self.reward_t

    def Pick(self):

        R = 0
        for i in range(self.no_agents):
            p = self.agents[i].pick()
            self.agent_com_data[i] = (p, self.reward_t[p])
            regret = self.reward_t[self.optimal_bandit_index] - self.reward_t[p]
            R += regret

        return self.agent_com_data

    def update(self):
        for i in range(self.no_agents):
            self.agents[i].update()

    def Communicate(self, index, itr, directed = True, ):

        self.com_count += 1
        #FOR THE MATRIX OF F
        for ag in range(self.no_agents):
            for ban in range(self.no_bandits):
                self.self_F[ag][ban][itr] = self.self_F[ag][ban][itr-1]
                self.com_F[ag][ban][itr] = self.com_F[ag][ban][itr-1]
        #

        for i in range(self.no_agents):
            self.agents[i].reset_nxt_iteration()

        R_t = [0 for i in range(self.no_agents)]
        S_R_t = [0 for i in range(self.no_agents)]
        C_R_t = [[0 for j in range(self.no_bandits)] for i in range(self.no_agents)]

        #Matrices to store rewards
        REW_t = [0 for i in range(self.no_agents)]
        S_REW_t = [0 for i in range(self.no_agents)]
        C_REW_t = [[0 for j in range(self.no_bandits)] for i in range(self.no_agents)]
        D = [[0 for _ in range(self.no_bandits)] for __ in range(self.no_agents)]

        #self communication
        for agent in range(self.no_agents):
            regret = self.reward_t[self.optimal_bandit_index] - self.agent_com_data[agent][1]
            R_t[agent] += regret
            S_R_t[agent] += regret

            REW_t[agent] += self.agent_com_data[agent][1]
            S_REW_t[agent] += self.agent_com_data[agent][1]

            x = 0
            ###
            #updating the F_ik matrics for self communication
            pick = self.agent_com_data[agent][0]
            l_t = (4/(self.delta**2))*(self.agents[agent].get_gamma()[pick])
            if l_t >= self.agents[agent].get_Nij_T()[pick]:
               self.self_F[agent][pick][itr] += 1
            ###


            self.agents[agent].communicate(agent=self.agents[agent],
                                                     data=self.agent_com_data[agent], regret = regret)

            bandit  = self.agent_com_data[agent][0]
            self.duplicate_eliminator[agent][bandit] += 1


            Choices_of_Q, Q_max = self.agents[agent].pick_for_com()

            maxi_agents  = np.argsort(Q_max)
            maxi_agents = np.flip(maxi_agents, axis=0)



            n_limit = 0
            agent_pick = [self.agent_com_data[a][0] for a in range(self.no_agents)]

            for m in maxi_agents:
                if n_limit == self.n_j:
                    break
                if self.agent_com_data[m][0] != self.agent_com_data[agent][0]:
                    n_limit += 1
                else:
                    continue

                if m == agent:
                    continue
                agent_c = m
                bandit = self.agent_com_data[agent_c][0]

                if self.duplicate_eliminator[agent][bandit] == 0:
                    x += 1
                    regret = self.reward_t[self.optimal_bandit_index] - self.agent_com_data[agent_c][1]
                    R_t[agent] += regret
                    for ban in range(self.no_bandits):
                        if ban == bandit:
                            C_R_t[agent][ban] += regret
                            C_REW_t[agent][ban] += self.agent_com_data[agent_c][1]

                    REW_t[agent] += self.agent_com_data[agent_c][1]

                    ###
                    #updating the F_ik matrics for self communication
                    pick = self.agent_com_data[agent_c][0]
                    l_t = (4/(self.delta**2))*(self.agents[agent].get_gamma()[pick])
                    #if l_t >= self.agents[agent].get_nij_T()[pick]:
                    #   self.com_F[agent][pick][itr] += 1

                    if l_t >= self.agents[agent].get_Nij_T()[pick]:
                        self.com_F[agent][pick][itr] += 1
                    ###


                    self.agents[agent].communicate(agent=self.agents[agent_c],
                                                                  data=self.agent_com_data[agent_c], regret=regret)


                    self.duplicate_eliminator[agent][bandit] += 1
                else:
                    D[agent][bandit] += 1
            """
            Q = self.agents[agent].get_Q()
            maxi  = np.argsort(Q)
            maxi = np.flip(maxi, axis=0)

            n_limit = 0

            agent_pick = [self.agent_com_data[a][0] for a in range(self.no_agents)]
            Choice_Q, Q_max = self.agents[agent].pick_for_com()
            for m in maxi:
                if n_limit == self.n_j:
                    break

                n_limit += 1
                for a in range(self.no_agents):
                    if a == agent:
                        continue
                    if m == Choice_Q[a]:
                        bandit =  m
                        if self.duplicate_eliminator[agent][bandit] == 0:
                            x += 1
                            #n_limit += 1
                            regret = self.reward_t[self.optimal_bandit_index] - self.agent_com_data[a][1]
                            R_t[agent] += regret
                            C_R_t[agent] += regret
                            REW_t += self.agent_com_data[a][1]
                            C_REW_t += self.agent_com_data[a][1]

                            ###
                            #updating the F_ik matrics for self communication
                            pick = self.agent_com_data[a][0]
                            l_t = (4/(self.delta**2))*(self.agents[agent].get_gamma()[pick])
                            if l_t >= self.agents[agent].get_Nij_T()[pick]:
                                self.com_F[agent][pick][itr] += 1
                            ###


                            self.agents[agent].communicate(agent=self.agents[a],
                                                                  data=self.agent_com_data[a], regret=regret)

                            self.duplicate_eliminator[agent][bandit] += 1
                        else:
                            D[agent][bandit] += 1
        """
        Nijk_T = self.get_Nijk_T_for_all_Agents()
        Nij_T = self.get_Nij_T_for_all_agents()

        for i in range(self.no_agents):
            self.agent_tot_regret_with_time[i].append(self.agent_tot_regret_with_time[i][-1] + R_t[i])
            self.self_tot_regret_with_time[i].append(self.self_tot_regret_with_time[i][-1] + S_R_t[i])
            for ban in range(self.no_bandits):
                self.com_tot_reward_with_time[i][ban].append(self.com_tot_reward_with_time[i][ban][-1] + C_REW_t[i][ban])
                self.com_tot_regret_with_time[i][ban].append(self.com_tot_regret_with_time[i][ban][-1] + C_R_t[i][ban])

            self.agent_tot_reward_with_time[i].append(self.agent_tot_reward_with_time[i][-1] + REW_t[i])
            self.self_tot_reward_with_time[i].append(self.self_tot_reward_with_time[i][-1] + S_REW_t[i])


            for j in range(self.no_bandits):
                self.N_ij_s_T[i][j].append(Nijk_T[i][i][j])
                self.N_ij_T[i][j].append(Nij_T[i][j])
            '''
            for j in range(self.no_bandits):
                self.agent_Duplicate_count_with_time[i][j].append(self.agent_Duplicate_count_with_time[i][j][-1] + D[i][j])
            '''
        self.update()
        self.duplicate_eliminator = [[0 for _ in range(self.no_bandits)] for __ in range(self.no_agents)]



    def get_agent_tot_regret_with_time(self):
        return self.agent_tot_regret_with_time

    def get_agent_self_tot_regret_with_time(self):
        return self.self_tot_regret_with_time

    def get_agent_com_tot_regret_with_time(self):
        return self.com_tot_regret_with_time

    def get_agent_duplicate_eliminator(self):
        return self.agent_Duplicate_count_with_time

    def get_agent_tot_reward_with_time(self):
        return self.agent_tot_reward_with_time

    def get_agent_self_tot_reward_with_time(self):
        return self.self_tot_reward_with_time

    def get_agent_com_tot_reward_with_time(self):
        return self.com_tot_reward_with_time


    def get_self_F(self):
        return self.self_F

    def get_com_F(self):
        return self.com_F

    #internal function
    def get_Nijk_T_for_all_Agents(self):
        Nijk_T = []
        for a in self.agents:
            Nijk_T.append(a.get_N_ijk_T())

        return  Nijk_T
    #internal function
    def get_Nij_T_for_all_agents(self):

        Nij_T = []
        for a in self.agents:
            Nij_T.append(a.get_Nij_T())

        return Nij_T

    def get_Nij_T_s(self):
        return self.N_ij_s_T

    def get_Nij_T(self):
        return self.N_ij_T
