import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from Agent import Agent


class MAMAB():

    def __init__(self, no_bandits, no_agents, bandits, optimal_bandit_index, p, reward_vairance, no_iter, delta):
        self.no_bandits = no_bandits
        self.no_agents  = no_agents
        self.bandits = bandits
        self.delta = delta
        self.p = p

        self.com_count = 0

        self.agent_tot_regret_with_time = [[0] for i in range(self.no_agents)]
        self.self_tot_regret_with_time = [[0] for i in range(self.no_agents)]
        self.com_tot_regret_with_time = [[[0] for j in range(self.no_bandits)]  for i in range(self.no_agents)]

        self.agent_tot_reward_with_time = [[0] for i in range(self.no_agents)]
        self.self_tot_reward_with_time = [[0] for i in range(self.no_agents)]
        self.com_tot_reward_with_time = [[[0] for j in range(self.no_bandits)]  for i in range(self.no_agents)]

        self.global_tot_regret = [0] #regret for the machine (accumulated)


        self.agents = [0 for i in range(no_agents)]
        self.agent_com_data = [(None, 0) for i in range(no_agents)]


        #For computing the ratio F_ik
        self.self_F = [[[0 for i in range(no_iter+1)] for j in range(self.no_bandits)] for k in range(self.no_agents)]
        self.com_F =  [[[0 for i in range(no_iter+1)] for j in range(self.no_bandits)] for k in range(self.no_agents)]

        self.agent_Duplicate_count_with_time = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]
        self.duplicate_eliminator = [[0 for _ in range(self.no_bandits)] for __ in range(self.no_agents)]

        self.optimal_bandit_index = optimal_bandit_index #only for regret calculation

        self.N_ij_s_T = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]
        self.N_ij_T   = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]

        self.estimate = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]

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
        self.global_tot_regret.append(self.global_tot_regret[-1] + R)

        return self.agent_com_data

    def update(self):
        for i in range(self.no_agents):
            self.agents[i].update()

    def Communicate(self, itr, index,directed = True):

        self.com_count += 1


        G = nx.erdos_renyi_graph(self.no_agents,
                                 self.p,
                                 directed=directed)

        #FOR THE MATRIX OF F
        for ag in range(self.no_agents):
            for ban in range(self.no_bandits):
                self.self_F[ag][ban][itr] = self.self_F[ag][ban][itr-1]
                self.com_F[ag][ban][itr] = self.com_F[ag][ban][itr-1]
        #


        agent_pick = [self.agent_com_data[a][0] for a in range(self.no_agents)]


        for i in range(self.no_agents):
            self.agents[i].reset_nxt_iteration()

        R_t = [0 for i in range(self.no_agents)]
        S_R_t = [0 for i in range(self.no_agents)]
        C_R_t = [[0 for j in range(self.no_bandits)] for i in range(self.no_agents)]

        #Arrays to store rewards
        REW_t = [0 for i in range(self.no_agents)]
        S_REW_t = [0 for i in range(self.no_agents)]
        C_REW_t = [[0 for j in range(self.no_bandits)] for i in range(self.no_agents)]

        D = [[0 for _ in range(self.no_bandits)] for __ in range(self.no_agents)]

        #self communication

        for node in G.nodes:
            regret = self.reward_t[self.optimal_bandit_index] - self.agent_com_data[node][1]
            R_t[node] += regret
            S_R_t[node] += regret

            REW_t[node] += self.agent_com_data[node][1]
            S_REW_t[node] += self.agent_com_data[node][1]

            ###
            #updating the F_ik matrics for self communication
            pick = self.agent_com_data[node][0]
            l_t = (4/(self.delta**2))*(self.agents[node].get_gamma()[pick])
            if l_t >= self.agents[node].get_Nij_T()[pick]:
                self.self_F[node][pick][itr] += 1
            ###

            self.agents[node].communicate(agent=self.agents[node],
                                                     data=self.agent_com_data[node], regret = regret)

            self.duplicate_eliminator[node][self.agent_com_data[node][0]] += 1



        if directed == True:
            for edge in G.edges:

                bandit = self.agent_com_data[edge[1]][0]

                if self.duplicate_eliminator[edge[0]][bandit] == 0:

                    regret = self.reward_t[self.optimal_bandit_index] - self.agent_com_data[edge[1]][1]
                    R_t[edge[0]] += regret
                    for ban in range(self.no_bandits):
                        if ban == bandit:
                            C_R_t[edge[0]][ban] += regret
                            C_REW_t[edge[0]][ban] += self.agent_com_data[edge[1]][1]
                    REW_t[edge[0]] += self.agent_com_data[edge[1]][1]


                    ###
                    #updating the F_ik matrics for self communication
                    pick = self.agent_com_data[edge[1]][0]
                    l_t = (4/(self.delta**2))*(self.agents[edge[0]].get_gamma()[pick])
                    if l_t >= self.agents[edge[0]].get_Nij_T()[pick]:
                        self.com_F[edge[0]][pick][itr] += 1
                    ###


                    self.agents[edge[0]].communicate(agent=self.agents[edge[1]],
                                                          data=self.agent_com_data[edge[1]], regret=regret)


                    self.duplicate_eliminator[edge[0]][bandit] += 1

                else:
                    D[edge[0]][bandit] += 1


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


            '''
            for j in range(self.no_bandits):
                self.agent_Duplicate_count_with_time[i][j].append(self.agent_Duplicate_count_with_time[i][j][-1] + D[i][j])
            '''
            for j in range(self.no_bandits):
                self.N_ij_s_T[i][j].append(Nijk_T[i][i][j])
                self.N_ij_T[i][j].append(Nij_T[i][j])

        self.update()
        self.duplicate_eliminator = [[0 for _ in range(self.no_bandits)] for __ in range(self.no_agents)]

        for a in range(self.no_agents):
            est = self.agents[a].get_estimate()
            for b in range(self.no_bandits):
                self.estimate[a][b].append(est[b])




    def get_global_tot_regret(self):
        return self.global_tot_regret

    def get_agent_tot_regret_with_time(self):
        return self.agent_tot_regret_with_time


    def get_agent_self_tot_regret_with_time(self):
        return self.self_tot_regret_with_time

    def get_agent_com_tot_regret_with_time(self):
        return self.com_tot_regret_with_time

    def get_agent_tot_reward_with_time(self):
        return self.agent_tot_reward_with_time

    def get_agent_self_tot_reward_with_time(self):
        return self.self_tot_reward_with_time

    def get_agent_com_tot_reward_with_time(self):
        return self.com_tot_reward_with_time

    def get_agent_duplicate_eliminator(self):
        return self.agent_Duplicate_count_with_time

    def get_self_F(self):
        return self.self_F

    def get_com_F(self):
        return self.com_F

    def get_estimate(self):
        return self.estimate

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
