import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from Agent import Agent


class MAMAB():

    def __init__(self, no_bandits, no_agents, bandits, Sets, Ban_to_Set_dict, optimal_bandit_index, p, reward_vairance, no_iter, delta):
        self.no_bandits = no_bandits
        self.no_agents  = no_agents
        self.bandits = bandits
        self.delta = delta
        self.p = p
        self.Sets = Sets                          #list of all the sets
        self.Ban_to_Set_dic = Ban_to_Set_dict     # a dic mapping bandit to sets it is in

        self.com_count = 0

        self.agent_tot_regret_with_time = [[0] for i in range(self.no_agents)]
        self.self_tot_regret_with_time = [[0] for i in range(self.no_agents)]
        self.com_tot_regret_with_time = [[0] for i in range(self.no_agents)]

        self.agent_tot_reward_with_time = [[0] for i in range(self.no_agents)]
        self.self_tot_reward_with_time = [[0] for i in range(self.no_agents)]
        self.com_tot_reward_with_time = [[0] for i in range(self.no_agents)]

        #self.New_Com_Metric = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]

        self.choices = [[] for i in range(self.no_agents)]

        self.global_tot_regret = [0] #regret for the machine (accumulated)


        self.agents = [0 for i in range(no_agents)]
        self.agent_com_data = [(None, 0) for i in range(no_agents)]

        self.N_ij_s_T = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]
        self.N_ij_T   = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]


        #For computing the ratio F_ik
        self.self_F = [[[0 for i in range(no_iter+1)] for j in range(self.no_bandits)] for k in range(self.no_agents)]
        self.com_F =  [[[0 for i in range(no_iter+1)] for j in range(self.no_bandits)] for k in range(self.no_agents)]

        self.agent_Duplicate_count_with_time = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]
        self.duplicate_eliminator = [[0 for _ in range(self.no_bandits)] for __ in range(self.no_agents)]

        self.optimal_bandit_index = optimal_bandit_index #only for regret calculation


        self.estimate = [[[0] for i in range(self.no_bandits)] for j in range(self.no_agents)]

        for i in range(no_agents):
            #self.agents[i] = Agent(bandits=self.bandits, no_agents=self.no_agents, index=i,
            #                     reward_variance=reward_vairance, curr_set=0)
            self.agents[i] = Agent(bandits=self.bandits, no_agents=self.no_agents, index=i,
                                  reward_variance=reward_vairance, curr_set=np.random.randint(0,len(Sets)))
        for i in range(no_agents):
            self.agent_com_data[i] = 0


        self.reward_t = [0 for i in range(self.no_bandits)]
        #self.initial_Communicate()
        #print(self.agents[i].X_ij_T)
    def Sample(self):

        self.reward_t = [self.bandits[i].sample()[0] for i in range(self.no_bandits)]

        return self.reward_t

    def Pick(self):

        R = 0

        for i in range(self.no_agents):
            p = self.agents[i].pick(current_set=self.agents[i].current_set)
            set_index = self.agents[i].current_set

            if self.Sets[set_index].is_ban_in(p):

                self.agents[i].current_set = p #to get the moving in maze strat
                self.agent_com_data[i] = (p, self.reward_t[p])
                regret = self.reward_t[self.optimal_bandit_index] - self.reward_t[p]
                R += regret

                #for animation
                self.choices[i].append([p,self.agents[i].current_set])

            else:
                nxt_set = self.Sets[set_index].get_ban_set(bandit_index=p, Global_Set=self.Sets,
                                                 ban_to_set_dict=self.Ban_to_Set_dic)

                self.agents[i].current_set = nxt_set.index
                self.agent_com_data[i] = (None, 0)

                #for animation

                self.choices[i].append([None ,self.agents[i].current_set])

        self.global_tot_regret.append(self.global_tot_regret[-1] + R)

        return self.agent_com_data

    def update(self):
        for i in range(self.no_agents):
            self.agents[i].update()

    def initial_Communicate(self):
        #INITIAL SAMPLING FOR EVEYONE
        self.reward_t = [self.bandits[i].sample()[0] for i in range(self.no_bandits)]

        G = nx.erdos_renyi_graph(self.no_agents,
                                 self.p,
                                 directed=True)

        for ban in range(self. no_bandits):
            for agent in G.nodes:
                self.agents[agent].reset_nxt_iteration()
                regret = self.reward_t[self.optimal_bandit_index] - self.reward_t[ban]
                R_t = [0 for i in range(self.no_agents)]
                S_R_t = [0 for i in range(self.no_agents)]

                #Matrices to store rewards
                REW_t = [0 for i in range(self.no_agents)]
                S_REW_t = [0 for i in range(self.no_agents)]
                R_t[agent] += regret
                S_R_t[agent] += regret

                REW_t[agent] += self.reward_t[ban]
                S_REW_t[agent] += self.reward_t[ban]

                self.agents[agent].communicate(agent=self.agents[agent],
                                    data=(ban, self.reward_t[ban]), regret = regret, inital = True
                                               )
                #self.agent_tot_regret_with_time[agent].append(self.agent_tot_regret_with_time[i][-1] + R_t[i])
                #self.self_tot_regret_with_time[agent].append(self.self_tot_regret_with_time[i][-1] + S_R_t[i])
                #self.agent_tot_reward_with_time[agent].append(self.agent_tot_reward_with_time[i][-1] + REW_t[i])
                #self.self_tot_reward_with_time[agent].append(self.self_tot_reward_with_time[i][-1] + S_REW_t[i])


            self.update()

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
        C_R_t = [0 for i in range(self.no_agents)]
        #N_C_R_t = [[0 for i in range(self.no_bandits)] for j in range(self.no_agents)]
        #Arrays to store rewards
        REW_t = [0 for i in range(self.no_agents)]
        S_REW_t = [0 for i in range(self.no_agents)]
        C_REW_t = [0 for i in range(self.no_agents)]

        D = [[0 for _ in range(self.no_bandits)] for __ in range(self.no_agents)]

        #self communication

        for node in G.nodes:

            if self.agent_com_data[node][0] == None:
                pass
            else:
                regret = self.reward_t[self.optimal_bandit_index] - self.agent_com_data[node][1]
                R_t[node] += regret
                S_R_t[node] += regret

                REW_t[node] += self.agent_com_data[node][1]
                S_REW_t[node] += self.agent_com_data[node][1]

                ###
                #updating the F_ik matrics for self communication
                pick = self.agent_com_data[node][0]
                l_t = (4/(self.delta**2))*(self.agents[node].get_gamma()[pick])

                if l_t >= self.agents[node].get_nij_T()[pick]:
                    self.self_F[node][pick][itr] += 1

                ###

                self.agents[node].communicate(agent=self.agents[node],
                                                     data=self.agent_com_data[node], regret = regret)

                self.duplicate_eliminator[node][self.agent_com_data[node][0]] += 1



        if directed == True:
            for edge in G.edges:
                if self.agent_com_data[edge[1]][0] != None:
                    bandit = self.agent_com_data[edge[1]][0]

                    self.agents[edge[0]].communicate_for_com_estimate(agent=self.agents[edge[1]],
                                                              data=self.agent_com_data[edge[1]])

                    if self.duplicate_eliminator[edge[0]][bandit] == 0:

                        regret = self.reward_t[self.optimal_bandit_index] - self.agent_com_data[edge[1]][1]
                        R_t[edge[0]] += regret
                        C_R_t[edge[0]] += regret
                        #N_C_R_t[edge[0]][bandit] += 1

                        REW_t += self.agent_com_data[edge[1]][1]
                        C_REW_t += self.agent_com_data[edge[1]][1]

                        ###
                        #updating the F_ik matrics for self communication
                        pick = self.agent_com_data[edge[1]][0]
                        l_t = (4/(self.delta**2))*(self.agents[edge[0]].get_gamma()[pick])
                        if l_t >= self.agents[edge[0]].get_nij_T()[pick]:
                            self.com_F[edge[0]][pick][itr] += 1
                        ###


                        self.agents[edge[0]].communicate(agent=self.agents[edge[1]],
                                                              data=self.agent_com_data[edge[1]], regret=regret)


                        self.duplicate_eliminator[edge[0]][bandit] += 1

                    else:
                        D[edge[0]][bandit] += 1


        Nijk_T = self.get_Nijk_T_c_for_all_Agents()
        Nij_T = self.get_Nij_T_for_all_agents()


        for i in range(self.no_agents):
            self.agent_tot_regret_with_time[i].append(self.agent_tot_regret_with_time[i][-1] + R_t[i])
            self.self_tot_regret_with_time[i].append(self.self_tot_regret_with_time[i][-1] + S_R_t[i])

            self.com_tot_regret_with_time[i].append(self.com_tot_regret_with_time[i][-1] + C_R_t[i])

            self.agent_tot_reward_with_time[i].append(self.agent_tot_reward_with_time[i][-1] + REW_t[i])
            self.self_tot_reward_with_time[i].append(self.self_tot_reward_with_time[i][-1] + S_REW_t[i])
            self.com_tot_reward_with_time[i].append(self.com_tot_reward_with_time[i][-1] + C_REW_t[i])

            #for j in range(self.no_bandits):
            #    self.agent_Duplicate_count_with_time[i][j].append(self.agent_Duplicate_count_with_time[i][j][-1] + D[i][j])
            #for j in range(self.no_bandits):
            #    self.New_Com_Metric[i][j].append(self.New_Com_Metric[i][j][-1] + N_C_R_t[i][j])

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

    def get_choices(self):
        return self.choices

    #def get_New_Com_Metric(self):
    #    return self.New_Com_Metric

    #internal function
    def get_Nijk_T_for_all_Agents(self):
        Nijk_T = []
        for a in self.agents:
            Nijk_T.append(a.get_N_ijk_T())

        return  Nijk_T

    def get_Nijk_T_c_for_all_Agents(self):
        Nijk_T_c = []
        for a in self.agents:
            Nijk_T_c.append(a.get_N_ijk_T_c())

        return  Nijk_T_c


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
