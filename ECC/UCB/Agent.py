from Gaussian_Reward import Gaussian_Reward
from math import sqrt, log
from numpy import argmax, array, argsort, random, sum
import numpy as np


class Agent():

    def __init__(self, index, bandits, no_agents, reward_variance):


        self.T = 1
        self.Regret = 0


        self.index = index
        self.bandits = bandits
        self.no_agents = no_agents
        self.no_bandits = len(self.bandits)
        self.reward_variance = reward_variance

        #for calculating
        self.X_ijk_T_support = np.array([[random.normal(0, 1)*(1/self.no_agents)*12 for i in range(self.no_bandits)] for j in range(self.no_agents)])


        self.X_ij_support = [[random.normal(0, 1)*(1/self.no_agents)*12 for i in range(self.no_bandits)] for j in range(self.no_agents)] #support in the update of the mean (summation of the last term )
        self.X_ij_T = sum(self.X_ij_support, axis=0 ) #apprximated mean upto time T

        self.Q = [0 for i in range(self.no_bandits)]
        self.Q_com = [[0 for i in range(self.no_bandits)] for j in range(self.no_agents)]

        self.N_ij_t = [set() for i in range(len(self.bandits))]          #set of agents at that has communicated to j and said it has picked i at time t
        self.N_ij_T = [set() for i in range(len(self.bandits))]          #set of agents that has communicated to j and said it has picked i upto time T
        self.nij_T = [1 for i in range(self.no_bandits)]                  # no of agents that have communicated to j and said it has picked i upto time T

        self.Nij_T = [1 for i in range(self.no_bandits)]   #no of times j has known that i has been picked in the time horizon T

        self.Nijk_t = [[0 for i in range(self.no_bandits)] for  j in range(self.no_agents)]   #no of times k has communicated with j and said that it has picked arm i at time t
        self.Nijk_T = [[1 for i in range(self.no_bandits)] for  j in range(self.no_agents)]   #no of times k has communicated with j and said that it has picked arm i upto time t

        self.Njk_C_T = [1 for i in range(no_agents)] #how many time j have communicated to k

        self.X_ijk_T         = np.array(self.X_ijk_T_support)/np.array(self.Nijk_T)
        #parameters in computation

        self.k = [ (1/((8*self.reward_variance[i])**2)*2) for i in range(self.no_bandits)]


        self.reward_variance = reward_variance

    def reset_nxt_iteration(self):
        self.N_ij_t = [set() for i in range(len(self.bandits))]
        self.Nijk_t = [[0 for i in range(self.no_bandits)] for  j in range(self.no_agents)]

        self.T += 1

    def sample(self, bandit, size):
        return self.bandits[bandit].sample(size)



    def communicate(self, agent, data, regret):

        bandit = data[0]
        reward = data[1]

        self.Njk_C_T[agent.index] += 1

        self.N_ij_t[bandit].add(agent.index)
        self.N_ij_T[bandit].add(agent.index)

        self.nij_T[bandit] = len(self.N_ij_T[bandit])

        self.Nij_T[bandit] += 1

        self.Nijk_t[agent.index][bandit] += 1
        self.Nijk_T[agent.index][bandit] += 1


        self.X_ijk_T_support[agent.index][bandit] += reward
        self.X_ij_support[agent.index][bandit] += reward
        self.Regret += regret
        #print(self.Regret)

    def estimate(self):

        estimate = self.X_ijk_T
        return estimate


    def get_nij_T(self):
        return self.nij_T


    def get_gamma(self):
        GAMMA = [0 for i in range(self.no_bandits)]

        for i in range(self.no_bandits):
            k = self.k[i]
            alpha = 3/(2*k)
            gamma = alpha*log(self.T)
            GAMMA[i] = gamma

        return GAMMA

    #pick for self Q quation - 6
    def pick(self):

        Qij_T = [0 for i in range(self.no_bandits)]
        for i in range(self.no_bandits):
            k = self.k[i]
            alpha = 3/(2*k)
            gamma = alpha*log(self.T)
            Qij_T[i] = self.X_ij_T[i] + sqrt(gamma/self.Nij_T[i])
        self.Q = Qij_T
        sorted_choice = argsort(Qij_T)

        m = sorted_choice[-1]
        m_c = 1
        for i in range(1, self.no_bandits):

            if Qij_T[sorted_choice[-(i+1)]] != Qij_T[m]:
                break
            else:
                m_c += 1

        if m_c == 1:
            index = 1
        else:
            index = random.randint(1, m_c)
        choice = sorted_choice[-index]
        return choice

    #Q for other agents
    def pick_for_com(self):
        Qijk_T = [[0 for i in range(self.no_bandits)] for j in range(self.no_agents)]

        for j in range(self.no_agents):
            for i in range(self.no_bandits):
                k = self.k[i]
                alpha = 3/(2*k)
                gamma = alpha*log(self.T)
                #Qijk_T[j][i] = self.X_ijk_T[j][i] + sqrt(gamma/self.Njk_C_T[j])
                Qijk_T[j][i] = self.X_ijk_T[j][i] + sqrt(gamma/self.Nijk_T[j][i])
                #Qijk_T[j][i] = sqrt(gamma/self.Nijk_T[j][i])
        self.Q_com = Qijk_T
        sorted_choice = argsort(Qijk_T, axis=1)

        m = [sorted_choice[i][-1] for i in range(self.no_agents)]
        choice = []
        q_max = []
        for a in range(self.no_agents):
            m_c = 1
            for i in range(1, self.no_bandits):
                if Qijk_T[a][sorted_choice[a][-(i+1)]] != Qijk_T[a][m[a]]:
                    break
                else:
                    m_c += 1

            if m_c == 1:
                index = 1
            else:
                index = random.randint(1, m_c)
            choice.append(sorted_choice[a][-index])
            q_max.append(Qijk_T[a][sorted_choice[a][-index]])

        return choice, q_max

    def get_regret(self):
        return self.Regret

    def update(self):
        for i in range(self.no_bandits):

            term = 0

            N = self.Nij_T[i]
            for k in self.N_ij_T[i]:
                term += self.X_ij_support[k][i]

            self.X_ij_T[i] = term/N #/(self.nij_T[i]*sqrt(self.Nij_T[i]))

        self.X_ijk_T = np.array(self.X_ijk_T_support)/np.array(self.Nijk_T)

    def get_estimate(self):
        return self.X_ij_T

    def get_Q(self):
        return self.Q

    def get_Q_com(self):
        return self.Q_com

    def get_N_ijk_T(self):
        return  self.Nijk_T

    def get_Nij_T(self):
        return self.Nij_T


