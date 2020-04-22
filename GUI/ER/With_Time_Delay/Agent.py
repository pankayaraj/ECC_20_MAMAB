#from Gaussian_Reward import Gaussian_Reward
from math import sqrt, log
from numpy import argmax, array, argsort, random, sum


def euclidian_dist(c1, c2):
    return sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

class Agent():

    def __init__(self, index, bandits, no_agents, reward_variance, curr_set):


        self.T = 1
        self.Regret = 0

        self.current_set = curr_set

        self.nxt_ban_in = None

        self.index = index
        self.bandits = bandits
        self.no_agents = no_agents
        self.no_bandits = len(self.bandits)
        self.reward_variance = reward_variance

        self.X_ij_support = [[random.normal(0, 1)*(1/self.no_agents)*12 for i in range(self.no_bandits)] for j in range(self.no_agents)] #support in the update of the mean (summation of the last term )
        self.X_ij_T = sum(self.X_ij_support, axis=0 ) #apprximated mean upto time T

        self.Q = [0 for i in range(self.no_bandits)]

        self.N_ij_t = [set() for i in range(len(self.bandits))]          #set of agents at that has communicated to j and said it has picked i at time t
        self.N_ij_T = [set() for i in range(len(self.bandits))]          #set of agents that has communicated to j and said it has picked i upto time T
        self.nij_T = [1 for i in range(self.no_bandits)]                  # no of agents that have communicated to j and said it has picked i upto time T



        self.Nij_T = [1 for i in range(self.no_bandits)]  # no of agents that have communicated to j and said it has picked i upto time T


        self.Nijk_t = [[0 for i in range(self.no_bandits)] for  j in range(self.no_agents)]   #no of times k has communicated with j and said that it has picked arm i at time t
        self.Nijk_T = [[1 for i in range(self.no_bandits)] for  j in range(self.no_agents)]   #no of times k has communicated with j and said that it has picked arm i upto time t

        self.Nijk_t_c = [[0 for i in range(self.no_bandits)] for  j in range(self.no_agents)]   #no of times k has communicated with j and said that it has picked arm i at time t
        self.Nijk_T_c = [[1 for i in range(self.no_bandits)] for  j in range(self.no_agents)]


        #parameters in computation

        self.k = [ (1/((8*self.reward_variance[i])**2)*2) for i in range(self.no_bandits)]


        self.reward_variance = reward_variance

    def reset_nxt_iteration(self):
        self.N_ij_t = [set() for i in range(len(self.bandits))]
        self.Nijk_t = [[0 for i in range(self.no_bandits)] for  j in range(self.no_agents)]

        self.T += 1

    def sample(self, bandit, size):
        return self.bandits[bandit].sample(size)





        #print(self.Regret)
    def communicate(self, agent, data, regret):
        #this is not done in case of duplicates

        bandit = data[0]
        reward = data[1]

        self.N_ij_t[bandit].add(agent.index)
        self.N_ij_T[bandit].add(agent.index)

        self.nij_T[bandit] = len(self.N_ij_T[bandit])

        self.Nijk_t_c[agent.index][bandit] += 1
        self.Nijk_T_c[agent.index][bandit] += 1

        self.Nij_T[bandit] += 1
        self.X_ij_support[agent.index][bandit] += reward

    def communicate_for_com_estimate(self, agent, data):

        #this is done regardless of the duplicates
        bandit = data[0]
        reward = data[1]

        self.Nijk_t[agent.index][bandit] += 1
        self.Nijk_T[agent.index][bandit] += 1


    def estimate(self):

        estimate = self.X_ij_T
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
    def pick(self, current_set):

        Qij_T = [0 for i in range(self.no_bandits)]

        for i in range(self.no_bandits):

            ban_coordinate = (i//10, i%10)
            set_coordinate = (current_set//10, current_set%10)

            d_i = euclidian_dist(ban_coordinate, set_coordinate) #distance btw ban and the current one
            t_p = 0.001 #tuning parameter
            tau = sqrt(200) #the maximum dist between any two nodes

            dist_penalty = ((1 + t_p*tau)/(1 + t_p*d_i))

            k = self.k[i]
            alpha = 3/(2*k)
            gamma = alpha*log(self.T)
            Qij_T[i] = self.X_ij_T[i] + sqrt(dist_penalty*gamma/self.Nij_T[i])

        self.Q = Qij_T
        sorted_choice = argsort(Qij_T)


        m = sorted_choice[-1]
        m_c = 1
        for i in range(1, self.no_bandits):

            if Qij_T[sorted_choice[-(i+1)]] != Qij_T[m]:
                break
            else:
                m_c += 1

        index = 1
        if m_c == 1:
            if self.nxt_ban_in != None:
                if Qij_T[self.nxt_ban_in] == Qij_T[sorted_choice[-1]]:
                    choice = self.nxt_ban_in
                    return choice
                else:
                    index = 1
            else:
                index = 1
        else:
            if self.nxt_ban_in != None:
                if Qij_T[self.nxt_ban_in] == Qij_T[sorted_choice[-index]]:
                    choice = self.nxt_ban_in
                    return choice

                else:
                    index = random.randint(1, m_c)
            else:
                index = random.randint(1, m_c)

        choice = sorted_choice[-index]
        self.nxt_ban_in = choice

        return choice


    def get_regret(self):
        return self.Regret

    def update(self):
        for i in range(self.no_bandits):

            term = 0

            N = self.Nij_T[i]
            for k in self.N_ij_T[i]:
                term += self.X_ij_support[k][i]



            self.X_ij_T[i] = term/N #/(self.nij_T[i]*sqrt(self.Nij_T[i]))
            '''
            for k in self.N_ij_T[i]:
                term += self.X_ij_support[k][i]/sqrt(self.Nijk_T[k][i])


            self.X_ij_T[i] = term/(self.nij_T[i]*sqrt(self.Nij_T[i]))
            '''
        #print(self.X_ij_T)



    def get_estimate(self):
        return self.X_ij_T

    def get_Q(self):
        return self.Q

    def get_N_ijk_T(self):
        return  self.Nijk_T

    def get_N_ijk_T_c(self):
        return  self.Nijk_T_c

    def get_Nij_T(self):
        return self.Nij_T
