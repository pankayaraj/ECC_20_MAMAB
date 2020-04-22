from Agent import Agent
from Gaussian_Reward import Gaussian_Reward
from max_q import MAMAB
import numpy as np
from math import sqrt, log
import networkx as nx
from matplotlib import pyplot as plt


no_iterations = 20000

no_agents = 20
no_bandits = 100

#mean = [5.2, 4.1, 9.5, 2.4, 5.9, 6.9, 7.4, 0.5, 4.7, 2.1, 10.5, 1.5, 2.8, 8.8, 3.7, 4.4, 7.8, 3.0, 11.9, 8.3]
#print(len(mean))
#variance = [2 for i in range(no_bandits)]

'''
mean = [np.random.random()*12 for i in range(no_bandits) ]
variance = [2 for i in range(no_bandits)]
'''
mean = np.load("Mean.npy")
variance = np.load("Variance.npy")

np.save("Agent/Mean.npy", np.array(mean))
np.save("Agent/Variance.npy", np.array(variance))
o = np.argmax(np.array(mean))
print(mean)
bandits = [Gaussian_Reward(mean[i], variance[i]) for i in range(no_bandits)]

#COMPUTE DELTA
maxi_mean = np.argsort(mean)
maxi_mean = np.flip(maxi_mean, axis=0)
max_index = maxi_mean[0]
print(maxi_mean)
for i in range(1, len(maxi_mean)):
    if mean[maxi_mean[i]] != mean[maxi_mean[0]]:
        sec_max_index = maxi_mean[i]
        break

delta = mean[max_index]-mean[sec_max_index]
print(delta)
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
#for no in range(0, no_agents, int(no_agents/10)):
nj =[8, 16, 18]
for no in nj:
#for no in range(0, no_agents+1):
        GLOBAL_TOT_REGRET = np.array([0 for _ in range(no_iterations+1)])
        AGENT_TOT_REGRET = np.array([[0 for _ in range(no_iterations+1)] for i in range(no_agents)])
        SELF_TOT_REGRET = np.array([[0 for _ in range(no_iterations+1)] for __ in range(no_agents)])
        COM_TOT_REGRET = np.array([[[0 for _ in range(no_iterations+1)] for __ in range(no_bandits)] for ___ in range(no_agents)])

        AGENT_TOT_REWARD = np.array([[0 for _ in range(no_iterations+1)] for i in range(no_agents)])
        SELF_TOT_REWARD = np.array([[0 for _ in range(no_iterations+1)] for __ in range(no_agents)])
        COM_TOT_REWARD = np.array([[[0 for _ in range(no_iterations+1)] for __ in range(no_bandits)] for ___ in range(no_agents)])

        SELF_F =  [[[0 for _ in range(no_iterations+1)] for j in range(no_bandits)] for k in range(no_agents)]
        COM_F  =  [[[0 for _ in range(no_iterations+1)] for j in range(no_bandits)] for k in range(no_agents)]

        DUPLICATE = np.array([[[0 for _ in range(no_iterations+1)] for i in range(no_bandits)] for j in range(no_agents)])
        Nij_T_s_MAIN = [[[0 for _ in range(no_iterations+1)] for j in range(no_bandits)] for k in range(no_agents)]
        Nij_T_MAIN = [[[0 for _ in range(no_iterations+1)] for j in range(no_bandits)] for k in range(no_agents)]

        for experiment  in range(1, 5):
            print("Experiment " + str(experiment) + "for no " + str(no))
            G = MAMAB(no_bandits=no_bandits, no_agents=no_agents, bandits=bandits,optimal_bandit_index=o , reward_vairance=variance, delta=delta,
                                                                                                                    no_iter=no_iterations, n_j=no)

            for i in range(no_iterations):
                print(i)
                G.Sample()
                a = G.Pick()
                G.Communicate(index=i, itr=i)


            agent_tot_regret_T = G.get_agent_tot_regret_with_time()
            self_tot_regret = G.get_agent_self_tot_regret_with_time()
            com_tot_reget = G.get_agent_com_tot_regret_with_time()

            agent_tot_reward_T = G.get_agent_tot_reward_with_time()
            self_tot_reward = G.get_agent_self_tot_reward_with_time()
            com_tot_reward = G.get_agent_com_tot_reward_with_time()

            self_F = G.get_self_F()
            com_F  = G.get_com_F()


            duplicate = G.get_agent_duplicate_eliminator()

            Nij_T_s = G.get_Nij_T_s()
            Nij_T   = G.get_Nij_T()



            AGENT_TOT_REGRET = np.add(AGENT_TOT_REGRET, agent_tot_regret_T)
            SELF_TOT_REGRET = np.add(SELF_TOT_REGRET, self_tot_regret)
            COM_TOT_REGRET = np.add(COM_TOT_REGRET, com_tot_reget)

            AGENT_TOT_REWARD = np.add(AGENT_TOT_REWARD, agent_tot_reward_T)
            SELF_TOT_REWARD  = np.add(SELF_TOT_REWARD, self_tot_reward)
            COM_TOT_REWARD   = np.add(COM_TOT_REWARD, com_tot_reward)

            Nij_T_MAIN  = np.add(Nij_T_MAIN, Nij_T)
            Nij_T_s_MAIN = np.add(Nij_T_s_MAIN, Nij_T_s)

            SELF_F  =  np.add(SELF_F, self_F)
            COM_F   =  np.add(COM_F, com_F)

            DUPLICATE = np.add(DUPLICATE, duplicate)


        AGENT_TOT_REGRET = (1/experiment)*AGENT_TOT_REGRET
        SELF_TOT_REGRET = (1/experiment)*SELF_TOT_REGRET
        COM_TOT_REGRET = (1/experiment)*COM_TOT_REGRET

        AGENT_TOT_REWARD = (1/experiment)*AGENT_TOT_REWARD
        SELF_TOT_REWARD = (1/experiment)*SELF_TOT_REWARD
        COM_TOT_REWARD = (1/experiment)*COM_TOT_REWARD

        SELF_F = (1/experiment)*SELF_F
        COM_F  = (1/experiment)*COM_F

        GLOBAL_TOT_REGRET = (1/experiment)*GLOBAL_TOT_REGRET
        DUPLICATE = (1/experiment)*DUPLICATE


        Nij_t_MAIN = (1/experiment)*Nij_T_MAIN
        Nij_T_s_MAIN = (1/experiment)*Nij_T_s_MAIN


        np.save("Agent/Total/Agent_tot_regret" + str(no), AGENT_TOT_REGRET)
        np.save("Agent/Total/Agent_tot_reward" + str(no), AGENT_TOT_REWARD)

        #np.save("Agent/Total/Duplicate" + str(no), DUPLICATE)

        np.save("Self/Total/Self_tot_regret" + str(no), SELF_TOT_REGRET)
        np.save("Self/Total/Self_tot_reward" + str(no), SELF_TOT_REWARD)
        np.save("Self/Total/Self_F" + str(no), SELF_F)

        np.save("Com/Total/Com_tot_regret" + str(no), COM_TOT_REGRET)
        np.save("Com/Total/Com_tot_reward" + str(no), COM_TOT_REWARD)
        np.save("Com/Total/Com_F" + str(no), COM_F)

        np.save("Com/Total/N_ij_T" + str(no), Nij_T_MAIN)
        np.save("Com/Total/N_ij_T_s" + str(no), Nij_T_s_MAIN)

