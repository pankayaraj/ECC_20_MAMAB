import numpy as np
from .Gaussian_Reward import Gaussian_Reward
from .erdos_renyl_model_without_duplicate import MAMAB


def ER_wtd(no_iterations, no_agents, no_bandits, no_experiments, mean, variance):
    o = np.argmax(np.array(mean))
    bandits = [Gaussian_Reward(mean[i], variance[i]) for i in range(no_bandits)]

    #COMPUTE DELTA
    maxi_mean = np.argsort(mean)
    maxi_mean = np.flip(maxi_mean, axis=0)
    max_index = maxi_mean[0]
    sec_max_index = maxi_mean[0]
    for i in range(1, len(maxi_mean)):
        if mean[maxi_mean[i]] != mean[maxi_mean[0]]:
            sec_max_index = maxi_mean[i]
            break

    if sec_max_index != max_index:
        delta = mean[max_index]-mean[sec_max_index]
    else:
        delta = 1

    P = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    A_RW = []
    C_RW = []
    S_RW = []
    A_R = []
    C_R = []
    S_R = []
    C_F = []
    S_F = []
    N_T = []
    NS_T = []

    for prob in P:

            AGENT_TOT_REGRET = np.array([[0 for _ in range(no_iterations+1)] for i in range(no_agents)])
            SELF_TOT_REGRET = np.array([[0 for _ in range(no_iterations+1)] for __ in range(no_agents)])
            COM_TOT_REGRET = np.array([[0 for _ in range(no_iterations+1)] for __ in range(no_agents)])

            AGENT_TOT_REWARD = np.array([[0 for _ in range(no_iterations+1)] for i in range(no_agents)])
            SELF_TOT_REWARD = np.array([[0 for _ in range(no_iterations+1)] for __ in range(no_agents)])
            COM_TOT_REWARD = np.array([[0 for _ in range(no_iterations+1)] for __ in range(no_agents)])

            SELF_F =  [[[0 for _ in range(no_iterations+1)] for j in range(no_bandits)] for k in range(no_agents)]
            COM_F  =  [[[0 for _ in range(no_iterations+1)] for j in range(no_bandits)] for k in range(no_agents)]

            Nij_T_s_MAIN = [[[0 for _ in range(no_iterations+1)] for j in range(no_bandits)] for k in range(no_agents)]
            Nij_T_MAIN = [[[0 for _ in range(no_iterations+1)] for j in range(no_bandits)] for k in range(no_agents)]

            for experiment  in range(1, no_experiments+1):


                p = prob/10
                print("Experiment " + str(experiment) + "for probability " + str(p))
                G = MAMAB(no_bandits=no_bandits, no_agents=no_agents, bandits=bandits,optimal_bandit_index=o , p=p, reward_vairance=variance, delta=delta,
                                                                                                                                    no_iter=no_iterations)

                for i in range(no_iterations):

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

            AGENT_TOT_REGRET = (1/experiment)*AGENT_TOT_REGRET
            SELF_TOT_REGRET = (1/experiment)*SELF_TOT_REGRET
            COM_TOT_REGRET = (1/experiment)*COM_TOT_REGRET

            AGENT_TOT_REWARD = (1/experiment)*AGENT_TOT_REWARD
            SELF_TOT_REWARD = (1/experiment)*SELF_TOT_REWARD
            COM_TOT_REWARD = (1/experiment)*COM_TOT_REWARD

            SELF_F = (1/experiment)*SELF_F
            COM_F  = (1/experiment)*COM_F

            Nij_T_MAIN = (1/experiment)*Nij_T_MAIN
            Nij_T_s_MAIN = (1/experiment)*Nij_T_s_MAIN


            A_RW.append(AGENT_TOT_REWARD)
            C_RW.append(COM_TOT_REWARD)
            S_RW.append(SELF_TOT_REWARD)
            S_R.append(SELF_TOT_REGRET)
            C_R.append(COM_TOT_REGRET)
            A_R.append(AGENT_TOT_REGRET)
            C_F.append(COM_F)
            S_F.append(SELF_F)
            N_T.append(Nij_T_MAIN)
            NS_T.append(Nij_T_s_MAIN)

    return A_RW, C_RW, S_RW, A_R, C_R, S_R, C_F, S_F, N_T, NS_T, sec_max_index




