import numpy as np
import matplotlib.pyplot as plt



no_agents = 20
no_bandits = 100


#prob = [0, 1, 4, 8, 10]
#prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#leg = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
#leg = ["p = 0", "p = 0.1", "p = 0.3", "p = 0.5", "p = 0.7", "p = 0.9", "p = 1"]
prob = [0, 1, 4, 8, 10]
leg = [str(k) for  k in prob]
for i in range(len(leg)):
    leg[i] = "p = 0." + leg[i]
leg[-1] = 'p = 1'
Nij_T_s = []
Nij_T = []

for i in prob:
    Nij_T_s.append(np.load("Com/Total/N_ij_T_s" + str(i)+".npy"))
    Nij_T.append(np.load("Com/Total/N_ij_T" + str(i)+".npy"))

print(Nij_T_s[0][0][99].tolist())
C = []
for i in range(len(prob)):
    c = [[] for a in range(no_agents)]
    for j in range(no_agents):
        for k in range(no_bandits):
            c[j].append(np.divide(Nij_T[i][j][k] - Nij_T_s[i][j][k], Nij_T_s[i][j][k]))

    C.append(c)

C = np.array(C)
for ban in range(no_bandits):
    if ban==99 or ban==44 or ban==3:
        fig1, ax = plt.subplots(1, 1, figsize=(8,8))
        for i in range(len(prob)):
            print(ban, i)

            plt.plot((1/no_agents)*(np.sum(C[i][:,ban], axis=0)),linewidth=3)
            if ban ==99:
                plt.title("Agent's expected Communication Effect for the optimal bandit no " + str(ban))
            if ban ==44:
                plt.title("Agent's expected Communication Effect for the sub optimal bandit no " + str(ban))
            if ban ==3:
                plt.title("Agent's expected Communication Effect for the lowest bandit no " + str(ban))
            plt.xlabel("Time",fontsize=15)
            plt.ylabel("Effect",fontsize=15)
            plt.legend(leg, fontsize=15)

            if ban ==99:
                name = "For_Prob_" + str(prob[i])+ "Agent_Com_Effect_for_"  + "Optimal"
            if ban ==44:
                name = "For_Prob_" + str(prob[i])+ "Agent_Com_Effect_for_"  + "Sub_Optimal"
            if ban ==3:
                name = "For_Prob_" + str(prob[i])+ "Agent_Com_Effect_for_"  + "Lowest_bandit"
            plt.savefig("Com_Effect/Singular/" + name)
        plt.close(fig1)



fig1, ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):

    plt.plot((1/no_agents)*(np.sum(np.sum(C[i][:,0:98], axis=1), axis=0)),linewidth=3)
    plt.title("Agent's Expected Cumulative Communication Effect for all Sub Optimal Bandits")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Effect",fontsize=15)
    plt.legend(leg, fontsize=15)
    name = "Cumulative_Com_Effect_for_prob_" + str(prob[i])
    plt.savefig("Com_Effect/AVG/" + name)

