import numpy as np
import matplotlib.pyplot as plt
from math import  log


no_agents = 20
no_bandits = 100
no_iter = 5000

#prob = [0, 1, 2, 3, 4]
#prob = [0, 1, 2, 3, 4, 5, 6]
prob = [0, 4 ,8, 12, 16]
#prob = [2, 6]
#leg = [str(k*4) for  k in prob]
leg = [str(k) for  k in prob]
for i in range(len(leg)):
    leg[i] = 'nj = ' + leg[i]

Nij_T_s = []
Nij_T = []

#for i in range(0,20,4):
#for i in range(7):


for i in prob:
    print(i)
    Nij_T_s.append(np.load("Com/Total/N_ij_T_s" + str(i)+".npy"))
    Nij_T.append(np.load("Com/Total/N_ij_T" + str(i)+".npy"))




print(Nij_T[0][4][44].tolist()[4990:5000])
print(Nij_T_s[0][4][44].tolist()[4990:5000])


variance = np.load("Variance.npy")
K = [ (1/((8*variance[i])**2)*2) for i in range(no_bandits)]


C = []
for i in range(len(prob)):
    c = [[] for a in range(no_agents)]
    for j in range(no_agents):
        for k in range(no_bandits):
            k_ = K[k]
            alpha = 3/(2*k_)

            #GAMMA = [ alpha*log(T) for T in range(1,no_iter+1)]
            #GAMMA = [ log(T) for T in range(1,no_iter+1)]
            #GAMMA.insert(0,0)

            c[j].append(np.divide(Nij_T[i][j][k]-Nij_T_s[i][j][k], Nij_T_s[i][j][k]))
            #c[j].append(np.divide(Nij_T[i][j][k] - Nij_T_s[i][j][k], GAMMA))

    C.append(c)

C = np.array(C)

'''
for ban in range(no_bandits):
    #if ban==99 or ban==44 or ban==3:
    if ban != 101:
        fig1, ax = plt.subplots(1, 1, figsize=(8,8))
        for i in range(len(prob)):
            print(ban, i)
            plt.plot((1/no_agents)*(np.sum(C[i][:,ban], axis=0)),linewidth=3)
            #if ban ==99:
            #    plt.title("Agent's expected Communication Effect for the optimal bandit no " + str(ban))
            #if ban ==44:
            #    plt.title("Agent's expected Communication Effect for the sub optimal bandit no " + str(ban))
            #if ban ==3:
        plt.title("Agent's expected Communication Effect for the bandit no " + str(ban))
        plt.xlabel("Time",fontsize=15)
        plt.ylabel("Effect",fontsize=15)
        plt.legend(leg, fontsize=15)
        #if ban ==99:
        #    name = "For_Prob_" + str(i)+ "Agent_Com_Effect_for_"  + "Optimal"
        #if ban ==44:
        #    name = "For_Prob_" + str(i)+ "Agent_Com_Effect_for_"  + "Sub_Optimal"
        #if ban ==3:
        #    name = "For_Prob_" + str(i)+ "Agent_Com_Effect_for_"  + "Lowest_bandit"
        name = "For_Prob_" + str(i)+ "Agent_Com_Effect_for_"  + str(ban)
        plt.savefig("Com_Effect/Singular/" + name)
        plt.close(fig1)
        
'''

fig1, ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    print(i)
    plt.plot((1/no_agents)*(np.sum((1/98)*np.sum(C[i][:,0:98], axis=1), axis=0)),linewidth=3)
    plt.title("Agent's Expected Cumulative Communication Effect for all Sub Optimal Bandits")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Effect",fontsize=15)
    plt.legend(leg, fontsize=15)
    name = "Cumulative_Com_Effect_for_prob_" + str(i)
    plt.savefig("Com_Effect/AVG_SUB_OPTIMAL/" + name)

fig1, ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):

    plt.plot((1/no_agents)*(np.sum((1/99)*np.sum(C[i][:,0:99], axis=1), axis=0)),linewidth=3)
    plt.title("Agent's Expected Cumulative Communication Effect for all Bandits")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Effect",fontsize=15)
    plt.legend(leg, fontsize=15)
    name = "Cumulative_Com_Effect_for_prob_" + str(i)
    plt.savefig("Com_Effect/AVG_ALL/" + name)
