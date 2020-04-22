import numpy as np
import matplotlib.pyplot as plt



no_agents = 20
no_bandits = 100


#prob = [0, 1, 2, 3, 4]
prob = [0, 1, 4, 8, 9]
#leg = [str(k*4) for  k in prob]
leg = [str(2*k) for  k in prob]
for i in range(len(leg)):
    leg[i] = "nj = " + leg[i]

Nij_T_s = []
Nij_T = []

#for i in range(0,20,4):
for i in prob:
    Nij_T_s.append(np.load("Com/Total/N_ij_T_s" + str(i*2)+".npy"))
    Nij_T.append(np.load("Com/Total/N_ij_T" + str(i*2)+".npy"))

N_S_Change = [[[[0 for i in range(20000)] for j in range(no_bandits)] for k in range(no_agents)] for t in range(len(prob))]
N_Change   = [[[[0 for i in range(20000)] for j in range(no_bandits)] for k in range(no_agents)] for t in range(len(prob))]

for i in range(len(prob)):
    print(i)
    for j in range(no_agents):
        for k in range(no_bandits):

            N_S_Change[i][j][k] = Nij_T_s[i][j][k][1:20001] - Nij_T_s[i][j][k][0:20000]
            N_Change[i][j][k] = Nij_T[i][j][k][1:20001]- Nij_T[i][j][k][0:20000]
N_Change = np.array(N_Change)
N_S_Change = np.array(N_S_Change)
C_ = []

for i in range(len(prob)):
    c = [[] for a in range(no_agents)]
    for j in range(no_agents):
        for k in range(no_bandits):
            print(i, j, k)
            c[j].append(np.divide(N_Change[i][j][k] - N_S_Change[i][j][k],N_S_Change[i][j][k]))

    C_.append(c)

print(N_S_Change[1][0][44].tolist())
print(N_Change[1][0][44].tolist())
for i in range(len(prob)):
    fig1, ax = plt.subplots(1, 1, figsize=(8,8))
    plt.plot(C_[i][0][99],linewidth=3)
    plt.title("Self")
    plt.show()
    plt.close(fig1)
    fig1, ax = plt.subplots(1, 1, figsize=(8,8))
    plt.plot(C_[i][0][99],linewidth=3)
    plt.title("Total")
    plt.show()
    plt.close(fig1)

print(len(Nij_T_s[0][0][99].tolist()))
print(len(Nij_T_s))
C = []
for i in range(len(prob)):
    c = [[] for a in range(no_agents)]
    for j in range(no_agents):
        for k in range(no_bandits):
            print(i, j, k)
            c[j].append(np.divide(Nij_T[i][j][k] - Nij_T_s[i][j][k], Nij_T_s[i][j][k]))

    C.append(c)

C = np.array(C)
print(C.shape)
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
                name = "For_Prob_" + str(i)+ "Agent_Com_Effect_for_"  + "Optimal"
            if ban ==44:
                name = "For_Prob_" + str(i)+ "Agent_Com_Effect_for_"  + "Sub_Optimal"
            if ban ==3:
                name = "For_Prob_" + str(i)+ "Agent_Com_Effect_for_"  + "Lowest_bandit"
            plt.savefig("Com_Effect/Singular/" + name)
        plt.close(fig1)



fig1, ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):

    plt.plot((1/no_agents)*(np.sum(np.sum(C[i][:,0:98], axis=1), axis=0)),linewidth=3)
    plt.title("Agent's Expected Cumulative Communication Effect for all Sub Optimal Bandits")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Effect",fontsize=15)
    plt.legend(leg, fontsize=15)
    name = "Cumulative_Com_Effect_for_prob_" + str(i)
    plt.savefig("Com_Effect/AVG/" + name)
plt.close(fig1)


