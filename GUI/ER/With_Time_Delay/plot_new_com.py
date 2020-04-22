import numpy as np
import matplotlib.pyplot as plt



no_agents = 20
no_bandits = 100


#prob = [0, 1, 4, 8, 10]
prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
leg = ["p = 0", "p = 0.1", "p = 0.3", "p = 0.5", "p = 0.7", "p = 0.9", "p = 1"]

New_Com_Metric  = []

mean = np.load("Mean.npy")
variance = np.load("Variance.npy")

print(np.argsort(mean))
for i in prob:
    New_Com_Metric.append(np.load("Com/Total/New_Com_Metric" + str(i) + ".npy"))

print(np.shape(New_Com_Metric[6][0][99][0:10]))
for ban in range(no_bandits):
    if ban==99 or ban==44 or ban==3:
        fig1, ax = plt.subplots(1, 1, figsize=(8,8))
        for i in range(len(prob)):
            print(ban, i)
            if i == 4 or i == 6 or i == 8 or i == 2:
                continue
            plt.plot((1/no_agents)*(np.sum(New_Com_Metric[i][:,ban], axis=0)),linewidth=3)
            if ban ==99:
                plt.title("Agent's expected Communication Efficency for the optimal bandit no " + str(ban))
            if ban ==44:
                plt.title("Agent's expected Communication Efficency for the sub optimal bandit no " + str(ban))
            if ban ==3:
                plt.title("Agent's expected Communication Efficency for the lowest bandit no " + str(ban))
            plt.xlabel("Time",fontsize=15)
            plt.ylabel("Efficency",fontsize=15)
            plt.legend(leg, fontsize=15)

            if ban ==99:
                name = "For_Prob_" + str(i)+ "Agent_new_Com_for_"  + "Optimal"
            if ban ==44:
                name = "For_Prob_" + str(i)+ "Agent_new_Com_for_"  + "Sub_Optimal"
            if ban ==3:
                name = "For_Prob_" + str(i)+ "Agent_new_Com_for_"  + "Lowest_bandit"
            plt.savefig("Com/New/Plot/" + name)
        plt.close(fig1)
