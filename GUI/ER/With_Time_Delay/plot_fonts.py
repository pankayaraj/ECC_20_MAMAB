import numpy as np
import matplotlib.pyplot as plt



no_agents = 20
no_bandits = 100


#prob = [0, 1, 4, 8, 10]
prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
leg = ["p = 0", "p = 0.1", "p = 0.3", "p = 0.5", "p = 0.7", "p = 0.9", "p = 1"]



agent_tot = []

mean = np.load("Mean.npy")
variance = np.load("Variance.npy")

print(np.argsort(mean))
for i in prob:
    agent_tot.append(np.load("Agent/Total/Agent_tot_regret"+str(i)+".npy"))

#1

# agent's average expected total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8, 8))
for i in range(len(prob)):
        if i == 4 or i == 6 or i == 8 or i == 2:
                continue
        plt.plot((1/no_agents)*np.sum(agent_tot[i], axis=0))
        plt.title("Agent's expected Cumulative regret")
        plt.xlabel("Time")
        plt.ylabel("Regret")
        plt.legend(leg)

        name = "1"
plt.savefig("Plot_Samples/" + name)
plt.close(fig3)

#2

# agent's average expected total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8, 8))
for i in range(len(prob)):
        if i == 4 or i == 6 or i == 8 or i == 2:
                continue
        plt.plot((1/no_agents)*np.sum(agent_tot[i], axis=0))
        plt.title("Agent's expected Cumulative regret")
        plt.xlabel("Time")
        plt.ylabel("Regret")
        plt.legend(leg, fontsize=15)

        name = "2"
plt.savefig("Plot_Samples/" + name)
plt.close(fig3)

#3

# agent's average expected total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8, 8))
for i in range(len(prob)):
        if i == 4 or i == 6 or i == 8 or i == 2:
                continue
        plt.plot((1/no_agents)*np.sum(agent_tot[i], axis=0) ,linewidth=3)
        plt.title("Agent's expected Cumulative regret")
        plt.xlabel("Time")
        plt.ylabel("Regret")
        plt.legend(leg, fontsize=15)

        name = "3"
plt.savefig("Plot_Samples/" + name)
plt.close(fig3)


