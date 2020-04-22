import numpy as np
import matplotlib.pyplot as plt



no_agents = 20
no_bandits = 100


#prob = [0, 1, 4, 8, 10]
prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
leg = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]



agent_tot = []
com_total = []
self_total = []

mean = np.load("Mean.npy")
variance = np.load("Variance.npy")

print(np.argsort(mean))
for i in prob:
    agent_tot.append(np.load("Agent/Total/Agent_tot_regret"+str(i)+".npy"))
    com_total.append(np.load("Com/Total/Com_tot_regret" + str(i) + ".npy"))
    self_total.append(np.load("Self/Total/Self_tot_regret" + str(i) + ".npy"))
print(mean)
#print(self_total[6][0:10])
#mean plot
fig = plt.figure()
plt.bar([i for i in range(no_bandits)], mean)
plt.title("Bandit Mean")
plt.xlabel("Bandit Index",fontsize=15)
plt.ylabel("Mean",fontsize=15)
plt.savefig("Mean")
plt.close(fig)

#variance plot
fig = plt.figure()
plt.bar([i for i in range(no_bandits)],variance)
plt.title("Bandit Variance")
plt.xlabel("Bandit Index",fontsize=15)
plt.ylabel("Variance",fontsize=15)
plt.savefig("Variance")
plt.close(fig)



leg = ["p = 0", "p = 0.1", "p = 0.3", "p = 0.5", "p = 0.7", "p = 0.9", "p = 1"]

# agent's average expected total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
        if i == 4 or i == 6 or i == 8 or i == 2:
                continue
        plt.plot((1/no_agents)*np.sum(agent_tot[i], axis=0) ,linewidth=3)
        plt.title("Agent's expected Cumulative regret")
        plt.xlabel("Time",fontsize=15)
        plt.ylabel("Regret",fontsize=15)
        plt.legend(leg, fontsize=15)

        name = "Agent_avg_tot_regret_T_0_" + str(i)
        plt.savefig("Agent/Total/Plot/" + name)
plt.close(fig3)

#agent's average expected  communication total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    if i == 4 or i == 6 or i == 8 or i == 2:
                continue
    plt.plot((1/no_agents)*np.sum(com_total[i], axis=0) ,linewidth=3)
    plt.title("Agent's expected Communication regret")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    plt.legend(leg, fontsize=15)

    name = "Agent_avg_com_tot_regret_T_0_" + str(i)
    plt.savefig("Com/Total/Plot/" + name)
plt.close(fig3)


#agent's average expected  self total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    if i == 4 or i == 6 or i == 8 or i == 2:
                continue
    plt.plot((1/no_agents)*np.sum(self_total[i], axis=0) ,linewidth=3)
    plt.title("Agent's expected Self regret")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    plt.legend(leg, fontsize=15)

    name = "Agent_avg_self_tot_regret_T_0_" + str(i)
    plt.savefig("Self/Total/Plot/" + name)
plt.close(fig3)

'''
agent_tot_rew = []
com_total_rew = []
self_total_rew = []

self_F = []
com_F = []

duplicate = []
estimate = []
agent_tot = []
global_total = []
com_total = []
self_total = []



New_Com_Metric  = []

choice = []
for i in prob:

    choice.append(np.load("Agent/Total/Agent_choice"+str(i)+".npy"))

    agent_tot.append(np.load("Agent/Total/Agent_tot_regret"+str(i)+".npy"))

    global_total.append(np.load("Global/Total/Global_tot_regret" + str(i) + ".npy"))
    com_total.append(np.load("Com/Total/Com_tot_regret" + str(i) + ".npy"))
    self_total.append(np.load("Self/Total/Self_tot_regret" + str(i) + ".npy"))

    agent_tot_rew.append(np.load("Agent/Total/Agent_tot_reward"+str(i)+".npy"))
    com_total_rew.append(np.load("Com/Total/Com_tot_reward" + str(i) + ".npy"))
    self_total_rew.append(np.load("Self/Total/Self_tot_reward" + str(i) + ".npy"))

    self_F.append(np.load("Self/Total/Self_F" + str(i) + ".npy"))
    com_F.append(np.load("Com/Total/Com_F" + str(i) + ".npy"))
#REWARD PLOTS

#agent's average expected total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(10, 10))
for i in range(len(prob)):
    plt.plot((1/no_agents)*np.sum(agent_tot_rew[i], axis=0))
    plt.title("Agent's expected Cumulative reward")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.legend(leg)

    name = "Agent_avg_tot_reward_T_0_" + str(i)
    plt.savefig("Agent/Total/Plot/" + name)
plt.close(fig3)



#agent's average expected  communication total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(10, 10))
for i in range(len(prob)):
    plt.plot((1/no_agents)*np.sum(com_total_rew[i], axis=0))
    plt.title("Agent's expected Communication reward")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.legend(leg)

    name = "Agent_avg_com_tot_reward_T_0_" + str(i)
    plt.savefig("Com/Total/Plot/" + name)
plt.close(fig3)


#agent's average expected  self total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(10, 10))
for i in range(len(prob)):
    plt.plot((1/no_agents)*np.sum(self_total_rew[i], axis=0))
    plt.title("Agent's expected Self reward")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.legend(leg)

    name = "Agent_avg_self_tot_reward_T_0_" + str(i)
    plt.savefig("Self/Total/Plot/" + name)
plt.close(fig3)



#REGRET PLOTS

#Global expected total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(10, 10))
for i in range(len(prob)):
    plt.plot(global_total[i])
    plt.title("Global expected regret witn p")
    plt.xlabel("Time")
    plt.ylabel("Regret")

    plt.legend(leg)
    name = "Global_tot_regret_T_0_" + str(i)
    plt.savefig("Global/Total/Plot/" + name)
plt.close(fig3)

for i in range(len(prob)):


    fig3 , ax = plt.subplots(1, 1, figsize=(10, 10))

    plt.plot((1/no_agents)*np.sum(com_total[i],axis=0))
    plt.plot((1/no_agents)*np.sum(self_total[i],axis=0))
    plt.title("Expecte Communication vs Self regret for prob" + str(i/10))
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.legend([ "Communication", "Self"])
    name = "Com_tot_regret_V_Self_total_T_0_" + str(i)
    plt.savefig("Com/Total/Plot/" + name)
    plt.close(fig3)


#PLOTTING F_ik
SELF_F = np.array(self_F)
COM_F =  np.array(com_F)




for ban in range(no_bandits):
    fig1, ax = plt.subplots(1, 1, figsize=(7, 7))
    if ban != 18:
        leg = [ "p = 0.1", "p = 0.2", "p = 0.3", "p = 0.4", "p = 0.5", "p = 0.6", "p = 0.7", "p = 0.8", "p = 0.9", "p = 1"]
    else:
        leg = ["p = 0", "p = 0.1", "p = 0.2", "p = 0.3", "p = 0.4", "p = 0.5", "p = 0.6", "p = 0.7", "p = 0.8", "p = 0.9", "p = 1"]
    for i in range(len(prob)):

        if ban != 18 and i==0:
            continue
        plt.plot(np.divide(SELF_F[i][3][ban], (SELF_F[i][3][ban]+COM_F[i][3][ban])))
        plt.title("Agent's F_ik ratio for bandit no " + str(ban))
        plt.xlabel("Time")
        plt.ylabel("F_ik Ratio")
        plt.legend(leg)
        name = "Agent_F_ik"  + str(ban)
        plt.savefig("Agent/Ratio/Plot/" + name)
    plt.close(fig1)

'''
