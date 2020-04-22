import numpy as np
import matplotlib.pyplot as plt



no_agents = 20
no_bandits = 100


prob = [0, 1, 4, 8, 10]

#prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#leg = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9" "1"]
leg = [str(k) for  k in prob]
for i in range(len(leg)):
    leg[i] = "p = 0." + leg[i]
leg[-1] = 'p = 1'

agent_tot = []
global_total = []
com_total = []
self_total = []

agent_tot_rew = []
com_total_rew = []
self_total_rew = []

self_F = []
com_F = []

duplicate = []
estimate = []

mean = np.load("Agent/Mean.npy")
variance = np.load("Agent/Variance.npy")

print(np.argsort(mean))
for i in prob:

    agent_tot.append(np.load("Agent/Total/Agent_tot_regret"+str(i)+".npy"))

    global_total.append(np.load("Global/Total/Global_tot_regret" + str(i) + ".npy"))
    com_total.append(np.load("Com/Total/Com_tot_regret" + str(i) + ".npy"))
    self_total.append(np.load("Self/Total/Self_tot_regret" + str(i) + ".npy"))

    agent_tot_rew.append(np.load("Agent/Total/Agent_tot_reward"+str(i)+".npy"))
    com_total_rew.append(np.load("Com/Total/Com_tot_reward" + str(i) + ".npy"))
    self_total_rew.append(np.load("Self/Total/Self_tot_reward" + str(i) + ".npy"))

    duplicate.append(np.load("Agent/Total/Duplicate" + str(i) + ".npy"))
    estimate.append(np.load("Agent/Total/Estimate" + str(i)+ ".npy"))

for i in prob:
    self_F.append(np.load("Self/Total/Self_F" + str(i) + ".npy"))
    com_F.append(np.load("Com/Total/Com_F" + str(i) + ".npy"))
print(mean)


#mean plot
fig = plt.figure()
plt.bar([i for i in range(no_bandits)], mean)
plt.title("Bandit Mean")
plt.xlabel("Bandit Index")
plt.ylabel("Mean")
plt.savefig("Mean")
plt.close(fig)

#variance plot
fig = plt.figure()
plt.bar([i for i in range(no_bandits)],variance)
plt.title("Bandit Variance")
plt.xlabel("Bandit Index")
plt.ylabel("Variance")
plt.savefig("Variance")
plt.close(fig)

#PLOTTING F_ik
SELF_F = np.array(self_F)
COM_F =  np.array(com_F)



print(np.array(com_F).shape)
print(np.array(COM_F).shape)

'''
for i in range(len(COM_F[0][0])):
    if COM_F[0][0][i] == 0:
        COM_F[0][0][i] = 0.000000000000001
'''


for ban in range(no_bandits):
    fig1, ax = plt.subplots(1, 1, figsize=(8,8))
    l = leg[1:6]
    print(l)
    for i in range(len(prob)):
        if i == 0:
            continue

        plt.plot(np.divide(SELF_F[i][3][ban], (SELF_F[i][3][ban]+COM_F[i][3][ban])),linewidth=3)
        plt.title("Agent's F_ik ratio for bandit no " + str(ban))
        plt.xlabel("Time",fontsize=15)
        plt.ylabel("F_ik Ratio",fontsize=15)
        plt.legend(l,fontsize=15)
        name = "Agent_F_ik"  + str(ban)
        plt.savefig("Agent/Ratio/Plot/" + name)
    plt.close(fig1)
print("en")

#leg = ["p = 0", "p = 0.1", "p = 0.2", "p = 0.3", "p = 0.4", "p = 0.5", "p = 0.6", "p = 0.7", "p = 0.8", "p = 0.9" "p = 1"]



#REGRET PLOTS

#Global expected total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    plt.plot(global_total[i],linewidth=3)
    plt.title("Global expected regret witn p")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)

    plt.legend(leg,fontsize=15)
    name = "Global_tot_regret_T_0_" + str(prob[i])
    plt.savefig("Global/Total/Plot/" + name)
plt.close(fig3)


# agent's average expected total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    print(i)
    plt.plot((1/no_agents)*np.sum(agent_tot[i], axis=0),linewidth=3)
    plt.title("Agent's expected Cumulative regret")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    plt.legend(leg,fontsize=15)

    name = "Agent_avg_tot_regret_T_0_" + str(prob[i])
    plt.savefig("Agent/Total/Plot/" + name)
plt.close(fig3)


#Agent's average self vs communication regret

for i in range(len(prob)):


    fig3 , ax = plt.subplots(1, 1, figsize=(8,8))

    plt.plot((1/no_agents)*np.sum(com_total[i],axis=0),linewidth=3)
    plt.plot((1/no_agents)*np.sum(self_total[i],axis=0),linewidth=3)
    plt.title("Expecte Communication vs Self regret for prob" + str(prob[i]/10))
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    plt.legend([ "Communication", "Self"],fontsize=15)
    name = "Com_tot_regret_V_Self_total_T_0_" + str(prob[i])
    plt.savefig("Com/Total/Plot/" + name)
    plt.close(fig3)


#agent's average expected  communication total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    plt.plot((1/no_agents)*np.sum(com_total[i], axis=0),linewidth=3)
    plt.title("Agent's expected Communication regret")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    plt.legend(leg,fontsize=15)

    name = "Agent_avg_com_tot_regret_T_0_" + str(prob[i])
    plt.savefig("Com/Total/Plot/" + name)
plt.close(fig3)




#agent's average expected  self total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    plt.plot((1/no_agents)*np.sum(self_total[i], axis=0),linewidth=3)
    plt.title("Agent's expected Self regret")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    plt.legend(leg,fontsize=15)

    name = "Agent_avg_self_tot_regret_T_0_" + str(prob[i])
    plt.savefig("Self/Total/Plot/" + name)
plt.close(fig3)


#REWARD PLOTS

#agent's average expected total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    plt.plot((1/no_agents)*np.sum(agent_tot_rew[i], axis=0),linewidth=3)
    plt.title("Agent's expected Cumulative reward")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Reward",fontsize=15)
    plt.legend(leg,fontsize=15)

    name = "Agent_avg_tot_reward_T_0_" + str(prob[i])
    plt.savefig("Agent/Total/Plot/" + name)
plt.close(fig3)



#agent's average expected  communication total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    plt.plot((1/no_agents)*np.sum(com_total_rew[i], axis=0),linewidth=3)
    plt.title("Agent's expected Communication reward")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Reward",fontsize=15)
    plt.legend(leg,fontsize=15)

    name = "Agent_avg_com_tot_reward_T_0_" + str(prob[i])
    plt.savefig("Com/Total/Plot/" + name)
plt.close(fig3)


#agent's average expected  self total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in range(len(prob)):
    plt.plot((1/no_agents)*np.sum(self_total_rew[i], axis=0),linewidth=3)
    plt.title("Agent's expected Self reward")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Reward",fontsize=15)
    plt.legend(leg,fontsize=15)

    name = "Agent_avg_self_tot_reward_T_0_" + str(prob[i])
    plt.savefig("Self/Total/Plot/" + name)
plt.close(fig3)



#agent's average expected  duplicate  n estimate plot for a bandit with p

for ban in range(no_bandits):
    fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
    for i in range(len(prob)):
        plt.plot(duplicate[i][0][ban],linewidth=3)
        plt.title("Agent's expected duplicate count for bandit " + str(ban) + " witn p")
        plt.xlabel("Time",fontsize=15)
        plt.ylabel("Regret",fontsize=15)
        plt.legend(leg,fontsize=15)

        name = "Agent_duplicate_b_"  + str(ban) + "_for_p_" + str(prob[i])
        plt.savefig("Agent/Total/Plot/" + name)
    plt.close(fig3)



for ban in range(no_bandits):
    fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
    for i in range(len(prob)):
        plt.plot(estimate[i][0][ban],linewidth=3)
        plt.title("Agent 0's Estimate for bandit " + str(ban) + " witn p")
        plt.xlabel("Time",fontsize=15)
        plt.ylabel("Estimate",fontsize=15)
        plt.legend(leg,fontsize=15)

        name = "Estimate_b_"  + str(ban) + "_for_p_" + str(prob[i])
        plt.savefig("Agent/Estimate/Plot/" + name)
    plt.close(fig3)


