import numpy as np
import matplotlib.pyplot as plt



no_agents = 20
no_bandits = 100


#prob = [0, 1, 4, 8, 9]
#prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
prob = [0, 1, 2, 3, 4]
#leg = [str(k) for  k in range(0, no_agents, int(no_agents/10))]
#leg = [str(k*2) for  k in prob]
leg = [str(k*4) for  k in prob]


for i in range(len(leg)):
    leg[i] = 'nj = ' + leg[i]

agent_tot = []
global_total = []
com_total = []
self_total = []

mean = np.load("Agent/Mean.npy")
variance = np.load("Agent/Variance.npy")




#for i in range(0, no_agents, int(no_agents/10)):
#for i in range(10):
#for i in range(6):
for i in range(0,20,4):
    print(i)
    agent_tot.append(np.load("Agent/Total/Agent_tot_regret"+str(i)+".npy"))
    #global_total.append(np.load("Global/Total/Global_tot_regret" + str(i) + ".npy"))
    com_total.append(np.load("Com/Total/Com_tot_regret" + str(i) + ".npy"))
    self_total.append(np.load("Self/Total/Self_tot_regret" + str(i) + ".npy"))
print(mean)
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


#agent's average expected total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8, 8))
for i in prob:
    print(i)
    plt.plot((1/no_agents)*np.sum(agent_tot[i], axis=0),linewidth=3)
    plt.title("Agent's expected Cumulative regret ")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    plt.legend(leg, fontsize=15)
    name = "Agent_avg_tot_regret_T_0_" + str(i)
    plt.savefig("Agent/Total/Plot/" + name)
plt.close(fig3)

#agent's average expected  communication total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8,8))
for i in prob:

    plt.plot((1/no_agents)*np.sum(com_total[i], axis=0),linewidth=3)
    plt.title("Agent's expected Communiation regret")
    plt.xlabel("Time",fontsize=15)
    plt.ylabel("Regret",fontsize=15)
    plt.legend(leg, fontsize=15)
    name = "Agent_avg_com_tot_regret_T_0_" + str(i)
    plt.savefig("Com/Total/Plot/" + name)
plt.close(fig3)


#agent's average expected  self total regret plot
fig3 , ax = plt.subplots(1, 1, figsize=(8, 8))
for i in prob:

    plt.plot((1/no_agents)*np.sum(self_total[i], axis=0),linewidth=3)
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

duplicate = []
estimate = []

self_F = []
com_F = []
for i in range(0,20,4):
    print(i)
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

#Global expected total regret plot
fig1, ax = plt.subplots(1, 1, figsize=(7, 7))
for i in prob:

    plt.plot(global_total[i])
    plt.title("Global total regret witn max allowed no")
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.legend(leg)
    name = "Global_tot_regret_T_0_" + str(i)
    plt.savefig("Global/Total/Plot/" + name)
plt.close(fig1)


#Agent's average self vs communication regret

for i in prob:


    fig1, ax = plt.subplots(1, 1, figsize=(7, 7))

    plt.plot((1/no_agents)*np.sum(com_total[i],axis=0))
    plt.plot((1/no_agents)*np.sum(self_total[i],axis=0))
    plt.title("Average Communication vs Self regret for Connection" + str(i))
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.legend([ "Communication", "Self"])
    name = "Com_tot_regret_V_Self_total_T_0_" + str(i)
    plt.savefig("Com/Total/Plot/" + name)
    plt.close(fig1)



#agent's average expected total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(7, 7))
for i in prob:
    plt.plot((1/no_agents)*np.sum(agent_tot_rew[i], axis=0))
    plt.title("Agent's expected Cumulative reward")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.legend(leg)

    name = "Agent_avg_tot_reward_T_0_" + str(i)
    plt.savefig("Agent/Total/Plot/" + name)
plt.close(fig3)



#agent's average expected  communication total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(7, 7))
for i in prob:
    plt.plot((1/no_agents)*np.sum(com_total_rew[i], axis=0))
    plt.title("Agent's expected Communication reward")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.legend(leg)

    name = "Agent_avg_com_tot_reward_T_0_" + str(i)
    plt.savefig("Com/Total/Plot/" + name)
plt.close(fig3)


#agent's average expected  self total reward plot
fig3 , ax = plt.subplots(1, 1, figsize=(7, 7))
for i in prob:
    plt.plot((1/no_agents)*np.sum(self_total_rew[i], axis=0))
    plt.title("Agent's expected Self reward ")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.legend(leg)

    name = "Agent_avg_self_tot_reward_T_0_" + str(i)
    plt.savefig("Self/Total/Plot/" + name)
plt.close(fig3)



'''

'''


#PLOTTING F_ik
SELF_F = (1/no_agents)*np.array(self_F).sum(1)
COM_F =  (1/no_agents)*np.array(com_F).sum(1)



print(np.array(com_F).shape)
print(np.array(COM_F).shape)
'''
#for i in range(len(COM_F[0][0])):
#    if COM_F[0][0][i] == 0:
#        COM_F[0][0][i] = 0.00000000000000000001
'''

for ban in range(no_bandits):
    fig1, ax = plt.subplots(1, 1, figsize=(7, 7))
    if ban != 18:
        leg = [ "nj = 2", "nj = 4", "nj = 6", "nj = 8", "nj = 10", "nj = 12", "nj = 14", "nj = 16", "nj = 18"]

    else:
        leg = ["nj = 0", "nj = 2", "nj = 4", "nj = 6", "nj = 8", "nj = 10", "nj = 12", "nj = 14", "nj = 16", "nj = 18"]

    for i in prob:

        if ban != 18 and i==0:
            continue
        #plt.plot(COM_F[i][ban])
        plt.plot(np.divide(SELF_F[i][ban], (SELF_F[i][ban]+COM_F[i][ban])))
    plt.title("Agent's F_ik ratio for bandit no " + str(ban))
    plt.xlabel("Time")
    plt.ylabel("F_ik Ratio")
    plt.legend(leg)
    name = "Agent_F_ik"  + str(ban)
    plt.savefig("Agent/Ratio/Plot/" + name)
    plt.close(fig1)

leg = ["nj = 0", "nj = 2", "nj = 4", "nj = 6", "nj = 8", "nj = 10", "nj = 12", "nj = 14", "nj = 16", "nj = 18"]
leg = [str(k*1) for  k in prob]

#REGRET PLOTSSELF_F = (1/no_agents)*np.array(self_F).sum(0)
'''
