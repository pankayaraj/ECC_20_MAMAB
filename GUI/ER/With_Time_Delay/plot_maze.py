import  numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.image as mpimg
import cv2
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 5
rcParams['figure.figsize'] = 12, 8

from PIL import Image

import matplotlib.cm as cm



from images2gif import writeGif
import imageio

from maze import Sets

no_agents = 20
no_bandits = 100
prob = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
leg = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9" "1"]
choice = []
agent_tot = []
#global_total = []
com_total = []
self_total = []
New_Com_Metric  = []
for i in range(11):
    choice.append(np.load("Agent/Total/Agent_choice"+str(i)+".npy", allow_pickle=True))
    agent_tot.append(np.load("Agent/Total/Agent_tot_regret"+str(i)+".npy",allow_pickle=True))
    #global_total.append(np.load("Global/Total/Global_tot_regret" + str(i) + ".npy",allow_pickle=True))
    com_total.append(np.load("Com/Total/Com_tot_regret" + str(i) + ".npy",allow_pickle=True))
    self_total.append(np.load("Self/Total/Self_tot_regret" + str(i) + ".npy",allow_pickle=True))
    New_Com_Metric.append(np.load("Com/Total/New_Com_Metric" + str(i) + ".npy",allow_pickle=True))

keyFrames = []
frames = 60.0


mean = np.load("mean.npy")
mean = mean.tolist()

M = [[0 for i in range(10)]for j in range(10)]
for j in range(10):
    for i in range(10):
        M[j][i] = mean[j*10+i]

o = np.argmax(np.array(mean))


#imgplot.set_interpolation('sinc')
#plt.show()

matplotlib.colors.Normalize(vmin=-1.,vmax=1.)
#theCM = cm.get_cmap("seismic")
theCM = cm.get_cmap("BrBG")
theCM._init()
#theCM = cm.get_cmap("BrBG")


alphas = np.abs(np.linspace(0, 1, theCM.N))
theCM._lut[:-3,-1] = alphas
n = 6
agents = 20

#for i in range(len(choice)):
a = 0
b = 20000
n_iter = 20000
agents = 20
for i in range(a,b):
    t = [None for i in range(20)]
    agent_loc = [[-1 for i in range(10)] for j in range(10)]
    print(i)
    fig = plt.figure(figsize=(15,10))
    gs = fig.add_gridspec(4, 4)
    ax0 = fig.add_subplot(gs[1:3, 2:4])

    ax3_self = fig.add_subplot(gs[0:1,2:4])

    ax4_mean = fig.add_subplot(gs[3:4,0:4])

    ax5_com_eff_for_optimal = fig.add_subplot(gs[0:1,0:2])
    ax5_com_eff_for_sub_optimal = fig.add_subplot(gs[1:2, 0:2])
    ax5_com_eff_for_lowest = fig.add_subplot(gs[2:3,0:2])

    for ag in range(agents):

        k = choice[n][ag][i][1] #indices prob, agent, iter, nextban

        #adj = Sets[j*10+i].adj_sets
        #print(Sets[k].adj_sets)
        '''
        adj = Sets[k].adj_sets
        
        for s_i in adj:
            agent_loc[s_i//10][s_i%10] = 60

        '''
        if choice[n][ag][i][0] != None:
            #agent_loc[k//10][k%10] = 0
            agent_loc[k//10][k%10] = 0.9
            #print("nt_none" + str(k))
        else:
            agent_loc[k//10][k%10] = -0.9
        t[ag] = ax0.text(k%10, k//10, "A"+str(ag),fontsize=10)

    imgplot = ax0.imshow(M)
    imgplot.set_interpolation('None')
    #imgplot.set_interpolation('sinc')
    imgplot = ax0.imshow(agent_loc, vmin=-1, vmax=1, cmap=theCM)
    ax0.text(o%10, o//10, "G",fontsize=10)
    markers_on = [i]


    #ax1_ag.plot((1/no_agents)*np.sum(agent_tot[n][:,0:i+1], axis=0), markevery=markers_on,markerfacecolor='red', marker='o')


    ax3_self.plot((1/no_agents)*np.sum(self_total[n][:,0:i+1], axis=0))
    ax3_self.set_title("Agent's expected Cumulative Self regret")
    ax3_self.set_xlabel("Time")
    ax3_self.set_ylabel("Regret")



    ax4_mean.bar([i for i in range(no_bandits)], mean)
    for ag in range(agents):
        if choice[n][ag][i][0] != None:
            pick = choice[n][ag][i][1]
            pick = pick//10*10 + pick%10
            ax4_mean.bar([pick], [mean[pick]], color='g')
    ax4_mean.set_title("Mean")

    ax5_com_eff_for_optimal.plot((1/no_agents)*(np.sum(New_Com_Metric[n][:,99][:,0:i+1], axis=0)))
    ax5_com_eff_for_optimal.set_title("Agent's Com Regret for optimal bandit no " + str(99))
    ax5_com_eff_for_optimal.set_xlabel("Time")
    ax5_com_eff_for_optimal.set_ylabel("Communication Regret")

    ax5_com_eff_for_sub_optimal.plot((1/no_agents)*(np.sum(New_Com_Metric[n][:,44][:,0:i+1], axis=0)))
    ax5_com_eff_for_sub_optimal.set_title("Agent's Com Regret for the second optimal bandit no " + str(44))
    ax5_com_eff_for_sub_optimal.set_xlabel("Time")
    ax5_com_eff_for_sub_optimal.set_ylabel("Communication Regret")

    ax5_com_eff_for_lowest.plot((1/no_agents)*(np.sum(New_Com_Metric[n][:,3][:,0:i+1], axis=0)))
    ax5_com_eff_for_lowest.set_title("Agent's Com Regret for the lowest bandit no " + str(3))
    ax5_com_eff_for_lowest.set_xlabel("Time")
    ax5_com_eff_for_lowest.set_ylabel("Communication Regret")

    plt.tight_layout()
    filename = "Maze_Images1/maze" + str(i) + ".png"
    plt.savefig(filename)
    keyFrames.append(filename)
    for agent in range(agents):
        t[agent].set_visible(False)

    plt.tight_layout()
    plt.close(fig)

#images = [imageio.imread(fn) for fn in keyFrames]
#gifFilename = "maze.gif"
#writeGif(gifFilename, images, duration=0.3)
#plt.clf()

#imageio.mimsave('maze_with_interpolation.gif', images, fps=3)
#imageio.mimsave('maze_fpr_prob' + str(n) +"_for_iter_" +str(a) + "-" + str(b) + '.gif', images, fps=3)
'''
size = (12, 8)

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'XVID'), 15, size)
for im in images:
    out.write(im)
out.release()
'''
