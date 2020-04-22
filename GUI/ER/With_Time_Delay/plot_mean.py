import  numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

no_agents = 20
no_bandits = 100

mean = np.load("mean.npy")
mean = mean.tolist()
fig, ax = plt.subplots(1)
M = [[0 for i in range(10)]for j in range(10)]
for j in range(10):
    for i in range(10):
        M[j][i] = mean[j*10+i]

o = np.argmax(np.array(mean))
arg_mean  = np.argsort(np.array(mean))
s_o = arg_mean[-2]

c1 =Circle((9, 9), radius=0.1, color='red')
c2 =Circle((4, 4), radius=0.1, color='blue')

m = [9]
imgplot = ax.imshow(M)
imgplot.set_interpolation('None')
ax.set_title("Bandit Distribution")
plt.colorbar(imgplot)
ax.add_patch(c1)
ax.add_patch(c2)
ax.text(o%10, o//10, "Optimal",fontsize=5)
ax.text(s_o%10, s_o//10, "Sub Optimal",fontsize=5)
filename = "Mean_HeatMap_With_Labels.png"
plt.savefig(filename)
