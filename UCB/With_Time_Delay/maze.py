import numpy as np
from Bandit import Bandit
from Set import Set

mean = np.load("Mean.npy")
variance = np.load("Variance.npy")

no_bandits = 100


Bandits = [Bandit(i, mean=mean[i], std=variance[i]) for i in range(no_bandits)]

Sets = []
for i in range(1,9):
    for j in range(1, 9):

        Sets.append(Set(index=i*10+j,bandit_list=[
            Bandits[(i-1)*10 + j],
            Bandits[(i-1)*10 + j-1],
            Bandits[(i-1)*10 + j+1],
            Bandits[(i+1)*10 + j],
            Bandits[(i+1)*10 + j-1],
            Bandits[(i+1)*10 + j+1],
            Bandits[i*10+j],
            Bandits[i*10 + j-1],
            Bandits[i*10 + j+1]
        ],adj_set=[
            (i-1)*10 + j,
            (i-1)*10 + j-1,
            (i-1)*10 + j+1,
            (i+1)*10 + j,
            (i+1)*10 + j-1,
            (i+1)*10 + j+1,
            i*10 + j-1,
            i*10 + j+1
            ]))


for i in range(1, 9):

    Sets.append(Set(index=i,bandit_list=[

            Bandits[10 + i + 1],
            Bandits[10 + i - 1],
            Bandits[10 + i],
            Bandits[i],
            Bandits[i-1],
            Bandits[i+1]
        ],adj_set=[

            10 + i + 1,
            10 + i - 1,
            10 + i,
            i-1,
            i+1
            ]))

    j = i*10
    k = i
    Sets.append(Set(index=j,bandit_list=[

            Bandits[(k+1)*10 + 1],
            Bandits[(k+1)*10],
            Bandits[j+1],
            Bandits[j],
            Bandits[(k-1)*10 +1],
            Bandits[(k-1)*10]
        ],adj_set=[

            (k+1)*10,
            (k+1)*10+1,
            (k-1)*10+1,
            (k-1)*10,
            j+1
            ]))

    j = i*10+9
    k = i
    Sets.append(Set(index=j,bandit_list=[

            Bandits[(k+1)*10 + 8],
            Bandits[(k+1)*10 + 9],
            Bandits[j-1],
            Bandits[j],
            Bandits[(k-1)*10 + 8],
            Bandits[(k-1)*10 + 9]
        ],adj_set=[

            (k+1)*10 + 9,
            (k+1)*10 + 8,
            (k-1)*10 + 9,
            (k-1)*10 + 8,
            j-1
            ]))



for i in range(91, 99):
    j = i%10
    Sets.append(Set(index=i,bandit_list=[
            
            Bandits[80 + j + 1],
            Bandits[80 + j - 1],
            Bandits[80 + j],
            Bandits[i-1],
            Bandits[i+1],
            Bandits[i]
        ],adj_set=[
            
            80 + j + 1,
            80 + j - 1,
            80 + j,
            i-1,
            i+1
            ]))

Sets.append(Set(index=0,bandit_list=[Bandits[1],Bandits[10],Bandits[11],Bandits[0],],adj_set=[10,11,1]))
Sets.append(Set(index=9,bandit_list=[Bandits[8],Bandits[19],Bandits[18],Bandits[9],],adj_set=[18,19,8]))
Sets.append(Set(index=90,bandit_list=[Bandits[90],Bandits[91],Bandits[80],Bandits[81],],adj_set=[81,80,91]))
Sets.append(Set(index=99,bandit_list=[Bandits[98],Bandits[99],Bandits[88],Bandits[89],],adj_set=[98,89,88]))

S = []
for i in range(100):
    for j in Sets:
        if j.index == i:
            S.append(j)



Sets = S

