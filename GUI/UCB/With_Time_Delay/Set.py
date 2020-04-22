import numpy as np
from math import sqrt, log
from Bandit import Bandit

#from maze import Sets


'''
def get_nearby_set_to_ban(set_index_list, bandit_index, ban_to_set_dict):

    sets = ban_to_set_dict[str(bandit_index)]
    min_index = set_index_list[0]
    min = abs(sets[0]-min_index)
    for i in set_index_list:
        for j in sets:
            if abs(j-i) < min:
                min_index = i
                min = abs(j-i)




    return  min_index
'''
def euclidian_dist(c1, c2):

    return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2



#for maze formation
def get_nearby_set_to_ban(set_index_list, bandit_index, ban_to_set_dict):

    sets = ban_to_set_dict[str(bandit_index)]

    current_coordinates = [(i//10, i%10) for i in set_index_list]
    ban_coordinates = [(i//10, i%10) for i in sets]

    min_index = set_index_list[0]
    min = euclidian_dist(ban_coordinates[0], current_coordinates[0])

    for i in range(len(current_coordinates)):
        for j in range(len(ban_coordinates)):

            dist = euclidian_dist(current_coordinates[i], ban_coordinates[j])

            if dist < min:

                min_index = set_index_list[i]
                min = dist

    return  min_index


#for maze formation in a single choice setting
def get_nearby_set_to_ban_1(set_index_list, bandit_index, ban_to_set_dict):

    sets = ban_to_set_dict[str(bandit_index)]

    current_coordinates = [(i//10, i%10) for i in set_index_list]
    #ban_coordinates = [(i//10, i%10) for i in sets]
    ban_coordinate = (bandit_index//10, bandit_index%10)
    min_index = set_index_list[0]
    min = euclidian_dist(ban_coordinate, current_coordinates[0])

    for i in range(len(current_coordinates)):
        for j in range(len(ban_coordinate)):

            dist = euclidian_dist(current_coordinates[i], ban_coordinate) #since we r doing it for only one bandit


            if dist < min:

                min_index = set_index_list[i]
                min = dist

    return  min_index

class Set():

    def __init__(self, index, bandit_list, adj_set):

        self.index = index
        self.bandit_list = {}

        for ban in bandit_list:
            self.bandit_list[str(ban.index)] = ban

        self.bandit_list_index = set([ban.index for ban in bandit_list])
        self.adj_sets = adj_set



    def is_ban_in(self, bandit_index):

        if bandit_index in self.bandit_list_index:
            return True
        else:
            return False

    def get_bandit(self, bandit_index):

        try:
            return self.bandit_list[str(bandit_index)]
        except KeyError:
            return "Bandit not in the Set"


    def get_ban_set(self,bandit_index, Global_Set, ban_to_set_dict):

        set_in = get_nearby_set_to_ban_1(set_index_list=self.adj_sets, bandit_index=bandit_index, ban_to_set_dict=ban_to_set_dict)

        return Global_Set[set_in]




'''


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

            Bandits[(i+1)*10 + 1],
            Bandits[(i+1)*10 - 1],
            Bandits[(i+1)*10],
            Bandits[i],
            Bandits[i-1],
            Bandits[i+1]
        ],adj_set=[

            (i+1)*10,
            (i+1)*10-1,
            (i+1)*10+1,
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


Ban_to_Set_dict = {}
for i in range(100):
    Ban_to_Set_dict[str(i)] = []
for i in range(len(Sets)):
    for j in S[i].bandit_list:
        Ban_to_Set_dict[str(j)].append(i)

print(Ban_to_Set_dict)

s = S[0]

for i in range(10):
    print(s.index)

    if s.is_ban_in(90) == False:
        s = s.get_ban_set(90,S, Ban_to_Set_dict)
    else:
        pass
'''
