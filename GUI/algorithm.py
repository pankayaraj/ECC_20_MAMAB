import numpy as np


#for env with time delay UCB

#for env with time delay ER


#for env without time delay UCB

from UCB.Without_Time_Delay.algo import UCB_wtd

#for env without time delay ER

from ER.Without_Time_Delay.algo import ER_wtd




def algorithm_main(no_iterations, no_agents,  no_bandits, no_experiments, mean, variance,  algorithm_type = "UCB", enviornment_type = "Without Time Delay"):

    if enviornment_type == "Without Time Delay":
        if algorithm_type == "UCB":
            return UCB_wtd(no_iterations, no_agents, no_bandits, no_experiments, mean, variance)

        if algorithm_type == "ER":
            return ER_wtd(no_iterations, no_agents, no_bandits, no_experiments, mean, variance)

'''
mean = [0, 1, 4, 5]
variance = [2, 2, 2, 2]
no_agents = 2
no_bandits = 4
no_experiments = 1
no_iterations  = 10

print(algorithm_main(no_iterations, no_agents,  no_bandits, no_experiments, mean, variance, algorithm_type = "ER", enviornment_type = "Without Time Delay"))
'''
