import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import patient as pt
import random
import math


class mab_algorithms:

    def __init__(self):
        self.p = pt.Patient()
        self.iterations = pt.Config().iterations
        self.meds = 7
        self.med_reward = [0, 0, 0, 0, 0, 0, 0]
        self.bar = np.arange(7)

    def build_patients(self, num_of_patients):
        self.p.build_patient(num_of_patients)

    def pull_rand_or_norm(self, choice):
        pull = ''
        mu, sigma = 4, 1
        sums_of_reward = [0] * self.meds
        sums_of_regret = [0] * self.meds
        count_of_arms = [0] * self.meds
        total_reward = 0
        total_regret = 0
        patient_data = pd.read_csv('data.csv')
        for n in range(len(patient_data.index)):
            if choice == 0:#Select an arm randomly
                med = random.randrange(self.meds)
                count_of_arms[med] += 1
                pull = "Random Pull:"
            elif choice == 1:#Select an arm as per normal distribution
                med = int(np.random.normal(mu, sigma, 1)[0])
                if med > 6: med = 6
                if med < 0: med = 0
                count_of_arms[med] += 1
                pull = "Normal Dist Pull:"
            elif choice == 2:#Select an arm sequentially
                med = np.mod(n, 7)
                count_of_arms[med] += 1
                pull = "Sequential Pull:"
            reward = patient_data.values[n, med+8]
            regret = 1 - reward
            sums_of_reward[med] += reward
            sums_of_regret[med] += regret
            total_reward = total_reward + reward
            total_regret = total_regret + regret
        tr = [total_reward] * self.meds
        '''
        print('----------------- ' + pull + ' -------------------')
        print("Count:: " + str(count_of_arms))
        print("Rewards " + str(sums_of_reward))
        print("Total Rewards " + str(total_reward))
        p1 = plt.bar(self.bar, sums_of_reward, 0.35)
        p2 = plt.bar(self.bar, count_of_arms, 0.35, bottom=sums_of_reward)
        p3 = plt.plot(tr)
        plt.xticks(self.bar, ('Med1', 'Med2', 'Med3', 'Med4', 'Med5', 'Med6', 'Med7'))
        plt.yticks(np.arange(0, 550, 50))
        plt.legend((p1[0], p2[0], p3[0]), ('Rewards', 'Total Counts', 'Total Rewards'))
        plt.ylabel(pull + 'Rewards')
        plt.title(pull + 'Rewards Vs. Pulls of each arm')
        plt.show()
        '''
        return total_reward, total_regret

    def pull_ucb(self):
        count_of_arms = [0] * self.meds
        sums_of_reward = [0] * self.meds
        sums_of_regret = [0] * self.meds
        total_reward = 0
        total_regret = 0
        patient_data = pd.read_csv('data.csv')
        for n in range(len(patient_data.index)):
            med = 0
            max_upper_bound = 0
            for i in range(0, self.meds):
                if count_of_arms[i] > 0:
                    average_reward = sums_of_reward[i] / count_of_arms[i]
                    delta_i = math.sqrt(2 * math.log(n + 1) / count_of_arms[i])
                    upper_bound = average_reward + delta_i
                else:
                    upper_bound = 1e400
                if upper_bound > max_upper_bound:
                    max_upper_bound = upper_bound
                    med = i
            count_of_arms[med] += 1
            reward = patient_data.values[n, med + 8]
            regret = 1 - reward
            sums_of_reward[med] += reward
            sums_of_regret[med] += regret
            total_reward = total_reward + reward
            total_regret = total_regret + regret
        tr = [total_reward] * self.meds
        '''
        print('----------------- UCB -------------------')
        print("Count:: " + str(count_of_arms))
        print("Rewards " + str(sums_of_reward))
        print("Total Rewards " + str(total_reward))        
        p1 = plt.bar(self.bar, sums_of_reward, 0.35)
        p2 = plt.bar(self.bar, count_of_arms, 0.35, bottom=sums_of_reward)
        p3 = plt.plot(tr)
        plt.xticks(self.bar, ('Med1', 'Med2', 'Med3', 'Med4', 'Med5', 'Med6', 'Med7'))
        plt.yticks(np.arange(0, 750, 75))
        plt.legend((p1[0], p2[0], p3[0]), ('Rewards', 'Total Counts', 'Total Rewards'))
        plt.ylabel('UCB Pull:Rewards')
        plt.title('UCB Pull:Rewards Vs. Pulls of each arm')
        plt.show()
        '''
        return total_reward, total_regret


num_iterations = 100
num_patients = 100
#reward_r = np.zeros(num_iterations)
#reward_n = np.zeros(num_iterations)
#reward_s = np.zeros(num_iterations)
#reward_u = np.zeros(num_iterations)
regret_r = np.zeros(num_iterations)
regret_n = np.zeros(num_iterations)
regret_s = np.zeros(num_iterations)
regret_u = np.zeros(num_iterations)
for i in range(num_iterations):
    #Build a list of 1000 patients
    mab = mab_algorithms()
    mab.build_patients(num_patients)
    #Run the bandit randomly
    r11, r12 = mab.pull_rand_or_norm(0)
    #reward_r[i] += r11
    regret_r[i] += r12
    #Run the bandit following normal distribution
    r21, r22 = mab.pull_rand_or_norm(1)
    #reward_n[i] += r21
    regret_n[i] += r22
    #Run the bandit with arms selected sequentially
    r31, r32 = mab.pull_rand_or_norm(2)
    #reward_s[i] += r31
    regret_s[i] += r32
    #Run the UCB bandit
    r41, r42 = mab.pull_ucb()
    #reward_u[i] += r41
    regret_u[i] += r42

#plt.figure(figsize=(12, 8))

#plt.plot(reward_r, label="Random")
#plt.plot(reward_n, label="Normal")
#plt.plot(reward_s, label="Sequential")
#plt.plot(reward_u, label="UCB")

plt.plot(regret_r, label="Random")
plt.plot(regret_n, label="Normal")
plt.plot(regret_s, label="Sequential")
plt.plot(regret_u, label="UCB")
plt.xlabel("Number of Iterations")
plt.ylabel("Total Regrets")
plt.title("Performance of Bandit Algorithms")
plt.legend()
plt.show()

'''
mab = mab_algorithms()
mab.pull_ucb()
'''

