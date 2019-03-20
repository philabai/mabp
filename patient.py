import numpy as np
import random
import os


class Config:
    def __init__(self):
        '''
        Simulated dataset
        Country of patient visit    -> [0]
        Oracle-Patient has Malaria? -> [1] with Yes/No values
        Drug Resistance-Chloroquine -> [2]
        Species of Malaria :: label -> [array element = probability of parasite]
            Plasmodium vivax        -> [3]
            Plasmodium falciparum   -> [4]
            Plasmodium ovale        -> [5]
            Plasmodium malariae     -> [6]
            Plasmodium knowlesi     -> [7]

        Number of Arms              -> K=7 (for each of the medications listed below)
        Recommended Drugs :: drug reward -> [array element = reward of administering drug 0/-1,
                                            following Bernoulli distribution]
            Atovaquone Proguanil    -> [8]  is 0 if given to right patient & -1 otherwise
            Doxycycline             -> [9]
            Mefloquine              -> [10]
            Chloroquine             -> [11]
            Primaquine              -> [12]
            Tafenoquine             -> [13]
            No Medication           -> [14]

        How Rewards work?

            If patient comes from AF & has Malaria      ->
            If patient comes from AF & has no Malaria   ->
        '''

        # Features for patient visited Afghanistan with malaria
        self.my_af = ['AF', 'Yes', 'Chloroquine', .8, .2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        # Features for patient visited Afghanistan without malaria
        self.mn_af = ['AF', 'No', 'Chloroquine', .8, .2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # Features for patient visited Angola with malaria
        self.my_ag = ['AG', 'Yes', 'Chloroquine', .05, .9, .05, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        # Features for patient visited Angola without malaria
        self.mn_ag = ['AG', 'No', 'Chloroquine', .05, .9, .05, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # Features for patient visited Bangladesh with malaria
        self.my_bn = ['BN', 'Yes', 'Chloroquine', .1, .9, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        # Features for patient visited Bangladesh without malaria
        self.mn_bn = ['BN', 'No', 'Chloroquine', .1, .9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # Features for patient visited Bolivia with malaria
        self.my_bv = ['BV', 'Yes', 'Chloroquine', .93, .07, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        # Features for patient visited Bolivia without malaria
        self.mn_bv = ['BV', 'No', 'Chloroquine', .93, .07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # Features for patient visited Uganda with malaria
        self.my_ug = ['UG', 'Yes', 'Chloroquine', .05, .85, .05, .05, 0, 1, 1, 1, 0, 0, 0, 0]
        # Features for patient visited Uganda without malaria
        self.mn_ug = ['UG', 'No', 'Chloroquine', .05, .85, .05, .05, 0, 0, 0, 0, 0, 0, 0, 1]
        # Features for patient from other countries without Malaria (about 100 countries)
        self.mn_ot = ['Others', 'No', 'None', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # Assumption is that 50% of countries worldwide are Malaria endemic
        self.country_malaria_split = 50
        self.num_country_with_malaria = 5
        self.risk_number = 10
        self.num_of_patients = 1000
        self.iterations = 100


class Patient:

    def __init__(self):
        self.patient = []
        self.batch_of_patients = []
        self.config = Config()

    def build_patient(self, num):
        for i in range(num):
            n = random.randint(1, 100)
            country_choice = random.randint(1, self.config.num_country_with_malaria)
            risk_number = random.randint(1, self.config.risk_number)
            if n > self.config.country_malaria_split:
                self.batch_of_patients.append(self.config.mn_ot)
            else:
                if country_choice == 1:
                    if risk_number <= 6:
                        self.batch_of_patients.append(self.config.my_af)
                    else:
                        self.batch_of_patients.append(self.config.mn_af)
                if country_choice == 2:
                    if risk_number <= 8:
                        self.batch_of_patients.append(self.config.my_ag)
                    else:
                        self.batch_of_patients.append(self.config.mn_ag)
                if country_choice == 3:
                    if risk_number <= 4:
                        self.batch_of_patients.append(self.config.my_bn)
                    else:
                        self.batch_of_patients.append(self.config.mn_bn)
                if country_choice == 4:
                    if risk_number <= 6:
                        self.batch_of_patients.append(self.config.my_bv)
                    else:
                        self.batch_of_patients.append(self.config.mn_bv)
                if country_choice == 5:
                    if risk_number <= 8:
                        self.batch_of_patients.append(self.config.my_ug)
                    else:
                        self.batch_of_patients.append(self.config.mn_ug)
        file_name = "data.csv"
        ## If file exists, delete it ##
        if os.path.isfile(file_name):
            os.remove(file_name)
        else:  ## Show an error ##
            print("Error: %s file not found" % file_name)
        np.savetxt(file_name, self.batch_of_patients, delimiter=",", fmt="%s")


p = Patient()
p.build_patient(1000)


