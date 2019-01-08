from __future__ import division
from math import log
import xlrd
import numpy as np
import math
import statistics
import pandas as pd
import matplotlib.pyplot as plt

def entropy(pi):
    '''
    Return the Entropy of a probability distribution:
    Entropy(S) = − Pi * log(Pi) - Pj * Log(Pj)
    Entropy(S) = − SUM (Pi * log(Pi))
    '''
    total = 0
    for p in pi:
        if sum(pi) == 0:
            break
        p = p / sum(pi)
        if p != 0:
            total += p * log(p, len(pi))
        else:
            total += 0
    total *= -1
    return total


def gain(S, A):
    '''
    Return the Information Gain:
    Gain(S, A) = Entropy(S)− SUM{for values of A} (|Si| / |S| * Entropy(Si))
    '''
    total = 0
    B = []
    for i in range(len(A)):
        if sum(A[i]) != 0:
            B.append(A[i])      # we append values in list B to make sure we are only having non-zero values
    for v in B:    # it was for v in A at the beginning
        total += sum(v) / sum(S) * entropy(v)

    gain = entropy(S) - total
    return gain

training_excel = r"C:\Users\chtv2985\Desktop\Assig3\Training_Data.xlsx"
training_data = pd.read_excel(training_excel)

HS_list = training_data["HS"].tolist()
AS_list = training_data["AS"].tolist()
HST_list = training_data["HST"].tolist()
AST_list = training_data["AST"].tolist()
HF_list = training_data["HF"].tolist()
AF_list = training_data["AF"].tolist()
HC_list = training_data["HC"].tolist()
AC_list = training_data["AC"].tolist()
HY_list = training_data["HY"].tolist()
AY_list = training_data["AY"].tolist()
HR_list = training_data["HR"].tolist()
AR_list = training_data["AR"].tolist()
FTR_list = training_data["FTR"].tolist()

# Now need to discretize each attribute in order to compute its Entropy & Gain
# And accordingly select the root node
def root_discretize(attribute):
    '''
    This function will help to decide the root node
    As it will process each column and discretize its elements into two classes according to if smaller or greater than the mean.
    Afterwards, in each of the two classes, we will count the occurrences of corresponding FTR (whether H or A)
    '''
    attribute_low = []
    attribute_high = []
    FTR_low = []
    FTR_high = []
    low_position = []
    high_position = []
    low_H_acc = 0
    low_A_acc = 0
    high_H_acc = 0
    high_A_acc = 0

    for i in range(len(attribute)):
        if attribute[i] <= statistics.mean(attribute):
            attribute_low.append(attribute[i])
            FTR_low.append(FTR_list[i])    # append the corresponding original FTR value in (FTR_low) for further use
            low_position.append(i)    # record the indexes of high category so that we can refer to corresponding values of other attributes
        else:
            attribute_high.append(attribute[i])
            FTR_high.append(FTR_list[i])   # append the corresponding original FTR value in (FTR_high) for further use
            high_position.append(i)   # record the indexes of high category so that we can refer to corresponding values of other attributes

    for j in range(len(attribute_low)):   # iterate on 1st half of attribute and count the occurrances for H & A in FTR
        if FTR_low[j] == 'H':
            low_H_acc += 1
        else:
            low_A_acc += 1

    for k in range(len(attribute_high)):   # iterate on 2nd half of attribute and count the occurrances for H & A in FTR
        if FTR_high[k] == 'H':
            high_H_acc += 1
        else:
            high_A_acc += 1

    discretized_attribute = [[low_H_acc, low_A_acc], [high_H_acc, high_A_acc]]
    return discretized_attribute, low_position, high_position

HS_discretized, HS_low_index, HS_high_index = root_discretize(HS_list)
AS_discretized, AS_low_index, AS_high_index = root_discretize(AS_list)
HST_discretized, HST_low_index, HST_high_index = root_discretize(HST_list)
AST_discretized, AST_low_index, AST_high_index = root_discretize(AST_list)
HF_discretized, HF_low_index, HF_high_index = root_discretize(HF_list)
AF_discretized, AF_low_index, AF_high_index = root_discretize(AF_list)
HC_discretized, HC_low_index, HC_high_index = root_discretize(HC_list)
AC_discretized, AC_low_index, AC_high_index = root_discretize(AC_list)
HY_discretized, HY_low_index, HY_high_index = root_discretize(HY_list)
AY_discretized, AY_low_index, AY_high_index = root_discretize(AY_list)
HR_discretized, HR_low_index, HR_high_index = root_discretize(HR_list)
AR_discretized, AR_low_index, AR_high_index = root_discretize(AR_list)


discretized_training_data = [HS_discretized, AS_discretized, HST_discretized, AST_discretized, HF_discretized,
                             AF_discretized, HC_discretized, AC_discretized, HY_discretized, AY_discretized,
                             HR_discretized, AR_discretized]

# Discretize FTR:
FTR_H = 0
FTR_A = 0
for n in range(len(FTR_list)):
    if FTR_list[n] == 'H':
        FTR_H += 1
    else:
        FTR_A += 1
FTR_discretized = [FTR_H, FTR_A]

# Calculate Gains of different attributes:
gains = [gain(FTR_discretized, HS_discretized), gain(FTR_discretized, AS_discretized), gain(FTR_discretized, HST_discretized),
         gain(FTR_discretized, AST_discretized), gain(FTR_discretized, HF_discretized), gain(FTR_discretized, AF_discretized),
         gain(FTR_discretized, HC_discretized), gain(FTR_discretized, AC_discretized), gain(FTR_discretized, HY_discretized),
         gain(FTR_discretized, AY_discretized), gain(FTR_discretized, HR_discretized), gain(FTR_discretized, AR_discretized)]

print("Max Gain of All Attributes = " + str(max(gains)))
print("Index of Max Gain is: " + str(np.argmax(np.array(gains))))   # Therefore, HST is the root node
print("Root Node = HST")



##############################################################################################################################
################################################ Determine The Two Nodes in the First Level ##################################
##############################################################################################################################

training_data['HST_Class'] = np.where(training_data['HST'] <= training_data["HST"].mean(), 'Low', 'High')   # Add new column with Title 'HST Class'

training_data_L1_updated = training_data.set_index('HST_Class')  # Copy data to new df and make HST_Class is the index

# Create Two New Dataframes based on HST Low & High:
training_data_HST_low = training_data_L1_updated.loc['Low']
training_data_HST_low = training_data_HST_low.reset_index()
training_data_HST_high = training_data_L1_updated.loc['High']
training_data_HST_high = training_data_HST_high.reset_index()

# Discretize other attributes into 3 classes based on equal spacing discretization (uniform).
HS_step = (training_data["HS"].max() - training_data["HS"].min()) / 3
AS_step = (training_data["AS"].max() - training_data["AS"].min()) / 3
AST_step = (training_data["AST"].max() - training_data["AST"].min()) / 3
HF_step = (training_data["HF"].max() - training_data["HF"].min()) / 3
AF_step = (training_data["AF"].max() - training_data["AF"].min()) / 3
HC_step = (training_data["HC"].max() - training_data["HC"].min()) / 3
AC_step = (training_data["AC"].max() - training_data["AC"].min()) / 3
HY_step = (training_data["HY"].max() - training_data["HY"].min()) / 3
AY_step = (training_data["AY"].max() - training_data["AY"].min()) / 3
HR_step = (training_data["HR"].max() - training_data["HR"].min()) / 3
AR_step = (training_data["AR"].max() - training_data["AR"].min()) / 3


def discretize_to_next_level(df, col_name, col_min, col_step):
    '''
    This function will help to discretize other attributes in first and second levels
    As it classify each attribute into 3 classes: low, medium and high
    Also, it will record the occurrences of FTR to be used in calculating entropy & gain
    '''
    low_acc = 0
    med_acc = 0
    high_acc = 0
    FTR_low_H = 0
    FTR_low_A = 0
    FTR_med_H = 0
    FTR_med_A = 0
    FTR_high_H = 0
    FTR_high_A = 0

    for index, row in df.iterrows():
        if row[col_name] <= (col_min + col_step):
            low_acc += 1
            if row['FTR'] == 'H':
                FTR_low_H += 1
            else:
                FTR_low_A += 1

        else:
            if (col_min + col_step) < row[col_name] <= (col_min + (col_step * 2)):
                med_acc += 1
                if row['FTR'] == 'H':
                    FTR_med_H += 1
                else:
                    FTR_med_A += 1

            else:
                high_acc += 1
                if row['FTR'] == 'H':
                    FTR_high_H += 1
                else:
                    FTR_high_A += 1

    positions = [low_acc, med_acc, high_acc]
    discretized_col = [[FTR_low_H, FTR_low_A], [FTR_med_H, FTR_med_A], [FTR_high_H, FTR_high_A]]
    return discretized_col, positions

# Firstly, discretize the attributes in the dataframe HST-Low:
HS_discretized_L1_low, HS_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'HS', training_data["HS"].min(), HS_step)
AS_discretized_L1_low, AS_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'AS', training_data["AS"].min(), AS_step)
AST_discretized_L1_low, AST_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'AST', training_data["AST"].min(), AST_step)
HF_discretized_L1_low, HF_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'HF', training_data["HF"].min(), HF_step)
AF_discretized_L1_low, AF_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'AF', training_data["AF"].min(), AF_step)
HC_discretized_L1_low, HC_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'HC', training_data["HC"].min(), HC_step)
AC_discretized_L1_low, AC_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'AC', training_data["AC"].min(), AC_step)
HY_discretized_L1_low, HY_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'HY', training_data["HY"].min(), HY_step)
AY_discretized_L1_low, AY_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'AY', training_data["AY"].min(), AY_step)
HR_discretized_L1_low, HR_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'HR', training_data["HR"].min(), HR_step)
AR_discretized_L1_low, AR_L1_low_positions = discretize_to_next_level(training_data_HST_low, 'AR', training_data["AR"].min(), AR_step)

print("================================= Level 1 =========================================")

# discretize HST-LOW to compute gain:
low_HST_H_counter = 0
low_HST_A_counter = 0
for index, row in training_data_HST_low.iterrows():
    if row['FTR'] == 'H':
        low_HST_H_counter += 1
    else:
        low_HST_A_counter += 1

low_HST_discretized = [low_HST_H_counter, low_HST_A_counter]

# Calculate the Gain of other attributes w.r.t HST Low:
gain_L1_low = [gain(low_HST_discretized, HS_discretized_L1_low), gain(low_HST_discretized, AS_discretized_L1_low), gain(low_HST_discretized, AST_discretized_L1_low),
               gain(low_HST_discretized, HF_discretized_L1_low), gain(low_HST_discretized, AF_discretized_L1_low),
               gain(low_HST_discretized, HC_discretized_L1_low), gain(low_HST_discretized, AC_discretized_L1_low),
               gain(low_HST_discretized, HY_discretized_L1_low), gain(low_HST_discretized, AY_discretized_L1_low),
               gain(low_HST_discretized, HR_discretized_L1_low), gain(low_HST_discretized, AR_discretized_L1_low)]

print("HST of L1-LOW Branch: " + str(low_HST_discretized))

# Get the largest gain in low branch:
print("Max Gain in L1-LOW Branch = " + str(max(gain_L1_low)))
print("Index of Max Gain is: " + str(np.argmax(np.array(gain_L1_low))))     # AST
print("L1-LOW Node = AST")

# Secondly, discretize the attributes in the dataframe HST-High:
HS_discretized_L1_high, HS_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'HS', training_data["HS"].min(), HS_step)
AS_discretized_L1_high, AS_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'AS', training_data["AS"].min(), AS_step)
AST_discretized_L1_high, AST_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'AST', training_data["AST"].min(), AST_step)
HF_discretized_L1_high, HF_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'HF', training_data["HF"].min(), HF_step)
AF_discretized_L1_high, AF_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'AF', training_data["AF"].min(), AF_step)
HC_discretized_L1_high, HC_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'HC', training_data["HC"].min(), HC_step)
AC_discretized_L1_high, AC_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'AC', training_data["AC"].min(), AC_step)
HY_discretized_L1_high, HY_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'HY', training_data["HY"].min(), HY_step)
AY_discretized_L1_high, AY_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'AY', training_data["AY"].min(), AY_step)
HR_discretized_L1_high, HR_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'HR', training_data["HR"].min(), HR_step)
AR_discretized_L1_high, AR_L1_high_positions = discretize_to_next_level(training_data_HST_high, 'AR', training_data["AR"].min(), AR_step)

print("==================================================================================")

# discretize HST-High to compute gain:
high_HST_H_counter = 0
high_HST_A_counter = 0
for index, row in training_data_HST_high.iterrows():
    if row['FTR'] == 'H':
        high_HST_H_counter += 1
    else:
        high_HST_A_counter += 1

high_HST_discretized = [high_HST_H_counter, high_HST_A_counter]

# Calculate the Gain of other attributes w.r.t HST High:
gain_L1_high = [gain(high_HST_discretized, HS_discretized_L1_high), gain(high_HST_discretized, AS_discretized_L1_high), gain(high_HST_discretized, AST_discretized_L1_high),
               gain(high_HST_discretized, HF_discretized_L1_high), gain(high_HST_discretized, AF_discretized_L1_high),
               gain(high_HST_discretized, HC_discretized_L1_high), gain(high_HST_discretized, AC_discretized_L1_high),
               gain(high_HST_discretized, HY_discretized_L1_high), gain(high_HST_discretized, AY_discretized_L1_high),
               gain(high_HST_discretized, HR_discretized_L1_high), gain(high_HST_discretized, AR_discretized_L1_high)]

print("HST of L1-HIGH Branch: " + str(high_HST_discretized))

# Get the largest gain in High branch:
print("Max Gain in L1-HIGH Branch = " + str(max(gain_L1_high)))       # HY
print("Index of Max Gain is: " + str(np.argmax(np.array(gain_L1_high))))
print("L1-High Node = HY")

print("=================================== Level 2 =========================================")


##############################################################################################################################
################################################ Determine The Other Nodes in the Second Level ###############################
##############################################################################################################################


############################################################### L2-LOW-LOW Level #############################################

# Create New Column for AST Classes (Low, Medium & High) in LOW-HST DF:
training_data_HST_low['AST_Class']=['Low' if training_data_HST_low['AST'][i] <= (training_data["AST"].min() + AST_step) else 'Medium' if (training_data["AST"].min() + AST_step) < training_data_HST_low['AST'][i] <= (training_data["AST"].min() + (2 * AST_step)) else 'High' for i in range(len(training_data_HST_low))]

training_data_HST_low_L2_updated = training_data_HST_low.set_index('AST_Class')  # Copy data to new df and make AST_Class is the index

# Create Three New Dataframes based on AST Low, Medium & High:
training_data_HST_low_AST_low = training_data_HST_low_L2_updated.loc['Low']
training_data_HST_low_AST_low = training_data_HST_low_AST_low.reset_index()

training_data_HST_low_AST_med = training_data_HST_low_L2_updated.loc['Medium']
training_data_HST_low_AST_med = training_data_HST_low_AST_med.reset_index()

training_data_HST_low_AST_high = training_data_HST_low_L2_updated.loc['High']
training_data_HST_low_AST_high = training_data_HST_low_AST_high.reset_index()

# Discretize the attributes in the dataframe HST-Low-AST-Low:
HS_discretized_L2_HST_low_AST_low, HS_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'HS', training_data["HS"].min(), HS_step)
AS_discretized_L2_HST_low_AST_low, AS_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'AS', training_data["AS"].min(), AS_step)
HF_discretized_L2_HST_low_AST_low, HF_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'HF', training_data["HF"].min(), HF_step)
AF_discretized_L2_HST_low_AST_low, AF_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'AF', training_data["AF"].min(), AF_step)
HC_discretized_L2_HST_low_AST_low, HC_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'HC', training_data["HC"].min(), HC_step)
AC_discretized_L2_HST_low_AST_low, AC_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'AC', training_data["AC"].min(), AC_step)
HY_discretized_L2_HST_low_AST_low, HY_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'HY', training_data["HY"].min(), HY_step)
AY_discretized_L2_HST_low_AST_low, AY_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'AY', training_data["AY"].min(), AY_step)
HR_discretized_L2_HST_low_AST_low, HR_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'HR', training_data["HR"].min(), HR_step)
AR_discretized_L2_HST_low_AST_low, AR_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'AR', training_data["AR"].min(), AR_step)

# discretize HST-LOW-AST-LOW branch to compute gain:
low_HST_low_AST_H_counter = 0
low_HST_low_AST_A_counter = 0
for index, row in training_data_HST_low_AST_low.iterrows():
    if row['FTR'] == 'H':
        low_HST_low_AST_H_counter += 1
    else:
        low_HST_low_AST_A_counter += 1

low_HST_low_AST_discretized = [low_HST_low_AST_H_counter, low_HST_low_AST_A_counter]

# Calculate the Gain of other attributes w.r.t HST-Low-AST-Low:
gain_L2_low_low = [gain(low_HST_low_AST_discretized, HS_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, AS_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, HF_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, AF_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, HC_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, AC_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, HY_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, AY_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, HR_discretized_L2_HST_low_AST_low),
                   gain(low_HST_low_AST_discretized, AR_discretized_L2_HST_low_AST_low)]

print("AST of L2-LOW-LOW Branch: " + str(low_HST_low_AST_discretized))

# Get the largest gain in L2-Low-low branch:
print("Max Gain in L2-LOW-LOW Branch = " + str(max(gain_L2_low_low)))
print("Index of Max Gain is: " + str(np.argmax(np.array(gain_L2_low_low))))     # AY
print("L2-LOW-LOW Node = AY")
print("The Three branches of AY are = " + str(AY_discretized_L2_HST_low_AST_low))

############################################################### L2-LOW-MEDIUM Level #############################################

# Discretize the attributes in the dataframe HST-Low-AST-Medium:
HS_discretized_L2_HST_low_AST_med, HS_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'HS', training_data["HS"].min(), HS_step)
AS_discretized_L2_HST_low_AST_med, AS_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'AS', training_data["AS"].min(), AS_step)
HF_discretized_L2_HST_low_AST_med, HF_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'HF', training_data["HF"].min(), HF_step)
AF_discretized_L2_HST_low_AST_med, AF_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'AF', training_data["AF"].min(), AF_step)
HC_discretized_L2_HST_low_AST_med, HC_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'HC', training_data["HC"].min(), HC_step)
AC_discretized_L2_HST_low_AST_med, AC_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'AC', training_data["AC"].min(), AC_step)
HY_discretized_L2_HST_low_AST_med, HY_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'HY', training_data["HY"].min(), HY_step)
AY_discretized_L2_HST_low_AST_med, AY_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'AY', training_data["AY"].min(), AY_step)
HR_discretized_L2_HST_low_AST_med, HR_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'HR', training_data["HR"].min(), HR_step)
AR_discretized_L2_HST_low_AST_med, AR_L2_HST_low_AST_med_positions = discretize_to_next_level(training_data_HST_low_AST_med, 'AR', training_data["AR"].min(), AR_step)

print("==================================================================================")

# discretize HST-LOW-AST-MEDIUM branch to compute gain:
low_HST_med_AST_H_counter = 0
low_HST_med_AST_A_counter = 0
for index, row in training_data_HST_low_AST_med.iterrows():
    if row['FTR'] == 'H':
        low_HST_med_AST_H_counter += 1
    else:
        low_HST_med_AST_A_counter += 1

low_HST_med_AST_discretized = [low_HST_med_AST_H_counter, low_HST_med_AST_A_counter]

# Calculate the Gain of other attributes w.r.t HST-Low-AST-Medium:
gain_L2_low_med = [gain(low_HST_med_AST_discretized, HS_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, AS_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, HF_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, AF_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, HC_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, AC_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, HY_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, AY_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, HR_discretized_L2_HST_low_AST_med),
                   gain(low_HST_med_AST_discretized, AR_discretized_L2_HST_low_AST_med)]

print("AST of L2-LOW-MEDIUM Branch: " + str(low_HST_med_AST_discretized))

# Get the largest gain in L2-low-Medium branch:
print("Max Gain in L2-LOW-MEDIUM Branch = " + str(max(gain_L2_low_med)))
print("Index of Max Gain is: " + str(np.argmax(np.array(gain_L2_low_med))))    # AF
print("L2-LOW-MEDIUM Node = AF")
print("The Three Branches of AF are = " + str(AF_discretized_L2_HST_low_AST_med))
print("==================================================================================")

############################################################### L2-LOW-HIGH Level #############################################

# discretize HST-LOW-AST-HIGH branch to compute gain:
low_HST_high_AST_H_counter = 0
low_HST_high_AST_A_counter = 0
for index, row in training_data_HST_low_AST_high.iterrows():
    if row['FTR'] == 'H':
        low_HST_high_AST_H_counter += 1
    else:
        low_HST_high_AST_A_counter += 1

low_HST_high_AST_discretized = [low_HST_high_AST_H_counter, low_HST_high_AST_A_counter]

print("AST of L2-LOW-HIGH Branch: " + str(low_HST_high_AST_discretized))
print("Therefore, A Team wins!")


############################################################### L2-HIGH-LOW Level #############################################

# Create New Column for HY Classes (Low, Medium & High) in HIGH-HST DF:
training_data_HST_high['HY_Class']=['Low' if training_data_HST_high['HY'][i] <= (training_data["HY"].min() + HY_step) else 'Medium' if (training_data["HY"].min() + HY_step) < training_data_HST_high['HY'][i] <= (training_data["HY"].min() + (2 * HY_step)) else 'High' for i in range(len(training_data_HST_high))]

training_data_HST_high_L2_updated = training_data_HST_high.set_index('HY_Class')  # Copy data to new df and make HY_Class is the index

# Create Three New Dataframes based on HY Low, Medium & High:
training_data_HST_high_HY_low = training_data_HST_high_L2_updated.loc['Low']
training_data_HST_high_HY_low = training_data_HST_high_HY_low.reset_index()

training_data_HST_high_HY_med = training_data_HST_high_L2_updated.loc['Medium']
training_data_HST_high_HY_med = training_data_HST_high_HY_med.reset_index()

training_data_HST_high_HY_high = training_data_HST_high_L2_updated.loc['High']
training_data_HST_high_HY_high = training_data_HST_high_HY_high.reset_index()

# Discretize the attributes in the dataframe HST-High-HY-Low:
HS_discretized_L2_HST_high_HY_low, HS_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'HS', training_data["HS"].min(), HS_step)
AS_discretized_L2_HST_high_HY_low, AS_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'AS', training_data["AS"].min(), AS_step)
HF_discretized_L2_HST_high_HY_low, HF_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'HF', training_data["HF"].min(), HF_step)
AF_discretized_L2_HST_high_HY_low, AF_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'AF', training_data["AF"].min(), AF_step)
HC_discretized_L2_HST_high_HY_low, HC_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'HC', training_data["HC"].min(), HC_step)
AC_discretized_L2_HST_high_HY_low, AC_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'AC', training_data["AC"].min(), AC_step)
#HY_discretized_L2_HST_high_HY_low, HY_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'HY', training_data["HY"].min(), HY_step)
AY_discretized_L2_HST_high_HY_low, AY_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'AY', training_data["AY"].min(), AY_step)
HR_discretized_L2_HST_high_HY_low, HR_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'HR', training_data["HR"].min(), HR_step)
AR_discretized_L2_HST_high_HY_low, AR_L2_HST_high_HY_low_positions = discretize_to_next_level(training_data_HST_high_HY_low, 'AR', training_data["AR"].min(), AR_step)

print("==================================================================================")

# discretize HST-HIGH-HY-LOW branch to compute gain:
high_HST_low_HY_H_counter = 0
high_HST_low_HY_A_counter = 0
for index, row in training_data_HST_high_HY_low.iterrows():
    if row['FTR'] == 'H':
        high_HST_low_HY_H_counter += 1
    else:
        high_HST_low_HY_A_counter += 1

high_HST_low_HY_discretized = [high_HST_low_HY_H_counter, high_HST_low_HY_A_counter]

# Calculate the Gain of other attributes w.r.t HST-High-HY-Low:
gain_L2_high_low = [gain(high_HST_low_HY_discretized, HS_discretized_L2_HST_high_HY_low),
                   gain(high_HST_low_HY_discretized, AS_discretized_L2_HST_high_HY_low),
                   gain(high_HST_low_HY_discretized, HF_discretized_L2_HST_high_HY_low),
                   gain(high_HST_low_HY_discretized, AF_discretized_L2_HST_high_HY_low),
                   gain(high_HST_low_HY_discretized, HC_discretized_L2_HST_high_HY_low),
                   gain(high_HST_low_HY_discretized, AC_discretized_L2_HST_high_HY_low),
                   gain(high_HST_low_HY_discretized, AY_discretized_L2_HST_high_HY_low),
                   gain(high_HST_low_HY_discretized, HR_discretized_L2_HST_high_HY_low),
                   gain(high_HST_low_HY_discretized, AR_discretized_L2_HST_high_HY_low)]

print("HY of L2-HIGH-LOW Branch: " + str(high_HST_low_HY_discretized))

# Get the largest gain in L2-High-low branch:
print("Max Gain in L2-HIGH-LOW Branch = " + str(max(gain_L2_high_low)))
print("Index of Max Gain is: " + str(np.argmax(np.array(gain_L2_high_low))))   # AC
print("H2-High-Low Node = AC")
print("The Three Branches of AC are = " + str(AC_discretized_L2_HST_high_HY_low))

print("==================================================================================")

############################################################### L2-HIGH-MEDIUM Level #############################################

# Discretize the attributes in the dataframe HST-High-HY-Medium:
HS_discretized_L2_HST_high_HY_med, HS_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'HS', training_data["HS"].min(), HS_step)
AS_discretized_L2_HST_high_HY_med, AS_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'AS', training_data["AS"].min(), AS_step)
HF_discretized_L2_HST_high_HY_med, HF_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'HF', training_data["HF"].min(), HF_step)
AF_discretized_L2_HST_high_HY_med, AF_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'AF', training_data["AF"].min(), AF_step)
HC_discretized_L2_HST_high_HY_med, HC_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'HC', training_data["HC"].min(), HC_step)
AC_discretized_L2_HST_high_HY_med, AC_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'AC', training_data["AC"].min(), AC_step)
#HY_discretized_L2_HST_high_HY_low, HY_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'HY', training_data["HY"].min(), HY_step)
AY_discretized_L2_HST_high_HY_med, AY_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'AY', training_data["AY"].min(), AY_step)
HR_discretized_L2_HST_high_HY_med, HR_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'HR', training_data["HR"].min(), HR_step)
AR_discretized_L2_HST_high_HY_med, AR_L2_HST_high_HY_med_positions = discretize_to_next_level(training_data_HST_high_HY_med, 'AR', training_data["AR"].min(), AR_step)

# discretize HST-HIGH-HY-MEDIUM branch to compute gain:
high_HST_med_HY_H_counter = 0
high_HST_med_HY_A_counter = 0
for index, row in training_data_HST_high_HY_med.iterrows():
    if row['FTR'] == 'H':
        high_HST_med_HY_H_counter += 1
    else:
        high_HST_med_HY_A_counter += 1

high_HST_med_HY_discretized = [high_HST_med_HY_H_counter, high_HST_med_HY_A_counter]

# Calculate the Gain of other attributes w.r.t HST-High-HY-Medium:
gain_L2_high_med = [gain(high_HST_med_HY_discretized, HS_discretized_L2_HST_high_HY_med),
                   gain(high_HST_med_HY_discretized, AS_discretized_L2_HST_high_HY_med),
                   gain(high_HST_med_HY_discretized, HF_discretized_L2_HST_high_HY_med),
                   gain(high_HST_med_HY_discretized, AF_discretized_L2_HST_high_HY_med),
                   gain(high_HST_med_HY_discretized, HC_discretized_L2_HST_high_HY_med),
                   gain(high_HST_med_HY_discretized, AC_discretized_L2_HST_high_HY_med),
                   gain(high_HST_med_HY_discretized, AY_discretized_L2_HST_high_HY_med),
                   gain(high_HST_med_HY_discretized, HR_discretized_L2_HST_high_HY_med),
                   gain(high_HST_med_HY_discretized, AR_discretized_L2_HST_high_HY_med)]

print("HY of L2-HIGH-MEDIUM Branch: " + str(high_HST_med_HY_discretized))

# Get the largest gain in L2-High-Medium branch:
print("Max Gain in L2-HIGH-MEDIUM Branch = " + str(max(gain_L2_high_med)))
print("Index of Max Gain is: " + str(np.argmax(np.array(gain_L2_high_med))))    # HF
print("L2-HIGH-MEDIUM Node = HF")
print("The Three Branches of HF are = " + str(HF_discretized_L2_HST_high_HY_med))

print("==================================================================================")

############################################################### L2-HIGH-HIGH Level #############################################

# Discretize the attributes in the dataframe HST-High-HY-High:
HS_discretized_L2_HST_high_HY_high, HS_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'HS', training_data["HS"].min(), HS_step)
AS_discretized_L2_HST_high_HY_high, AS_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'AS', training_data["AS"].min(), AS_step)
HF_discretized_L2_HST_high_HY_high, HF_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'HF', training_data["HF"].min(), HF_step)
AF_discretized_L2_HST_high_HY_high, AF_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'AF', training_data["AF"].min(), AF_step)
HC_discretized_L2_HST_high_HY_high, HC_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'HC', training_data["HC"].min(), HC_step)
AC_discretized_L2_HST_high_HY_high, AC_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'AC', training_data["AC"].min(), AC_step)
#HY_discretized_L2_HST_high_HY_low, HY_L2_HST_low_AST_low_positions = discretize_to_next_level(training_data_HST_low_AST_low, 'HY', training_data["HY"].min(), HY_step)
AY_discretized_L2_HST_high_HY_high, AY_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'AY', training_data["AY"].min(), AY_step)
HR_discretized_L2_HST_high_HY_high, HR_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'HR', training_data["HR"].min(), HR_step)
AR_discretized_L2_HST_high_HY_high, AR_L2_HST_high_HY_high_positions = discretize_to_next_level(training_data_HST_high_HY_high, 'AR', training_data["AR"].min(), AR_step)

# discretize HST-HIGH-HY-HIGH branch to compute gain:
high_HST_high_HY_H_counter = 0
high_HST_high_HY_A_counter = 0
for index, row in training_data_HST_high_HY_high.iterrows():
    if row['FTR'] == 'H':
        high_HST_high_HY_H_counter += 1
    else:
        high_HST_high_HY_A_counter += 1

high_HST_high_HY_discretized = [high_HST_high_HY_H_counter, high_HST_high_HY_A_counter]

# Calculate the Gain of other attributes w.r.t HST-High-HY-High:
gain_L2_high_high = [gain(high_HST_high_HY_discretized, HS_discretized_L2_HST_high_HY_high),
                   gain(high_HST_high_HY_discretized, AS_discretized_L2_HST_high_HY_high),
                   gain(high_HST_high_HY_discretized, HF_discretized_L2_HST_high_HY_high),
                   gain(high_HST_high_HY_discretized, AF_discretized_L2_HST_high_HY_high),
                   gain(high_HST_high_HY_discretized, HC_discretized_L2_HST_high_HY_high),
                   gain(high_HST_high_HY_discretized, AC_discretized_L2_HST_high_HY_high),
                   gain(high_HST_high_HY_discretized, AY_discretized_L2_HST_high_HY_high),
                   gain(high_HST_high_HY_discretized, HR_discretized_L2_HST_high_HY_high),
                   gain(high_HST_high_HY_discretized, AR_discretized_L2_HST_high_HY_high)]

print("HY of L2-HIGH-HIGH Branch: " + str(high_HST_high_HY_discretized))

# Get the largest gain in L2-High-High branch:
print("Max Gain in L2-HIGH-HIGH Branch = " + str(max(gain_L2_high_high)))
print("Index of Max Gain is: " + str(np.argmax(np.array(gain_L2_high_high))))      # AY
print("L2-HIGH-HIGH Node = AY")
print("The Three Branches of AY are = " + str(AY_discretized_L2_HST_high_HY_high))
print("==================================================================================")


################################################################################################################################################
################################################# Applying Decision Tree on Test Data (Liverpool) ##############################################
################################################################################################################################################

HST_threshold = training_data['HST'].mean()
# Low Path:
AST_thresholds = [(training_data['AST'].min() + AST_step), (training_data['AST'].min() + (2 * AST_step))]
AY_thresholds = [(training_data['AY'].min() + AY_step), (training_data['AY'].min() + (2 * AY_step))]
AF_thresholds = [(training_data['AF'].min() + AF_step), (training_data['AF'].min() + (2 * AF_step))]
# High Path:
HY_thresholds = [(training_data['HY'].min() + HY_step), (training_data['HY'].min() + (2 * HY_step))]
HF_thresholds = [(training_data['HF'].min() + HF_step), (training_data['HF'].min() + (2 * HF_step))]

test_excel = r"C:\Users\chtv2985\Desktop\Assig3\Liverpool.xlsx"
test_data = pd.read_excel(test_excel)

print("Liverpool Test Data:")
print((test_data.head()))
print('\n')

actual_test_FTR = test_data['FTR'].tolist()
predicted_test_FTR = []

for i in range(len(test_data)):
    # Low Path:
    if (test_data['HST'][i] <= HST_threshold) and (test_data['AST'][i] <= AST_thresholds[0]):
        predicted_test_FTR.append('H')
    elif (test_data['HST'][i] <= HST_threshold) and (test_data['AST'][i] > AST_thresholds[0]):
        predicted_test_FTR.append('A')
    # High Path:
    elif (test_data['HST'][i] > HST_threshold) and (test_data['HY'][i] <= HY_thresholds[0]):
        predicted_test_FTR.append('H')
    elif (test_data['HST'][i] > HST_threshold) and (HY_thresholds[0] < test_data['HY'][i] <= HY_thresholds[1]) and (test_data['HF'][i] <= HF_thresholds[1]):
        predicted_test_FTR.append('H')
    elif (test_data['HST'][i] > HST_threshold) and (test_data['HY'][i] > HY_thresholds[1]) and (test_data['AY'][i] <= AY_thresholds[0]):
        predicted_test_FTR.append('H')
    else:
        predicted_test_FTR.append('A')

# Test Accuracy:
correct_prediction = 0
wrong_prediction = 0
wrong_prediction_positions = []
for j in range(len(predicted_test_FTR)):
    if predicted_test_FTR[j] == actual_test_FTR[j]:
        correct_prediction += 1
    else:
        wrong_prediction += 1
        wrong_prediction_positions.append(j)

print("Actual   : " + str(actual_test_FTR))
print("Predicted: " + str(predicted_test_FTR))
print("Indexes of Worng Predictions: " + str(wrong_prediction_positions[0]) + ', ' + str(wrong_prediction_positions[1]) + ' & ' + str(wrong_prediction_positions[2]))
print("Accuracy = " + str((correct_prediction/len(predicted_test_FTR)*100)) + "%")
print('\n')

# Calculate Confusion Matrix:
y_actu = pd.Series(actual_test_FTR, name='Actual')
y_pred = pd.Series(predicted_test_FTR, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

# Plot Confusion Matrix:
def plot_confusion_matrix(df_confusion, title='Confusion Matrix (Liverpool Test Data)', cmap=plt.cm.Blues):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion)
plt.show()