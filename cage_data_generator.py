from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import statistics
from matplotlib.lines import Line2D
EXP_NO_CAGE_L = [62.50,71.88,79.17,87.50,95.83]         #all minute 30
EXP_NO_CAGE_C =	[0.00,8.33,14.58,21.88,29.17]
EXP_NO_CAGE_R = [0.00,3.13,6.25,9.38,12.50]
CAGE_data = pd.read_csv("C:\\Users\\vstok\\Desktop\\cage.csv")

min_L_CAGE = CAGE_data.loc[:,'min_L']
q_one_L_CAGE = CAGE_data.loc[:,'q_one_L']
median_L_CAGE = CAGE_data.loc[:,'median_L']
q_three_L_CAGE = CAGE_data.loc[:,'q_three_L']
max_L_CAGE = CAGE_data.loc[:,'max_L']

min_C_CAGE = CAGE_data.loc[:,'min_C']
q_one_C_CAGE = CAGE_data.loc[:,'q_one_C']
median_C_CAGE = CAGE_data.loc[:,'median_C']
q_three_C_CAGE = CAGE_data.loc[:,'q_three_C']
max_C_CAGE = CAGE_data.loc[:,'max_C']

min_R_CAGE = CAGE_data.loc[:,'min_R']
q_one_R_CAGE = CAGE_data.loc[:,'q_one_R']
median_R_CAGE = CAGE_data.loc[:,'median_R']
q_three_R_CAGE = CAGE_data.loc[:,'q_three_R']
max_R_CAGE = CAGE_data.loc[:,'max_R']

EXP_minute_zero_L	= [min_L_CAGE[0],q_one_L_CAGE[0],median_L_CAGE[0],q_three_L_CAGE[0],max_L_CAGE[0]]
EXP_minute_fifteen_L = [min_L_CAGE[15],q_one_L_CAGE[15],median_L_CAGE[15],q_three_L_CAGE[15],max_L_CAGE[15]]
EXP_minute_thirty_L = [min_L_CAGE[30],q_one_L_CAGE[30],median_L_CAGE[30],q_three_L_CAGE[30],max_L_CAGE[30]]

EXP_minute_zero_C	= [min_C_CAGE[0],q_one_C_CAGE[0],median_C_CAGE[0],q_three_C_CAGE[0],max_C_CAGE[0]]
EXP_minute_fifteen_C = [min_C_CAGE[15],q_one_C_CAGE[15],median_C_CAGE[15],q_three_C_CAGE[15],max_C_CAGE[15]]
EXP_minute_thirty_C = [min_C_CAGE[30],q_one_C_CAGE[30],median_C_CAGE[30],q_three_C_CAGE[30],max_C_CAGE[30]]

EXP_minute_zero_R	= [min_R_CAGE[0],q_one_R_CAGE[0],median_R_CAGE[0],q_three_R_CAGE[0],max_R_CAGE[0]]
EXP_minute_fifteen_R = [min_R_CAGE[15],q_one_R_CAGE[15],median_R_CAGE[15],q_three_R_CAGE[15],max_R_CAGE[15]]
EXP_minute_thirty_R = [min_R_CAGE[30],q_one_R_CAGE[30],median_R_CAGE[30],q_three_R_CAGE[30],max_R_CAGE[30]]

#######_EXPERIMENT_DATA_#######################################
NOISE_CAGE_DATA_0 = pd.read_csv("cage0.csv")
NOISE_CAGE_DATA_1 = pd.read_csv("cage1.csv")
NOISE_CAGE_DATA_2 = pd.read_csv("cage2.csv")
NOISE_CAGE_DATA_3 = pd.read_csv("cage3.csv")
NOISE_CAGE_DATA_4 = pd.read_csv("cage4.csv")
NOISE_CAGE_DATA_5 = pd.read_csv("cage5.csv")
NOISE_CAGE_DATA_6 = pd.read_csv("cage6.csv")
NOISE_CAGE_DATA_7 = pd.read_csv("cage7.csv")
NOISE_CAGE_DATA_8 = pd.read_csv("cage8.csv")
NOISE_CAGE_DATA_9 = pd.read_csv("cage9.csv")
NOISE_CAGE_DATA_10 = pd.read_csv("cage10.csv")
NOISE_CAGE_DATA_11 = pd.read_csv("cage11.csv")
NOISE_CAGE_DATA_12 = pd.read_csv("cage12.csv")
NOISE_CAGE_DATA_13 = pd.read_csv("cage13.csv")
NOISE_CAGE_DATA_14 = pd.read_csv("cage14.csv")
NOISE_CAGE_DATA_15 = pd.read_csv("cage15.csv")
NOISE_CAGE_DATA_16 = pd.read_csv("cage16.csv")
NOISE_CAGE_DATA_17 = pd.read_csv("cage17.csv")
NOISE_CAGE_DATA_18 = pd.read_csv("cage18.csv")
NOISE_CAGE_DATA_19 = pd.read_csv("cage19.csv")
#NOISE_CAGE_DATA_20 = pd.read_csv("cage20.csv")

MIN_L = []
Q1_L = []
MEDIAN_L = []
Q3_L = []
MAX_L = []

MIN_C = []
Q1_C = []
MEDIAN_C = []
Q3_C = []
MAX_C = []

MIN_R = []
Q1_R = []
MEDIAN_R = []
Q3_R = []
MAX_R = []

for i in range(0,len(NOISE_CAGE_DATA_0.loc[:,'L'])):
    SAMPLE = []
    SAMPLE.append((NOISE_CAGE_DATA_0.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_1.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_2.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_3.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_4.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_5.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_6.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_7.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_8.loc[i,'L']))
    SAMPLE.append((NOISE_CAGE_DATA_9.loc[i,'L']))

    q1, median, q3= np.percentile(SAMPLE, [25,50,75])
    min, max = np.min(SAMPLE), np.max(SAMPLE)
    MIN_L.append(min)
    Q1_L.append(q1)
    MEDIAN_L.append(median)
    Q3_L.append(q3)
    MAX_L.append(max)

for i in range(0,len(NOISE_CAGE_DATA_0.loc[:,'C'])):
    SAMPLE = []
    SAMPLE.append((NOISE_CAGE_DATA_0.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_1.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_2.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_3.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_4.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_5.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_6.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_7.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_8.loc[i,'C']))
    SAMPLE.append((NOISE_CAGE_DATA_9.loc[i,'C']))
    q1, median, q3= np.percentile(SAMPLE, [25,50,75])
    min, max = np.min(SAMPLE), np.max(SAMPLE)
    MIN_C.append(min)
    Q1_C.append(q1)
    MEDIAN_C.append(median)
    Q3_C.append(q3)
    MAX_C.append(max)

for i in range(0,len(NOISE_CAGE_DATA_0.loc[:,'R'])):
    SAMPLE = []
    SAMPLE.append((NOISE_CAGE_DATA_0.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_1.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_2.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_3.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_4.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_5.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_6.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_7.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_8.loc[i,'R']))
    SAMPLE.append((NOISE_CAGE_DATA_9.loc[i,'R']))
    q1, median, q3= np.percentile(SAMPLE, [25,50,75])
    min, max = np.min(SAMPLE), np.max(SAMPLE)
    MIN_R.append(min)
    Q1_R.append(q1)
    MEDIAN_R.append(median)
    Q3_R.append(q3)
    MAX_R.append(max)

MOD_minute_zero_L	= [MIN_L[0],Q1_R[0],MEDIAN_L[0],Q3_L[0],MAX_L[0]]
MOD_minute_fifteen_L = [MIN_L[15],Q1_R[15],MEDIAN_L[15],Q3_L[15],MAX_L[15]]
MOD_minute_thirty_L = [MIN_L[30],Q1_R[30],MEDIAN_L[30],Q3_L[30],MAX_L[30]]

MOD_minute_zero_C	= [MIN_C[0],Q1_R[0],MEDIAN_C[0],Q3_C[0],MAX_C[0]]
MOD_minute_fifteen_C = [MIN_C[15],Q1_R[15],MEDIAN_C[15],Q3_C[15],MAX_C[15]]
MOD_minute_thirty_C = [MIN_C[30],Q1_R[30],MEDIAN_C[30],Q3_C[30],MAX_C[30]]

MOD_minute_zero_R	= [MIN_R[0],Q1_R[0],MEDIAN_R[0],Q3_R[0],MAX_R[0]]
MOD_minute_fifteen_R = [MIN_R[15],Q1_R[15],MEDIAN_R[15],Q3_R[15],MAX_R[15]]
MOD_minute_thirty_R = [MIN_R[30],Q1_R[30],MEDIAN_R[30],Q3_R[30],MAX_R[30]]

###################### plot #################

# data_L = [EXP_minute_thirty_L, MOD_minute_thirty_L]
# data_C = [EXP_minute_thirty_C, MOD_minute_thirty_C]
# data_R = [EXP_minute_thirty_R, MOD_minute_thirty_R]

data_EXP = [EXP_minute_thirty_L,EXP_minute_thirty_C,EXP_minute_thirty_R]
data_MOD = [MOD_minute_thirty_L,MOD_minute_thirty_C,MOD_minute_thirty_R]

ticks = ['L', 'C', 'R']

def set_box_color_EXP(bp, color):
    plt.setp(bp['boxes'], color="black")
    plt.setp(bp['whiskers'], color="black")
    plt.setp(bp['caps'], color="black")
    plt.setp(bp['medians'], color="red")

def set_box_color_MOD(bp, color):
    plt.setp(bp['boxes'], color="black")
    plt.setp(bp['whiskers'], color="black")
    plt.setp(bp['caps'], color="black")
    plt.setp(bp['medians'], color="blue")

bpl_EXP = plt.boxplot(data_EXP, positions=[-0.5,1.5,3.5], sym='x', widths=0.8, whis=20)
bpl_MOD = plt.boxplot(data_MOD, positions=[0.5,2.5,4.5], sym='x', widths=0.8, whis=20)
#bpl_R = plt.boxplot(data_R, positions=[3.5,4.5], sym='x', widths=0.8, whis=20)
set_box_color_EXP(bpl_EXP, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color_MOD(bpl_MOD, '#D7191C') # colors are from http://colorbrewer2.org/
plt.legend()
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-1, 101)
plt.xlabel('time[min]')
plt.ylabel('median fraction of bees[%]')
plt.tight_layout()
plt.show()

'''#####################################################################################################################################'''

data_EXP = [EXP_minute_thirty_L,EXP_minute_thirty_C,EXP_minute_thirty_R]
data_EXP_NoC = [EXP_NO_CAGE_L,EXP_NO_CAGE_C,EXP_NO_CAGE_R]

ticks = ['L', 'C', 'R']

def set_box_color_EXP(bp, color):
    plt.setp(bp['boxes'], color="black", linewidth=1.3)
    plt.setp(bp['whiskers'], color="black", linewidth=1.3)
    plt.setp(bp['caps'], color="black", linewidth=1.3)
    plt.setp(bp['medians'], color="red", linewidth=1.5)

def set_box_color_EXP_NoC(bp, color):
    plt.setp(bp['boxes'], color="black", linewidth=1.3)
    plt.setp(bp['whiskers'], color="black", linewidth=1.3)
    plt.setp(bp['caps'], color="black", linewidth=1.3)
    plt.setp(bp['medians'], color="blue", linewidth=1.5)

EXP_NoC = [-0.4,1.6,3.6]
EXP_wC = [0.4,2.4,4.4]


legend_elements = [Line2D([0], [0], color='r', lw=4, label='w/o social stimulus'),Line2D([0], [0], color='b', lw=4, label='w social stimulus')]
plt.legend(handles=legend_elements, loc='upper center')

bpl_EXP_NoC = plt.boxplot(data_EXP_NoC, positions=EXP_NoC, sym='x', widths=0.6, whis=20)
bpl_EXP = plt.boxplot(data_EXP, positions=EXP_wC, sym='x', widths=0.6, whis=20)
# bpl_EXP_NoC = plt.boxplot(data_EXP_NoC, positions=[-0.5,1.5,3.5], sym='x', widths=0.8, whis=20)
# bpl_EXP = plt.boxplot(data_EXP, positions=[0.5,2.5,4.5], sym='x', widths=0.8, whis=20)
#bpl_R = plt.boxplot(data_R, positions=[3.5,4.5], sym='x', widths=0.8, whis=20)
set_box_color_EXP(bpl_EXP_NoC, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color_EXP_NoC(bpl_EXP, '#D7191C') # colors are from http://colorbrewer2.org/


plt.yticks(fontsize=14)
plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=16)
plt.xlim(-1, len(ticks)*2-1)
plt.ylim(-1, 101)
plt.xlabel('zone', fontsize=16)
plt.ylabel('median fraction of bees [%]', fontsize=16)
plt.tight_layout()
plt.show()
