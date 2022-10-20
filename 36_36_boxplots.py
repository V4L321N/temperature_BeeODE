import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

new_Data_36_36 = pd.read_csv("C:\\Users\\vstok\\Desktop\\36_36.csv")

min_L_36_36 = new_Data_36_36.loc[:,'min_L']
q_one_L_36_36 = new_Data_36_36.loc[:,'q_one_L']
median_L_36_36 = new_Data_36_36.loc[:,'median_L']
q_three_L_36_36 = new_Data_36_36.loc[:,'q_three_L']
max_L_36_36 = new_Data_36_36.loc[:,'max_L']

min_C_36_36 = new_Data_36_36.loc[:,'min_C']
q_one_C_36_36 = new_Data_36_36.loc[:,'q_one_C']
median_C_36_36 = new_Data_36_36.loc[:,'median_C']
q_three_C_36_36 = new_Data_36_36.loc[:,'q_three_C']
max_C_36_36 = new_Data_36_36.loc[:,'max_C']

min_R_36_36 = new_Data_36_36.loc[:,'min_R']
q_one_R_36_36 = new_Data_36_36.loc[:,'q_one_R']
median_R_36_36 = new_Data_36_36.loc[:,'median_R']
q_three_R_36_36 = new_Data_36_36.loc[:,'q_three_R']
max_R_36_36 = new_Data_36_36.loc[:,'max_R']

minute_zero_L	= [min_L_36_36[0]/100,q_one_L_36_36[0]/100,median_L_36_36[0]/100,q_three_L_36_36[0]/100,max_L_36_36[0]/100]
minute_fifteen_L = [min_L_36_36[15]/100,q_one_L_36_36[15]/100,median_L_36_36[15]/100,q_three_L_36_36[15]/100,max_L_36_36[15]/100]
minute_thirty_L = [min_L_36_36[30]/100,q_one_L_36_36[30]/100,median_L_36_36[30]/100,q_three_L_36_36[30]/100,max_L_36_36[30]/100]

minute_zero_C	= [min_C_36_36[0]/100,q_one_C_36_36[0]/100,median_C_36_36[0]/100,q_three_C_36_36[0]/100,max_C_36_36[0]/100]
minute_fifteen_C = [min_C_36_36[15]/100,q_one_C_36_36[15]/100,median_C_36_36[15]/100,q_three_C_36_36[15]/100,max_C_36_36[15]/100]
minute_thirty_C = [min_C_36_36[30]/100,q_one_C_36_36[30]/100,median_C_36_36[30]/100,q_three_C_36_36[30]/100,max_C_36_36[30]/100]

minute_zero_R	= [min_R_36_36[0]/100,q_one_R_36_36[0]/100,median_R_36_36[0]/100,q_three_R_36_36[0]/100,max_R_36_36[0]/100]
minute_fifteen_R = [min_R_36_36[15]/100,q_one_R_36_36[15]/100,median_R_36_36[15]/100,q_three_R_36_36[15]/100,max_R_36_36[15]/100]
minute_thirty_R = [min_R_36_36[30]/100,q_one_R_36_36[30]/100,median_R_36_36[30]/100,q_three_R_36_36[30]/100,max_R_36_36[30]/100]

data_L = [minute_zero_L, minute_fifteen_L, minute_thirty_L]
data_C = [minute_zero_C, minute_fifteen_C, minute_thirty_C]
data_R = [minute_zero_R, minute_fifteen_R, minute_thirty_R]

ticks = ['0', '15', '30']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color="black", linewidth=1.5)
    plt.setp(bp['whiskers'], color="black", linewidth=1.5)
    plt.setp(bp['caps'], color="black", linewidth=1.5)
    plt.setp(bp['medians'], color="red", linewidth=1.5)

plt.figure()

bpl_L = plt.boxplot(data_L, positions=np.array(range(len(data_L)))*2, sym='x', widths=1.5, whis=10)
set_box_color(bpl_L, '#D7191C') # colors are from http://colorbrewer2.org/
plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=20)
plt.yticks(fontsize=18)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-0.01, 1.1)
plt.xlabel('time [min]', fontsize=20)
plt.ylabel('median fraction of bees', fontsize=20)
plt.tight_layout()
plt.plot(0,0, marker="x", mew=1.5, mec='black')
plt.plot(2,0.45, marker="x", mew=1.5, mec='black')
plt.plot(4,0.45, marker="x", mew=1.5, mec='black')

plt.show()

bpl_C = plt.boxplot(data_C, positions=np.array(range(len(data_C)))*2, sym='x', widths=1.5, whis=10)
set_box_color(bpl_C, '#D7191C') # colors are from http://colorbrewer2.org/
plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=20)
plt.yticks(fontsize=18)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-0.01, 1.1)
plt.xlabel('time [min]', fontsize=20)
plt.ylabel(' ')#'median fraction of bees[%]')
plt.tight_layout()
plt.plot(0,1, marker="x", mew=1.5, mec='black')
plt.plot(2,0.10, marker="x", mew=1.5, mec='black')
plt.plot(4,0.10, marker="x", mew=1.5, mec='black')

plt.show()

bpl_R = plt.boxplot(data_R, positions=np.array(range(len(data_R)))*2.0, sym='x', widths=1.5, whis=10)
set_box_color(bpl_R, '#D7191C') # colors are from http://colorbrewer2.org/
plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=20)
plt.yticks(fontsize=18)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-0.01, 1.1)
plt.xlabel('time [min]', fontsize=20)
plt.ylabel(' ')#median fraction of bees[%]')
plt.tight_layout()
plt.plot(0,0, marker="x", mew=1.5, mec='black')
plt.plot(2,0.45, marker="x", mew=1.5, mec='black')
plt.plot(4,0.45, marker="x", mew=1.5, mec='black')

plt.show()
