import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

individual_runs_36_36 = pd.read_csv("C:\\Users\\vstok\\Desktop\\individual_runs_36_36_last.csv")

timeline = np.linspace(0, 30, 31)

# min_L_36_36 = new_Data_36_36.loc[:,'min_L']
# q_one_L_36_36 = new_Data_36_36.loc[:,'q_one_L']
# median_L_36_36 = new_Data_36_36.loc[:,'median_L']
# q_three_L_36_36 = new_Data_36_36.loc[:,'q_three_L']
# max_L_36_36 = new_Data_36_36.loc[:,'max_L']


exp1MB6436_36_rotated_L = individual_runs_36_36.loc[0:, '1MB6436_36_rotated_L']
exp1MB6436_36_rotated_C = individual_runs_36_36.loc[0:, '1MB6436_36_rotated_C']
exp1MB6436_36_rotated_R = individual_runs_36_36.loc[0:, '1MB6436_36_rotated_R']
LISTexp1MB6436_36_rotated_L = []
LISTexp1MB6436_36_rotated_C = []
LISTexp1MB6436_36_rotated_R = []
for item in range(1, len(exp1MB6436_36_rotated_L)):
    LISTexp1MB6436_36_rotated_L.append(float(exp1MB6436_36_rotated_L[item]))
    LISTexp1MB6436_36_rotated_C.append(float(exp1MB6436_36_rotated_C[item]))
    LISTexp1MB6436_36_rotated_R.append(float(exp1MB6436_36_rotated_R[item]))


exp2MB6436_36_MW_L = individual_runs_36_36.loc[0:, '2MB6436_36_MW_L']
exp2MB6436_36_MW_C = individual_runs_36_36.loc[0:, '2MB6436_36_MW_C']
exp2MB6436_36_MW_R = individual_runs_36_36.loc[0:, '2MB6436_36_MW_R']
LISTexp2MB6436_36_MW_L = []
LISTexp2MB6436_36_MW_C = []
LISTexp2MB6436_36_MW_R = []
for item in range(1, len(exp2MB6436_36_MW_L)):
    LISTexp2MB6436_36_MW_L.append(float(exp2MB6436_36_MW_L[item]))
    LISTexp2MB6436_36_MW_C.append(float(exp2MB6436_36_MW_C[item]))
    LISTexp2MB6436_36_MW_R.append(float(exp2MB6436_36_MW_R[item]))

exp3MB6436_36_MW_L = individual_runs_36_36.loc[0:, '3MB6436_36_MW_L']
exp3MB6436_36_MW_C = individual_runs_36_36.loc[0:, '3MB6436_36_MW_C']
exp3MB6436_36_MW_R = individual_runs_36_36.loc[0:, '3MB6436_36_MW_R']
LISTexp3MB6436_36_MW_L = []
LISTexp3MB6436_36_MW_C = []
LISTexp3MB6436_36_MW_R = []
for item in range(1, len(exp3MB6436_36_MW_L)):
    LISTexp3MB6436_36_MW_L.append(float(exp3MB6436_36_MW_L[item]))
    LISTexp3MB6436_36_MW_C.append(float(exp3MB6436_36_MW_C[item]))
    LISTexp3MB6436_36_MW_R.append(float(exp3MB6436_36_MW_R[item]))

exp4MB6436_36_MW_L = individual_runs_36_36.loc[0:, '4MB6436_36_MW_L']
exp4MB6436_36_MW_C = individual_runs_36_36.loc[0:, '4MB6436_36_MW_C']
exp4MB6436_36_MW_R = individual_runs_36_36.loc[0:, '4MB6436_36_MW_R']
LISTexp4MB6436_36_MW_L = []
LISTexp4MB6436_36_MW_C = []
LISTexp4MB6436_36_MW_R = []
for item in range(1, len(exp4MB6436_36_MW_L)):
    LISTexp4MB6436_36_MW_L.append(float(exp4MB6436_36_MW_L[item]))
    LISTexp4MB6436_36_MW_C.append(float(exp4MB6436_36_MW_C[item]))
    LISTexp4MB6436_36_MW_R.append(float(exp4MB6436_36_MW_R[item]))

exp5MB6436_36_L = individual_runs_36_36.loc[0:, '5MB6436_36_L']
exp5MB6436_36_C = individual_runs_36_36.loc[0:, '5MB6436_36_C']
exp5MB6436_36_R = individual_runs_36_36.loc[0:, '5MB6436_36_R']
LISTexp5MB6436_36_L = []
LISTexp5MB6436_36_C = []
LISTexp5MB6436_36_R = []
for item in range(1, len(exp5MB6436_36_L)):
    LISTexp5MB6436_36_L.append(float(exp5MB6436_36_L[item]))
    LISTexp5MB6436_36_C.append(float(exp5MB6436_36_C[item]))
    LISTexp5MB6436_36_R.append(float(exp5MB6436_36_R[item]))

exp6MB6436_36_MW_L = individual_runs_36_36.loc[0:, '6MB6436_36_MW_L']
exp6MB6436_36_MW_C = individual_runs_36_36.loc[0:, '6MB6436_36_MW_C']
exp6MB6436_36_MW_R = individual_runs_36_36.loc[0:, '6MB6436_36_MW_R']
LISTexp6MB6436_36_MW_L = []
LISTexp6MB6436_36_MW_C = []
LISTexp6MB6436_36_MW_R = []
for item in range(1, len(exp6MB6436_36_MW_L)):
    LISTexp6MB6436_36_MW_L.append(float(exp6MB6436_36_MW_L[item]))
    LISTexp6MB6436_36_MW_C.append(float(exp6MB6436_36_MW_C[item]))
    LISTexp6MB6436_36_MW_R.append(float(exp6MB6436_36_MW_R[item]))

plt.figure(figsize=(8,6.5))
plt.plot(timeline, LISTexp1MB6436_36_rotated_L, color = 'black', linewidth=2, linestyle = (0, (1,1)))
plt.plot(timeline, LISTexp2MB6436_36_MW_L, color = 'black', linewidth=2, linestyle = (0, (3,1,1,1)))
plt.plot(timeline, LISTexp3MB6436_36_MW_L,  color = 'black', linewidth=2, linestyle = (0, (5,1)))
plt.plot(timeline, LISTexp4MB6436_36_MW_L, color = 'black', linewidth=2, linestyle = 'dashdot')
plt.plot(timeline, LISTexp5MB6436_36_L, color = 'black', linewidth=2, linestyle = 'solid')
plt.plot(timeline, LISTexp6MB6436_36_MW_L, color = 'black', linewidth=2, linestyle = (0, (3,1,1,1,1,1)))
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.xlabel('time [min]', fontsize=20)
plt.ylabel('median fraction of bees', fontsize=20)
plt.ylim(-0.01, 1.1)
plt.show()

plt.figure(figsize=(8,6.5))
plt.plot(timeline, LISTexp1MB6436_36_rotated_C, color = 'black', linewidth=2, linestyle = (0, (1,1)))
plt.plot(timeline, LISTexp2MB6436_36_MW_C, color = 'black', linewidth=2, linestyle = (0, (3,1,1,1)))
plt.plot(timeline, LISTexp3MB6436_36_MW_C,  color = 'black', linewidth=2, linestyle = (0, (5,1)))
plt.plot(timeline, LISTexp4MB6436_36_MW_C, color = 'black', linewidth=2, linestyle = 'dashdot')
plt.plot(timeline, LISTexp5MB6436_36_C, color = 'black', linewidth=2, linestyle = 'solid')
plt.plot(timeline, LISTexp6MB6436_36_MW_C, color = 'black', linewidth=2, linestyle = (0, (3,1,1,1,1,1)))
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.xlabel('time [min]', fontsize=20)
plt.ylabel('median fraction of bees', fontsize=20)
plt.ylim(-0.01, 1.1)
plt.show()

plt.figure(figsize=(8,6.5))
plt.plot(timeline, LISTexp1MB6436_36_rotated_R, color = 'black', linewidth=2, linestyle = (0, (1,1)))
plt.plot(timeline, LISTexp2MB6436_36_MW_R, color = 'black', linewidth=2, linestyle = (0, (3,1,1,1)))
plt.plot(timeline, LISTexp3MB6436_36_MW_R,  color = 'black', linewidth=2, linestyle = (0, (5,1)))
plt.plot(timeline, LISTexp4MB6436_36_MW_R, color = 'black', linewidth=2, linestyle = 'dashdot')
plt.plot(timeline, LISTexp5MB6436_36_R, color = 'black', linewidth=2, linestyle = 'solid')
plt.plot(timeline, LISTexp6MB6436_36_MW_R, color = 'black', linewidth=2, linestyle = (0, (3,1,1,1,1,1)))
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.xlabel('time [min]', fontsize=20)
plt.ylabel('median fraction of bees', fontsize=20)
plt.ylim(-0.01, 1.1)
plt.show()
# plt.plot(timeline, LISTexp1MB6436_36_rotated_L, color = 'blue', label='run #1', linewidth=2, linestyle = (0, (1,1)))
# plt.plot(timeline, LISTexp2MB6436_36_MW_L, color = 'darkmagenta', label='run #2', linewidth=2, linestyle = (0, (3,1,1,1)))
# plt.plot(timeline, LISTexp3MB6436_36_MW_L,  color = 'red', label='run #3', linewidth=2, linestyle = (0, (5,1)))
# plt.plot(timeline, LISTexp4MB6436_36_MW_L, color = 'c', label='run #4', linewidth=2, linestyle = 'dashdot')
# plt.plot(timeline, LISTexp5MB6436_36_L, color = 'black', label='run #5', linewidth=2, linestyle = 'solid')
# plt.plot(timeline, LISTexp6MB6436_36_MW_L, color = 'orange', label='run #6', linewidth=2, linestyle = (0, (3,1,1,1,1,1)))
# plt.xlabel('time [min]', fontsize=12)
# plt.ylabel('median fraction of bees[%]', fontsize=12)
# plt.legend(loc='best')
# plt.ylim(-1, 101)#0)
# plt.show()
#
# plt.plot(timeline, LISTexp1MB6436_36_rotated_C, linestyle = "--", color = 'orange', label='x')
# plt.plot(timeline, LISTexp2MB6436_36_MW_C, linestyle = "--", color = 'orange', label='x')
# plt.plot(timeline, LISTexp3MB6436_36_MW_C, linestyle = "--", color = 'orange', label='x')
# plt.plot(timeline, LISTexp4MB6436_36_MW_C, linestyle = "--", color = 'orange', label='x')
# plt.plot(timeline, LISTexp5MB6436_36_C, linestyle = "--", color = 'orange', label='x')
# plt.plot(timeline, LISTexp6MB6436_36_MW_C, linestyle = "--", color = 'orange', label='x')
# plt.xlabel('time [min]', fontsize=12)
# plt.ylabel('median fraction of bees[%]', fontsize=12)
# plt.legend(loc='best')
# plt.ylim(-1, 101)#0)
# plt.show()
#
# plt.plot(timeline, LISTexp1MB6436_36_rotated_R, linestyle = "--", color = 'red', label='x')
# plt.plot(timeline, LISTexp2MB6436_36_MW_R, linestyle = "--", color = 'red', label='x')
# plt.plot(timeline, LISTexp3MB6436_36_MW_R, linestyle = "--", color = 'red', label='x')
# plt.plot(timeline, LISTexp4MB6436_36_MW_R, linestyle = "--", color = 'red', label='x')
# plt.plot(timeline, LISTexp5MB6436_36_R, linestyle = "--", color = 'red', label='x')
# plt.plot(timeline, LISTexp6MB6436_36_MW_R, linestyle = "--", color = 'red', label='x')
# plt.xlabel('time [min]', fontsize=12)
# plt.ylabel('median fraction of bees[%]', fontsize=12)
#plt.legend(loc='best')
#plt.ylim(-1, 101)#0)
#plt.show()
