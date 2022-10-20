import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

switch_timeline = np.linspace(0, 105, 105)

Temp_l = 35.8
Temp_r = 32.2

RealTemp_ALL = pd.read_csv("C:\\Users\\vstok\\Desktop\\WIPx_Arena_Gradienten.csv")
T_L = RealTemp_ALL.loc[:,'T Left']
STD = RealTemp_ALL.loc[:,'std_dev_L']

RealTemp_L = []
RealTemp_C =[]
RealTemp_R = []

def PULSE(start, end, height, t):
    return np.heaviside(t - start, 1) * height - np.heaviside(t - end, 0) * height # <-- for old sin decrease

def exp_DECAY(start, end, height, t):
    return Temp_l + (6 * np.exp(-(t - start)/45) - 6) * PULSE(start, end, height, t)

#print(exp_DECAY(30.00, 105.00, 1, switch_timeline))
plt.plot(exp_DECAY(30.00, 105.00, 1, switch_timeline), linestyle = "--", color = 'red', label='exponential decay')
plt.fill_between(switch_timeline, T_L-STD, T_L+STD, color='skyblue', alpha=0.5, label='standard deviation')
plt.plot(T_L, color = 'blue', label='mean sensor data')
plt.axvline(x=30, linestyle='dotted', color='black', linewidth=1.5)
plt.xlabel('time [min]', fontsize=12)
plt.ylabel('temperature [°C]', fontsize=12)
plt.xlim(0,105)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='best')
plt.show()
