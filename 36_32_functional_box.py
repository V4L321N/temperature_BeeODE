import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linspace, zeros, asarray

timeline = np.linspace(0, 30, 31)
new_Data_36_32 = pd.read_csv("C:\\Users\\vstok\\Desktop\\36_32_last.csv")

min_L_36_32 = new_Data_36_32.loc[:,'min_L']
q_one_L_36_32 = new_Data_36_32.loc[:,'q_one_L']
median_L_36_32 = new_Data_36_32.loc[:,'median_L']
q_three_L_36_32 = new_Data_36_32.loc[:,'q_three_L']
max_L_36_32 = new_Data_36_32.loc[:,'max_L']

min_C_36_32 = new_Data_36_32.loc[:,'min_C']
q_one_C_36_32 = new_Data_36_32.loc[:,'q_one_C']
median_C_36_32 = new_Data_36_32.loc[:,'median_C']
q_three_C_36_32 = new_Data_36_32.loc[:,'q_three_C']
max_C_36_32 = new_Data_36_32.loc[:,'max_C']

min_R_36_32 = new_Data_36_32.loc[:,'min_R']
q_one_R_36_32 = new_Data_36_32.loc[:,'q_one_R']
median_R_36_32 = new_Data_36_32.loc[:,'median_R']
q_three_R_36_32 = new_Data_36_32.loc[:,'q_three_R']
max_R_36_32 = new_Data_36_32.loc[:,'max_R']

x1 = 0.0056#62 #calculated minimum
x2 = 1.00
Cr = 0
Cl = 0

a = -2.28
b = 0.08
c = 5.15
d = 0.32
e = 28.5 #40 #50
f = 7
g = 1.4 #original was 15

dx = 1
M = 30        # depending in experiment eiter 30 or 105
N_t = int(M / dx)

def ode_FE(f, U_0, dt, T):
    N_t = int(round(float(T)/dt))
    # Ensure that any list/tuple returned from f_ is wrapped as array
    f_ = lambda u, t: asarray(f(u, t))
    u = zeros((N_t+1, len(U_0)))
    t = linspace(0, N_t*dt, len(u))
    u[0] = U_0
    for n in range(N_t):
        u[n+1] = u[n] + dt*f_(u[n], t[n])
    return u, t

A = 12
B = 1.2
C = 27
D = 13

def Wl_1():
    return (A * np.tanh((36/B)-C)+D)

def Wr_1():
    return (A * np.tanh((32/B)-C)+D)

def euler_ODE_LFR_1():
    def f(u, t):
        L, F, R = u

        # deL = (x1 * (F ** 2) / 2 + x1 * F * (L + x2 * Cl) - L / (Wl_1(Tl(t))))
        # deF = (L / (Wl_1(Tl(t))) + R / (Wr_1(Tr(t))) - x1 * (F ** 2) - x1 * F * (L + x2 * Cl) - x1 * F * (R + x2 * Cr))
        # deR = (x1 * (F ** 2) / 2 + x1 * F * (R + x2 * Cr) - R / (Wr_1(Tr(t))))
        deL = (x1 * (F ** 2) / 2 + x1 * F * (L + x2 * Cl) - L / (Wl_1()))
        deF = (L / (Wl_1()) + R / (Wr_1()) - x1 * (F ** 2) - x1 * F * (L + x2 * Cl) - x1 * F * (R + x2 * Cr))# with the only F(ree) bees you could use (x1 PLUS x1)!!!!!!
        deR = (x1 * (F ** 2) / 2 + x1 * F * (R + x2 * Cr) - R / (Wr_1()))
        return [deL, deF, deR]

    dt = dx               # Step size
    D = M               # Duration in minutes
    N_t = int(D/dt)      # Corresponding number of steps
    T = dt*N_t           # End time
    B = 64                #number of bees
    U_0 = [0, B, 0]

    u, t = ode_FE(f, U_0, dt, T)

    L = u[:,0]
    F = u[:,1]
    R = u[:,2]

    print(u*100/B)

    fig = plt.figure(figsize=(8,6.1))
    plt.xlim(0, 30)
    plt.ylim(-0.01, 1.1)
    plt.plot(t, (L*100/B/100), linestyle='dashed', color='black', label='model fit', linewidth=2)
    plt.fill_between(timeline, q_one_L_36_32, q_three_L_36_32, color='dodgerblue', alpha=0.5, label='IQR')
    plt.fill_between(timeline, min_L_36_32, max_L_36_32, color='skyblue', alpha=0.5, label='min, max')
    plt.plot(timeline, median_L_36_32, color='blue', label='median')
    plt.legend(loc='lower right', fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.xlabel(' ', fontsize=18)
    plt.ylabel('median fraction of bees', fontsize=20)
    plt.show()

    fig = plt.figure(figsize=(8,6.1))
    plt.xlim(0, 30)
    plt.ylim(-0.01, 1.1)
    plt.plot(t, (F*100/B/100), linestyle='dashed', color='black', label='model fit', linewidth=2)
    plt.fill_between(timeline, q_one_C_36_32, q_three_C_36_32, color='tomato', alpha=0.5, label='IQR')
    plt.fill_between(timeline, min_C_36_32, max_C_36_32, color='lightsalmon', alpha=0.5, label='min, max')
    plt.plot(timeline, median_C_36_32, color='r', label='median')
    plt.legend(loc='best', fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.xlabel(' ', fontsize=18)
    plt.ylabel(' ', fontsize=20)
    plt.show()

    fig = plt.figure(figsize=(8,6.1))
    plt.xlim(0, 30)
    plt.ylim(-0.01, 1.1)
    plt.plot(t, (R*100/B/100), linestyle='dashed', color='black', label='model fit', linewidth=2)
    plt.fill_between(timeline, q_one_R_36_32, q_three_R_36_32, color='sandybrown', alpha=0.5, label='IQR')
    plt.fill_between(timeline, min_R_36_32, max_R_36_32, color='peachpuff', alpha=0.5, label='min, max')

    plt.plot(timeline, median_R_36_32, color='peru', label='median')
    plt.legend(loc='best', fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.xlabel(' ', fontsize=18)
    plt.ylabel(' ', fontsize=20)
    plt.show()

    return(L, F, R)

if __name__ == '__main__':
    ODE_L_1, ODE_F_1, ODE_R_1 = euler_ODE_LFR_1()

# fig, ax1 = plt.subplots(1, 1, sharex=True)
#
# ax1.fill_between(timeline, min_L_36_32, max_L_36_32, color='skyblue', alpha=0.5)
# ax1.fill_between(timeline, q_one_L_36_32, q_three_L_36_32, color='dodgerblue', alpha=0.5)
# ax1.plot(timeline, median_L_36_32, color='blue')
# plt.xlim(-1, 31)
# plt.ylim(-1, 101)
# plt.xlabel('time[min]', fontsize=12)
# plt.ylabel('median fraction of bees[%]', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()
#
# fig, ax1 = plt.subplots(1, 1, sharex=True)
#
# ax1.fill_between(timeline, min_C_36_32, max_C_36_32, color='lightsalmon', alpha=0.5)
# ax1.fill_between(timeline, q_one_C_36_32, q_three_C_36_32, color='tomato', alpha=0.5)
# ax1.plot(timeline, median_C_36_32, color='r')
# plt.xlim(-1, 31)
# plt.ylim(-1, 101)
# plt.xlabel('time[min]', fontsize=12)
# plt.ylabel('median fraction of bees[%]', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()
#
# fig, ax1 = plt.subplots(1, 1, sharex=True)
#
# ax1.fill_between(timeline, min_R_36_32, max_R_36_32, color='peachpuff', alpha=0.5)
# ax1.fill_between(timeline, q_one_R_36_32, q_three_R_36_32, color='sandybrown', alpha=0.5)
# ax1.plot(timeline, median_R_36_32, color='peru')
# plt.xlim(-1, 31)
# plt.ylim(-1, 101)
# plt.xlabel('time[min]', fontsize=12)
# plt.ylabel('median fraction of bees[%]', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()
