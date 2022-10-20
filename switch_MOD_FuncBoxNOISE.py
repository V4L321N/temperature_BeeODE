from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import statistics

switch_timeline = np.linspace(0, 105, 106)
B=64

#######_EXPERIMENT_DATA_#######################################
NOISE_SWITCH_DATA_0 = pd.read_csv("data0.csv")
NOISE_SWITCH_DATA_1 = pd.read_csv("data1.csv")
NOISE_SWITCH_DATA_2 = pd.read_csv("data2.csv")
NOISE_SWITCH_DATA_3 = pd.read_csv("data3.csv")
NOISE_SWITCH_DATA_4 = pd.read_csv("data4.csv")
NOISE_SWITCH_DATA_5 = pd.read_csv("data5.csv")
NOISE_SWITCH_DATA_6 = pd.read_csv("data6.csv")
NOISE_SWITCH_DATA_7 = pd.read_csv("data7.csv")
NOISE_SWITCH_DATA_8 = pd.read_csv("data8.csv")
NOISE_SWITCH_DATA_9 = pd.read_csv("data9.csv")
NOISE_SWITCH_DATA_10 = pd.read_csv("data10.csv")
NOISE_SWITCH_DATA_11 = pd.read_csv("data11.csv")
NOISE_SWITCH_DATA_12 = pd.read_csv("data12.csv")
NOISE_SWITCH_DATA_13 = pd.read_csv("data13.csv")
NOISE_SWITCH_DATA_14 = pd.read_csv("data14.csv")
NOISE_SWITCH_DATA_15 = pd.read_csv("data15.csv")
NOISE_SWITCH_DATA_16 = pd.read_csv("data16.csv")
NOISE_SWITCH_DATA_17 = pd.read_csv("data17.csv")
NOISE_SWITCH_DATA_18 = pd.read_csv("data18.csv")
NOISE_SWITCH_DATA_19 = pd.read_csv("data19.csv")
#NOISE_SWITCH_DATA_20 = pd.read_csv("data20.csv")

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
# for i in range(1,len(NOISE_SWITCH_DATA_0.loc[:,'L'])):
#     SAMPLE = []
#     SAMPLE.append((NOISE_SWITCH_DATA_0.loc[i,'L'])*100/64)
#     SAMPLE.append((NOISE_SWITCH_DATA_1.loc[i,'L'])*100/64)
#     SAMPLE.append((NOISE_SWITCH_DATA_2.loc[i,'L'])*100/64)
#     SAMPLE.append((NOISE_SWITCH_DATA_3.loc[i,'L'])*100/64)
#     SAMPLE.append((NOISE_SWITCH_DATA_4.loc[i,'L'])*100/64)
#     SAMPLE.append((NOISE_SWITCH_DATA_5.loc[i,'L'])*100/64)
#     SAMPLE.append((NOISE_SWITCH_DATA_6.loc[i,'L'])*100/64)
#     q1, median, q3= np.percentile(SAMPLE, [25,50,75])
#     min, max = np.min(SAMPLE), np.max(SAMPLE)
#     MIN.append(min)
#     Q1.append(q1)
#     MEDIAN.append(median)
#     Q3.append(q3)
#     MAX.append(max)
#
# fig, ax1 = plt.subplots(1, 1, sharex=True)
#
# ax1.fill_between(switch_timeline, MIN, MAX, color='skyblue', alpha=0.5)
# ax1.fill_between(switch_timeline, Q1, Q3, color='dodgerblue', alpha=0.5)
# ax1.plot(switch_timeline, MEDIAN, color='blue')
# plt.xlim(-1, 106)
# plt.ylim(-1, 101)
# plt.xlabel('time[min]', fontsize=12)
# plt.ylabel('median fraction of bees[%]', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()
#


#######_CONSTANTS_#################################################

x1 = 0.0066 #33    # optimum lies @0.0132 as found in the program 09_02_21NEW_dara_AllAtOnce_fit
x2 = 0#1#0.5        # stopping probablity at a cage
Cr = 0            # number of caged bees right...
Cl = 0            # ...and left

a = -2.28
b = 0.08
c = 5.15
d = 0.32
e = 28.5 #40 #50
f = 7
g = 1.5 #original was 15

dx = 1    # step size for forward gauß algorithm
M = 105        # depending in experiment eiter 30 or 105 time steps (minutes)
N_t = int(M / dx) # number of steps for the algorithm

Temp_l = 35.8
Temp_r = 32.2

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

def PULSE(start, end, height, t):
    return np.heaviside(t - start, 1) * height - np.heaviside(t - end, 0) * height # <-- for old sin decrease

def exp_DECAY(start, end, height, t):
    return Temp_l + (6 * np.exp(-(t - start)/45) - 6) * PULSE(start, end, height, t)

def Tl(t):
    return exp_DECAY(30.00, 105.00, 1, t)

def Tr(t):
    return Temp_r + PULSE(30, 50, 0, t)

def Wl(t):
   return (((a + b * Tl(t)) ** c / ((a + b * Tl(t)) ** c + d ** c)) * e + f) / g

def Wr(t):
   return (((a + b * Tr(t)) ** c / ((a + b * Tr(t)) ** c + d ** c)) * e + f) / g



def euler_ODE_LFR():
    def f(u, t):
        L, F, R = u
        mu, sigma = 1, 0.333
        z1 = 1#float(np.random.normal(mu, sigma, 1))
        z2 = 1#float(np.random.normal(mu, sigma, 1))
        z3 = 1#float(np.random.normal(mu, sigma, 1))

        # z1 = 1
        # z2 = 1
        # z3 = 1

        n = 1
        epsilon_1 = n * z1
        epsilon_2 = n * z2
        epsilon_3 = n * z3

        deL = (epsilon_1 * x1 * (F ** 2) / 2 + epsilon_2 * x1 * F * L - L / (Wl(t)))
        deF = (L / (Wl(t)) + R / (Wr(t)) - epsilon_1 * x1 * (F ** 2) - epsilon_2 * x1 * F * L - epsilon_3 * x1 * F * (R + x2 * Cr))
        deR = (epsilon_1 * x1 * (F ** 2) / 2 + epsilon_3 * x1 * F * R - R / (Wr(t)))
        #print(round(deL+deF+deR), min(z1,z2,z3))
        return [deL, deF, deR]

    dt = dx               # Step size
    D = M               # Duration in minutes
    N_t = int(D/dt)      # Corresponding number of steps
    T = dt*N_t           # End time

    B = 64 #bees * 100 / bees
    U_0 = [0, B, 0]

    u, t = ode_FE(f, U_0, dt, T)

    L = u[:,0]
    F = u[:,1]
    R = u[:,2]
    #
    plt.figure(figsize=(8,6.1))
    plt.rcParams.update({'font.size': 15})
    for i in range(0,len(NOISE_SWITCH_DATA_0.loc[:,'L'])):
        SAMPLE = []
        SAMPLE.append((NOISE_SWITCH_DATA_0.loc[i,'L']))
        SAMPLE.append((NOISE_SWITCH_DATA_1.loc[i,'L']))
        SAMPLE.append((NOISE_SWITCH_DATA_2.loc[i,'L']))
        SAMPLE.append((NOISE_SWITCH_DATA_3.loc[i,'L']))
        SAMPLE.append((NOISE_SWITCH_DATA_4.loc[i,'L']))
        SAMPLE.append((NOISE_SWITCH_DATA_5.loc[i,'L']))
        SAMPLE.append((NOISE_SWITCH_DATA_6.loc[i,'L']))
        q1, median, q3= np.percentile(SAMPLE, [25,50,75])
        min, max = np.min(SAMPLE), np.max(SAMPLE)
        MIN_L.append(min)
        Q1_L.append(q1)
        MEDIAN_L.append(median)
        Q3_L.append(q3)
        MAX_L.append(max)
    plt.axvline(x=30, linestyle='dotted', color='black', linewidth=1.5)
    plt.fill_between(switch_timeline, Q1_L, Q3_L, color='dodgerblue', alpha=0.5, label='IQR')
    plt.fill_between(switch_timeline, MIN_L, MAX_L, color='skyblue', alpha=0.5, label='min, max')
    plt.plot(switch_timeline, MEDIAN_L, color='blue', label='median')
    #l1 = plt.plot(t, L, color='black', label="L_ODE", linestyle="dashed")
    plt.legend(loc='best', fontsize="small")
    plt.xlim(-1, 106)
    plt.ylim(-0.01, 1.1)
    plt.xlabel('time [min]', fontsize=18)
    plt.ylabel('median fraction of bees', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='best', fontsize=15)
    plt.show()

    plt.figure(figsize=(8,6.1))
    plt.rcParams.update({'font.size': 15})
    for i in range(0,len(NOISE_SWITCH_DATA_0.loc[:,'L'])):
        SAMPLE = []
        SAMPLE.append((NOISE_SWITCH_DATA_0.loc[i,'C']))
        SAMPLE.append((NOISE_SWITCH_DATA_1.loc[i,'C']))
        SAMPLE.append((NOISE_SWITCH_DATA_2.loc[i,'C']))
        SAMPLE.append((NOISE_SWITCH_DATA_3.loc[i,'C']))
        SAMPLE.append((NOISE_SWITCH_DATA_4.loc[i,'C']))
        SAMPLE.append((NOISE_SWITCH_DATA_5.loc[i,'C']))
        SAMPLE.append((NOISE_SWITCH_DATA_6.loc[i,'C']))
        q1, median, q3= np.percentile(SAMPLE, [25,50,75])
        min, max = np.min(SAMPLE), np.max(SAMPLE)
        MIN_C.append(min)
        Q1_C.append(q1)
        MEDIAN_C.append(median)
        Q3_C.append(q3)
        MAX_C.append(max)
    plt.fill_between(switch_timeline, Q1_C, Q3_C, color='tomato', alpha=0.5, label='IQR')
    plt.fill_between(switch_timeline, MIN_C, MAX_C, color='lightsalmon', alpha=0.5, label='min, max')
    plt.plot(switch_timeline, MEDIAN_C, color='r', label='median')
    #l2 = plt.plot(t, F, color='black', label="F_ODE", linestyle="dashed")
    plt.legend(loc='best', fontsize="small")
    plt.xlim(-1, 106)
    plt.ylim(-0.01, 1.1)
    plt.xlabel('time [min]', fontsize=18)
    plt.ylabel(' ', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='best', fontsize=15)
    plt.show()

    plt.figure(figsize=(8,6.1))
    plt.rcParams.update({'font.size': 15})
    for i in range(0,len(NOISE_SWITCH_DATA_0.loc[:,'L'])):
        SAMPLE = []
        SAMPLE.append((NOISE_SWITCH_DATA_0.loc[i,'R']))
        SAMPLE.append((NOISE_SWITCH_DATA_1.loc[i,'R']))
        SAMPLE.append((NOISE_SWITCH_DATA_2.loc[i,'R']))
        SAMPLE.append((NOISE_SWITCH_DATA_3.loc[i,'R']))
        SAMPLE.append((NOISE_SWITCH_DATA_4.loc[i,'R']))
        SAMPLE.append((NOISE_SWITCH_DATA_5.loc[i,'R']))
        SAMPLE.append((NOISE_SWITCH_DATA_6.loc[i,'R']))
        q1, median, q3= np.percentile(SAMPLE, [25,50,75])
        min, max = np.min(SAMPLE), np.max(SAMPLE)
        MIN_R.append(min)
        Q1_R.append(q1)
        MEDIAN_R.append(median)
        Q3_R.append(q3)
        MAX_R.append(max)

    plt.fill_between(switch_timeline, Q1_R, Q3_R, color='sandybrown', alpha=0.5, label='IQR')
    plt.fill_between(switch_timeline, MIN_R, MAX_R, color='peachpuff', alpha=0.5, label='min, max')
    plt.plot(switch_timeline, MEDIAN_R, color='peru', label='median')
    #l3 = plt.plot(t, R, color='black', label="R_ODE", linestyle="dashed")
    plt.legend(loc='best', fontsize="small")
    plt.xlim(-1, 106)
    plt.ylim(-0.01, 1.1)
    plt.xlabel('time [min]', fontsize=18)
    plt.ylabel(' ', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.legend(loc='best', fontsize=15)
    plt.show()

    return(L, F, R)


for i in range(1):

    #x1 -= 0.0001
    if __name__ == '__main__':
        ODE_L, ODE_F, ODE_R = euler_ODE_LFR()
        plt.show()




        # plt.plot(switch_timeline, Switch[0])
        # plt.plot(switch_timeline, Switch[1])
        # plt.plot(switch_timeline, Switch[2])
        # plt.errorbar(x, mean_L, deviation_L, linestyle='None', marker='.', color='red', label='L_ODE')
        # plt.errorbar(x, mean_F, deviation_F, linestyle='None', marker='.', color='orange', label='F_ODE')
        # plt.errorbar(x, mean_R, deviation_R, linestyle='None', marker='.', color='blue', label='R_ODE')
        #plt.legend(loc='best', fontsize="small")

        #plt.show()

# after running the program [36-32] we see that with this configuration we get an minimum at [0.0126, 0.0126, 299.5765]
# after running the program [36-30] we see that with this configuration we get an minimum at [0.0165, 0.0165, 235.4713]
