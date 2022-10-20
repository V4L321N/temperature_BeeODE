from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#######_EXPERIMENT_DATA_#######################################


Data_SWITCH = pd.read_csv("C:\\Users\\vstok\\Desktop\\SWITCH_exp_last.csv")

min_L_SWITCH = Data_SWITCH.loc[:,'min_L']
q_one_L_SWITCH = Data_SWITCH.loc[:,'q_one_L']
median_L_SWITCH = Data_SWITCH.loc[:,'median_L']
q_three_L_SWITCH = Data_SWITCH.loc[:,'q_three_L']
max_L_SWITCH = Data_SWITCH.loc[:,'max_L']

min_C_SWITCH = Data_SWITCH.loc[:,'min_C']
q_one_C_SWITCH = Data_SWITCH.loc[:,'q_one_C']
median_C_SWITCH = Data_SWITCH.loc[:,'median_C']
q_three_C_SWITCH = Data_SWITCH.loc[:,'q_three_C']
max_C_SWITCH = Data_SWITCH.loc[:,'max_C']

min_R_SWITCH = Data_SWITCH.loc[:,'min_R']
q_one_R_SWITCH = Data_SWITCH.loc[:,'q_one_R']
median_R_SWITCH = Data_SWITCH.loc[:,'median_R']
q_three_R_SWITCH = Data_SWITCH.loc[:,'q_three_R']
max_R_SWITCH = Data_SWITCH.loc[:,'max_R']

"""fresh data needed"""

switch_timeline = np.linspace(0, 105, 105)

#sfig, ax1 = plt.subplots(1, 1, sharex=True)
plt.figure(figsize=(8,6.1))
plt.axvline(x=30, linestyle='dotted', color='black', linewidth=1.5)
plt.fill_between(switch_timeline, q_one_L_SWITCH, q_three_L_SWITCH, color='dodgerblue', alpha=0.5, label='IQR')
plt.fill_between(switch_timeline, min_L_SWITCH, max_L_SWITCH, color='skyblue', alpha=0.5, label='min, max')
plt.plot(switch_timeline, median_L_SWITCH, color='blue', label='median')
plt.xlim(-1, 106)
plt.ylim(-0.01, 1.1)
plt.xlabel(' ', fontsize=18)
plt.ylabel('median fraction of bees', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.legend(loc='best', fontsize=15)
plt.show()

#fig, ax1 = plt.subplots(1, 1, sharex=True)
plt.figure(figsize=(8,6.1))
plt.fill_between(switch_timeline, q_one_C_SWITCH, q_three_C_SWITCH, color='tomato', alpha=0.5, label='IQR')
plt.fill_between(switch_timeline, min_C_SWITCH, max_C_SWITCH, color='lightsalmon', alpha=0.5, label='min, max')
plt.plot(switch_timeline, median_C_SWITCH, color='r', label='median')
plt.xlim(-1, 106)
plt.ylim(-0.01, 1.1)
plt.xlabel(' ', fontsize=18)
plt.ylabel(' ', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.legend(loc='best', fontsize=15)
plt.show()

#fig, ax1 = plt.subplots(1, 1, sharex=True)
plt.figure(figsize=(8,6.1))
plt.fill_between(switch_timeline, q_one_R_SWITCH, q_three_R_SWITCH, color='sandybrown', alpha=0.5, label='IQR')
plt.fill_between(switch_timeline, min_R_SWITCH, max_R_SWITCH, color='peachpuff', alpha=0.5, label='min, max')
plt.plot(switch_timeline, median_R_SWITCH, color='peru', label='median')
plt.xlim(-1, 106)
plt.ylim(-0.01, 1.1)
plt.xlabel(' ', fontsize=18)
plt.ylabel(' ', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=18)
plt.legend(loc='best', fontsize=15)
plt.show()

#######_CONSTANTS_#################################################

x1 = 0.0056#33    # optimum lies @0.0132 as found in the program 09_02_21NEW_dara_AllAtOnce_fit
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

plt.plot(exp_DECAY(30.00, 105.00, 1, switch_timeline))
plt.show()
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
        mu, sigma = 1, 0.5
        z1 = 1#float(np.random.normal(mu, sigma, 1))
        z2 = 1#float(np.random.normal(mu, sigma, 1))
        z3 = 1#float(np.random.normal(mu, sigma, 1))

        n = 1
        epsilon_1 = n * z1
        epsilon_2 = n * z2
        epsilon_3 = n * z3

        deL = (epsilon_1 * x1 * (F ** 2) / 2 + epsilon_2 * x1 * F * L - L / (Wl(t)))
        deF = (L / (Wl(t)) + R / (Wr(t)) - epsilon_1 * x1 * (F ** 2) - epsilon_2 * x1 * F * L - epsilon_3 * x1 * F * (R + x2 * Cr))
        deR = (epsilon_1 * x1 * (F ** 2) / 2 + epsilon_3 * x1 * F * R - R / (Wr(t)))
        print(round(deL+deF+deR), min(z1,z2,z3))
        return [deL, deF, deR]

    dt = dx               # Step size
    D = M               # Duration in minutes
    N_t = int(D/dt)      # Corresponding number of steps
    T = dt*N_t           # End time

    B = 64 #bees * 100 / bees
    U_0 = [0, B, 0]

    u, t = ode_FE(f, U_0, dt, T)

    L = u[:,0] * 100 / B
    F = u[:,1] * 100 / B
    R = u[:,2] * 100 / B

    fig = plt.figure(figsize=(11,8))

    switch_timeline = np.linspace(0, 105, 105)

    fig, ax1 = plt.subplots(1, 1, sharex=True)

    ax1.fill_between(switch_timeline, min_L_SWITCH, max_L_SWITCH, color='skyblue', alpha=0.5)
    ax1.fill_between(switch_timeline, q_one_L_SWITCH, q_three_L_SWITCH, color='dodgerblue', alpha=0.5)
    ax1.plot(switch_timeline, median_L_SWITCH, color='blue')
    ax1.plot(t, L, color='black', label="L_ODE", linestyle='dashed')
    plt.xlim(-1, 106)
    plt.ylim(-1, 101)
    plt.xlabel('time[min]', fontsize=12)
    plt.ylabel('median fraction of bees[%]', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    fig, ax1 = plt.subplots(1, 1, sharex=True)

    ax1.fill_between(switch_timeline, min_C_SWITCH, max_C_SWITCH, color='lightsalmon', alpha=0.5)
    ax1.fill_between(switch_timeline, q_one_C_SWITCH, q_three_C_SWITCH, color='tomato', alpha=0.5)
    ax1.plot(switch_timeline, median_C_SWITCH, color='r')
    ax1.plot(t, F, color='black', label="C_ODE", linestyle='dashdot')
    plt.xlim(-1, 106)
    plt.ylim(-1, 101)
    plt.xlabel('time[min]', fontsize=12)
    plt.ylabel('median fraction of bees[%]', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    fig, ax1 = plt.subplots(1, 1, sharex=True)

    ax1.fill_between(switch_timeline, min_R_SWITCH, max_R_SWITCH, color='peachpuff', alpha=0.5)
    ax1.fill_between(switch_timeline, q_one_R_SWITCH, q_three_R_SWITCH, color='sandybrown', alpha=0.5)
    ax1.plot(switch_timeline, median_R_SWITCH, color='peru')
    ax1.plot(t, R, color='black', label="C_ODE", linestyle='dotted')
    plt.xlim(-1, 106)
    plt.ylim(-1, 101)
    plt.xlabel('time[min]', fontsize=12)
    plt.ylabel('median fraction of bees[%]', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


    plt.rcParams.update({'font.size': 15})
    l1 = plt.plot(t, L, color='red', label="L_ODE")
    l2 = plt.plot(t, F, color='orange', label="F_ODE")
    l3 = plt.plot(t, R, color='blue', label="R_ODE")
    plt.legend(loc='best', fontsize="small")
    plt.xlabel('time [min]', fontsize="12")
    plt.ylabel('bees [%]', fontsize="12")
    plt.show()
    return(L, F, R)




for i in range(1):
    #x1 -= 0.0001
    if __name__ == '__main__':
        ODE_L, ODE_F, ODE_R = euler_ODE_LFR()

# after running the program [36-32] we see that with this configuration we get an minimum at [0.0126, 0.0126, 299.5765]
# after running the program [36-30] we see that with this configuration we get an minimum at [0.0165, 0.0165, 235.4713]
