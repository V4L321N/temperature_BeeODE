from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

#######_EXPERIMENT_DATA_#######################################

Switch = [
         [0.00,20.31,35.94,36.72,48.44,50.78,50.78,50.78,53.91,53.91,56.25,57.03,58.59,59.38,60.94,60.94,60.94,60.94,63.28,63.28,62.50,64.06,63.28,63.28,65.63,64.06,64.84,65.63,65.63,65.63,67.19,64.06,62.50,67.19,62.50,65.63,64.84,64.84,66.41,65.63,64.84,63.28,65.63,64.84,64.84,64.84,64.84,65.63,66.41,67.19,65.63,64.84,65.63,65.63,64.06,64.06,62.50,64.06,59.38,59.38,57.81,54.69,53.13,48.44,45.31,46.88,45.31,44.53,41.41,48.44,45.31,47.66,38.28,47.66,46.88,46.88,46.88,44.53,43.75,39.06,31.25,30.47,32.81,31.25,28.13,31.25,24.22,25.78,27.34,23.44,23.44,24.22,22.66,19.53,18.75,18.75,17.97,17.19,16.41,17.19,12.50,15.63,15.63,16.41,12.50,14.06],
         [100.00,64.06,43.75,39.84,25.00,20.31,20.31,15.63,17.97,13.28,11.72,9.38,14.06,14.06,11.72,12.50,10.16,8.59,12.50,7.03,9.38,7.81,6.25,6.25,6.25,6.25,7.81,7.03,6.25,6.25,6.25,6.25,7.81,7.81,5.47,6.25,4.69,6.25,4.69,8.59,6.25,4.69,6.25,7.81,7.81,7.03,7.03,9.38,10.94,10.94,11.72,10.94,9.38,9.38,13.28,15.63,14.06,12.50,12.50,12.50,17.19,14.06,17.19,17.97,17.19,17.19,15.63,17.19,13.28,15.63,21.88,17.19,15.63,15.63,14.06,13.28,15.63,15.63,15.63,14.06,14.06,14.06,10.16,7.81,13.28,16.41,15.63,17.19,18.75,20.31,17.97,14.06,21.88,18.75,17.19,20.31,18.75,17.19,15.63,20.31,16.41,12.50,15.63,16.41,14.06,14.06],
         [0.00,14.06,17.19,18.75,20.31,22.66,23.44,24.22,25.00,24.22,23.44,22.66,21.88,21.88,21.09,21.88,21.09,21.09,21.09,21.09,21.09,21.88,22.66,21.09,21.09,21.88,21.09,21.09,21.88,21.09,21.09,21.09,21.09,22.66,24.22,24.22,24.22,21.88,21.88,25.00,24.22,23.44,23.44,22.66,22.66,22.66,22.66,22.66,23.44,21.88,21.88,22.66,22.66,22.66,23.44,22.66,21.88,21.88,25.00,28.13,28.13,29.69,26.56,32.81,28.13,32.81,32.81,30.47,35.16,35.16,34.38,31.25,32.81,31.25,40.63,39.06,40.63,41.41,39.06,43.75,46.88,53.13,54.69,56.25,56.25,46.88,53.91,52.34,57.81,56.25,59.38,58.59,55.47,57.81,62.50,62.50,64.06,67.19,67.19,68.75,70.31,73.44,73.44,66.41,70.31,70.31]
         ]

"""fresh data needed"""

switch_timeline = np.linspace(0, 105, 106)

#######_CONSTANTS_#################################################

x1 = 0.0056#62#33    # optimum lies @0.0132 as found in the program 09_02_21NEW_dara_AllAtOnce_fit
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

name_number = 0

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

# def Wl(t):
#    return (((a + b * Tl(t)) ** c / ((a + b * Tl(t)) ** c + d ** c)) * e + f) / g
#
# def Wr(t):
#    return (((a + b * Tr(t)) ** c / ((a + b * Tr(t)) ** c + d ** c)) * e + f) / g

A = 12
B = 1.2
C = 27
D = 13

def Wl(t):
    return (A * np.tanh((Tl(t)/B)-C)+D)

def Wr(t):
    return (A * np.tanh((Tr(t)/B)-C)+D)


def euler_ODE_LFR():
    def f(u, t):
        L, F, R = u
        #mu, sigma = 1.2, 0.9
        # z1 = float(np.random.normal(mu, sigma, 1))
        # z2 = float(np.random.normal(mu, sigma, 1))
        # z3 = float(np.random.normal(mu, sigma, 1))
        # z1 = 1
        # z2 = 1
        # z3 = 1
        shape, scale = 4, 0.3
        z1 = float(np.random.gamma(shape, scale, 1))
        z2 = float(np.random.gamma(shape, scale, 1))
        z3 = float(np.random.gamma(shape, scale, 1))

        n = 1
        epsilon_1 = n * z1
        epsilon_2 = n * z2
        epsilon_3 = n * z3

        deL = (epsilon_1 * x1 * (F ** 2) / 2 + epsilon_2 * x1 * F * L - L / (Wl(t)))
        deF = (L / (Wl(t)) + R / (Wr(t)) - epsilon_1 * x1 * (F ** 2) - epsilon_2 * x1 * F * L - epsilon_3 * x1 * F * R)
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

    fig = plt.figure(figsize=(11,8))
    plt.rcParams.update({'font.size': 15})
    l1 = plt.plot(t, (L * 100 / B / 100), color='red', label="L_ODE")
    l2 = plt.plot(t, (F * 100 / B / 100), color='orange', label="F_ODE")
    l3 = plt.plot(t, (R * 100 / B / 100), color='blue', label="R_ODE")
    e1 = plt.plot(switch_timeline, Switch[0], color='red', ls = "--", label="L_EXP")
    e2 = plt.plot(switch_timeline, Switch[1], color='orange', ls = "--", label="F_EXP")
    e3 = plt.plot(switch_timeline, Switch[2], color='blue', ls = "--", label="R_EXP")
    plt.legend(loc='best', fontsize="small")
    plt.xlabel('time [min]', fontsize="12")
    plt.ylabel('bees [%]', fontsize="12")
    np.savetxt('data' + str(name_number) + '.csv', (u * 100 / B / 100), delimiter=',', header="L,C,R", comments='')
    #plt.show()
    return(L, F, R)


for i in range(20):
    #x1 -= 0.0001
    if __name__ == '__main__':
        ODE_L, ODE_F, ODE_R = euler_ODE_LFR()
        name_number +=1

        #plt.show()

# after running the program [36-32] we see that with this configuration we get an minimum at [0.0126, 0.0126, 299.5765]
# after running the program [36-30] we see that with this configuration we get an minimum at [0.0165, 0.0165, 235.4713]
