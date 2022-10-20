from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#######_EXPERIMENT_DATA_#######################################

Exp_36_30 = [
            [0,17.19,32.42,44.14,55.86,68.75,76.56,82.03,82.81,81.25,81.25,84.38,80.08,82.03,88.67,87.11,88.28,88.28,88.67,90.63,90.63,89.84,91.41,92.19,93.75,93.75,94.14,94.14,95.31,94.92,93.75],
            [100,69.53,54.30,48.05,40.23,28.91,21.48,16.41,16.80,16.80,15.63,12.50,17.19,14.84,9.77,10.55,10.16,10.55,7.81,8.59,8.59,10.16,7.42,7.42,5.08,5.86,5.08,5.47,4.69,4.30,4.30],
            [0,8.98,11.72,7.81,4.30,2.34,2.34,1.56,0.78,0.00,0.78,1.56,1.56,1.56,0.78,1.56,1.56,1.56,1.56,0.78,1.56,0.00,0.00,0.00,0.78,0.78,0.00,0.00,0.00,0.78,0.00]
            ]
Exp_36_32 = [
            [0,18.75,23.05,34.38,45.31,53.13,59.38,56.25,61.33,57.03,64.06,66.02,67.58,68.36,71.48,70.31,71.48,74.61,76.56,75.78,73.44,76.95,74.61,72.66,76.17,76.56,76.17,75.00,78.13,80.47,81.25],
            [100,71.48,55.86,42.58,33.20,29.30,22.66,25.00,17.19,18.75,17.97,18.36,12.89,18.75,12.11,14.06,12.50,10.16,10.55,8.98,14.06,10.94,10.94,7.81,7.81,11.33,9.38,7.81,8.59,7.81,7.42],
            [0,15.63,18.75,17.58,13.28,15.23,13.28,12.50,10.94,11.72,13.28,11.72,11.72,10.16,9.38,9.38,11.72,9.38,6.25,10.16,7.03,7.03,7.03,10.16,9.38,7.03,8.59,10.16,9.38,8.59,8.59],
            ]
Exp_36_36 = [
            [0,20.31,36.33,37.89,48.83,50.39,55.86,55.08,56.64,57.42,55.08,51.56,59.77,58.20,53.52,57.03,58.20,55.47,62.11,59.77,54.69,52.73,51.95,52.34,51.56,51.56,52.73,51.17,53.52,51.95,56.64],
            [100,53.91,35.94,21.88,14.45,12.89,13.28,8.59,7.81,10.94,12.50,10.16,10.16,9.77,10.94,7.42,9.77,7.42,5.86,3.91,8.20,8.20,7.81,6.64,6.25,5.86,4.69,3.91,4.30,3.52,3.13],
            [0,20.31,31.64,35.16,33.59,39.45,36.33,37.50,35.94,33.20,33.59,33.59,30.86,28.91,35.16,32.42,32.42,34.77,35.94,37.11,37.11,39.84,40.23,41.41,40.63,43.75,39.45,42.97,39.06,41.02,37.11],
            ]

Switch = [
         [0.00,20.31,35.94,36.72,48.44,50.78,50.78,50.78,53.91,53.91,56.25,57.03,58.59,59.38,60.94,60.94,60.94,60.94,63.28,63.28,62.50,64.06,63.28,63.28,65.63,64.06,64.84,65.63,65.63,65.63,67.19,64.06,62.50,67.19,62.50,65.63,64.84,64.84,66.41,65.63,64.84,63.28,65.63,64.84,64.84,64.84,64.84,65.63,66.41,67.19,65.63,64.84,65.63,65.63,64.06,64.06,62.50,64.06,59.38,59.38,57.81,54.69,53.13,48.44,45.31,46.88,45.31,44.53,41.41,48.44,45.31,47.66,38.28,47.66,46.88,46.88,46.88,44.53,43.75,39.06,31.25,30.47,32.81,31.25,28.13,31.25,24.22,25.78,27.34,23.44,23.44,24.22,22.66,19.53,18.75,18.75,17.97,17.19,16.41,17.19,12.50,15.63,15.63,16.41,12.50,14.06],
         [100.00,64.06,43.75,39.84,25.00,20.31,20.31,15.63,17.97,13.28,11.72,9.38,14.06,14.06,11.72,12.50,10.16,8.59,12.50,7.03,9.38,7.81,6.25,6.25,6.25,6.25,7.81,7.03,6.25,6.25,6.25,6.25,7.81,7.81,5.47,6.25,4.69,6.25,4.69,8.59,6.25,4.69,6.25,7.81,7.81,7.03,7.03,9.38,10.94,10.94,11.72,10.94,9.38,9.38,13.28,15.63,14.06,12.50,12.50,12.50,17.19,14.06,17.19,17.97,17.19,17.19,15.63,17.19,13.28,15.63,21.88,17.19,15.63,15.63,14.06,13.28,15.63,15.63,15.63,14.06,14.06,14.06,10.16,7.81,13.28,16.41,15.63,17.19,18.75,20.31,17.97,14.06,21.88,18.75,17.19,20.31,18.75,17.19,15.63,20.31,16.41,12.50,15.63,16.41,14.06,14.06],
         [0.00,14.06,17.19,18.75,20.31,22.66,23.44,24.22,25.00,24.22,23.44,22.66,21.88,21.88,21.09,21.88,21.09,21.09,21.09,21.09,21.09,21.88,22.66,21.09,21.09,21.88,21.09,21.09,21.88,21.09,21.09,21.09,21.09,22.66,24.22,24.22,24.22,21.88,21.88,25.00,24.22,23.44,23.44,22.66,22.66,22.66,22.66,22.66,23.44,21.88,21.88,22.66,22.66,22.66,23.44,22.66,21.88,21.88,25.00,28.13,28.13,29.69,26.56,32.81,28.13,32.81,32.81,30.47,35.16,35.16,34.38,31.25,32.81,31.25,40.63,39.06,40.63,41.41,39.06,43.75,46.88,53.13,54.69,56.25,56.25,46.88,53.91,52.34,57.81,56.25,59.38,58.59,55.47,57.81,62.50,62.50,64.06,67.19,67.19,68.75,70.31,73.44,73.44,66.41,70.31,70.31]
         ]

timeline = np.linspace(0, 30, 31)

switch_timeline = np.linspace(0, 105, 106)


RealTemp_ALL = pd.read_csv("C:\\Users\\vstok\\Desktop\\WIPx_Arena_Gradienten.csv")
df = RealTemp_ALL.loc[:,'T Left']
print(df)
RealTemp_L = []
RealTemp_C =[]
RealTemp_R = []



#######_CONSTANTS_#################################################

x1 = 0.0062#33    # optimum lies @0.0132 as found in the program 09_02_21NEW_dara_AllAtOnce_fit
x2 = 1#1#0.5        # stopping probablity at a cage
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


least_squares_list = []

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

def sin_DECREASE(start, end, height, t):
    return Temp_l - 6 * (np.sin(t/40 - 10.13) ** 2) * PULSE(start, end, 1, t) - 6 * PULSE(104.99, M + 1, 1, t)

# #--------original----------------
# def exp_DECAY(start, end, height, t):
#     return Temp_l + (6 * np.exp(-(t - start)/40) - 6) * PULSE(start, end, height, t)

#------------copy-------------
def exp_DECAY(start, end, height, t):
    return Temp_l + (6 * np.exp(-(t - start)/45) - 6) * PULSE(start, end, height, t)

#print(exp_DECAY(30.00, 105.00, 1, switch_timeline))
plt.plot(exp_DECAY(30.00, 105.00, 1, switch_timeline))
plt.plot(df)
plt.show()

def Tl(t):
    return exp_DECAY(30.00, 105.00, 1, t)

#def Tl(t):
#    return sin_DECREASE(30.00, 104.98, 6, t)

def Tr(t):
    return Temp_r + PULSE(30, 50, 0, t)


# def Wl(t):
#     return (24/8*(Tl(t)-28))
# def Wr(t):
#     return (24/8*(Tr(t)-28))


A = 12
B = 1.2
C = 27
D = 13

def Wl(t):
    return (A * np.tanh((Tl(t)/B)-C)+D)

def Wr(t):
    return (A * np.tanh((Tr(t)/B)-C)+D)

plt.plot(Wl(switch_timeline))
plt.show()

def Area_curve_exp(list, t):
    area = 0
    for n in range(len(t)-1):
        trapezoid = ((list[n+1] + list[n]) * (t[n+1] - t[n])) / 2
        area += trapezoid
    return area

def Area_curve_ODE(list, t):
    area = 0
    for n in range(len(t)-1):
        trapezoid = ((list[n+1] + list[n]) * (t[n+1] - t[n])) / 2
        area += trapezoid
    return area

#plt.rcParams.update({'font.size': 10})
#plt.plot(Tl(switch_timeline), label='left')
#plt.plot(Tr(switch_timeline), label='right')
#plt.plot(df)
#plt.xlabel('time[minutes]', fontsize="10")
#plt.ylabel('temperature[°C]', fontsize="10")
#plt.legend(loc='best', fontsize="small")
#plt.show()


def euler_ODE_LFR():
    def f(u, t):
        L, F, R = u
        NOISE = 0#((np.random.random()-0.5)*5)
        rand_weight = np.random.random()
        noise_weight_L = NOISE /2#* rand_weight
        noise_weight_R = NOISE /2#* (1 - rand_weight)

        deL = (x1 * (F ** 2) / 2 + x1 * F * (L + x2 * Cl) - L / (Wl(t))) - noise_weight_L
        deF = (L / (Wl(t)) + R / (Wr(t)) - x1 * (F ** 2) - x1 * F * (L + x2 * Cl) - x1 * F * (R + x2 * Cr)) + NOISE # with the only F(ree) bees you could use (x1 PLUS x1)!!!!!!
        deR = (x1 * (F ** 2) / 2 + x1 * F * (R + x2 * Cr) - R / (Wr(t))) - noise_weight_R
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

    #ODE_List_L = L

    fig = plt.figure(figsize=(11,8))
    plt.rcParams.update({'font.size': 15})
    l1 = plt.plot(t, L, color='blue', label="model left", ls = "--")
    l2 = plt.plot(t, F, color='red', label="model center", ls = "--")
    l3 = plt.plot(t, R, color='peru', label="model right", ls = "--")
    e1 = plt.plot(switch_timeline, Switch[0], color='blue', ls = "solid", label="experiment left")
    e2 = plt.plot(switch_timeline, Switch[1], color='red', ls = "solid", label="experiment center")
    e3 = plt.plot(switch_timeline, Switch[2], color='peru', ls = "solid", label="experiment right")
    #fig.legend((l1, l2, l3, e1, e2, e3), ('ODE_L', 'ODE_F', 'ODE_R', 'EXP_L', 'EXP_F', 'EXP_R'))#, 'lower right')
    plt.legend(loc='best', fontsize="small")
    plt.xlabel('time[minutes]', fontsize="15")
    plt.ylabel('bees[%]', fontsize="15")
    #plt.title('Comparison of experimental data [36°C/30°C] and the solutions of the differential equations', fontsize=10)
    #save_results_to = r"C:\Users\vstok\Desktop\Atom-Projects\ODE_BEES\BEECLUST_cage"
    #plt.savefig(save_results_to + '\experiment3630.png', dpi = 300)
    plt.show()
    return(L, F, R)

varList_L = []
varList_F = []
varList_R = []
n = 1
for i in range(n):
    #x1 += 0.0001
    #x1 += 0.0001
    if __name__ == '__main__':
        ODE_L, ODE_F, ODE_R = euler_ODE_LFR()
        varList_L.append(ODE_L)
        varList_F.append(ODE_F)
        varList_R.append(ODE_R)

variance_L = []
mean_L = []
deviation_L = []
for ii in range(len(ODE_L)):
    calclist = []
    for iii in range(n):
        calclist.append(varList_L[iii][ii])
    variance_L.append(np.var(calclist))
    mean_L.append(np.mean(calclist))
    deviation_L.append(np.std(calclist))

variance_F = []
mean_F = []
deviation_F = []
for ii in range(len(ODE_F)):
    calclist = []
    for iii in range(n):
        calclist.append(varList_F[iii][ii])
    variance_F.append(np.var(calclist))
    mean_F.append(np.mean(calclist))
    deviation_F.append(np.std(calclist))

variance_R = []
mean_R = []
deviation_R = []
for ii in range(len(ODE_R)):
    calclist = []
    for iii in range(n):
        calclist.append(varList_R[iii][ii])
    variance_R.append(np.var(calclist))
    mean_R.append(np.mean(calclist))
    deviation_R.append(np.std(calclist))

for i in range(0):
    #x1 -= 0.0001
    #x1 -= 0.0001
    if __name__ == '__main__':
        ODE_L, ODE_F, ODE_R = euler_ODE_LFR()
        t = np.linspace(0, (len(ODE_L)-1), len(ODE_L))

        Diff_L = Area_curve_exp((Switch[0]), switch_timeline) - (Area_curve_ODE(ODE_L, t) * dx)
        Diff_F = Area_curve_exp((Switch[1]), switch_timeline) - (Area_curve_ODE(ODE_F, t) * dx)
        Diff_R = Area_curve_exp((Switch[2]), switch_timeline) - (Area_curve_ODE(ODE_R, t) * dx)
        Sq_D_L = np.sqrt(Diff_L ** 2)
        Sq_D_F = np.sqrt(Diff_F ** 2)
        Sq_D_R = np.sqrt(Diff_R ** 2)
        least_squares = Sq_D_L + Sq_D_F + Sq_D_R
        least_squares_list.append([x1, x1, least_squares])
print(least_squares_list)

x = np.linspace(0, (len(mean_F)-1), len(mean_F))

plt.errorbar(x, mean_L, deviation_L, linestyle='None', marker='.', color='red', label='L_ODE')
plt.errorbar(x, mean_F, deviation_F, linestyle='None', marker='.', color='orange', label='F_ODE')
plt.errorbar(x, mean_R, deviation_R, linestyle='None', marker='.', color='blue', label='R_ODE')
plt.legend(loc='best', fontsize="small")

plt.show()

# after running the program [36-32] we see that with this configuration we get an minimum at [0.0126, 0.0126, 299.5765]
# after running the program [36-30] we see that with this configuration we get an minimum at [0.0165, 0.0165, 235.4713]
