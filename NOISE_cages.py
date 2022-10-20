from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#######_EXPERIMENT_DATA_#######################################

new_Data_36_32 = pd.read_csv("C:\\Users\\vstok\\Desktop\\36_32.csv")

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

minute_zero_L	= [min_L_36_32[0],q_one_L_36_32[0],median_L_36_32[0],q_three_L_36_32[0],max_L_36_32[0]]
minute_fifteen_L = [min_L_36_32[15],q_one_L_36_32[15],median_L_36_32[15],q_three_L_36_32[15],max_L_36_32[15]]
minute_thirty_L = [min_L_36_32[30],q_one_L_36_32[30],median_L_36_32[30],q_three_L_36_32[30],max_L_36_32[30]]

minute_zero_C	= [min_C_36_32[0],q_one_C_36_32[0],median_C_36_32[0],q_three_C_36_32[0],max_C_36_32[0]]
minute_fifteen_C = [min_C_36_32[15],q_one_C_36_32[15],median_C_36_32[15],q_three_C_36_32[15],max_C_36_32[15]]
minute_thirty_C = [min_C_36_32[30],q_one_C_36_32[30],median_C_36_32[30],q_three_C_36_32[30],max_C_36_32[30]]

minute_zero_R	= [min_R_36_32[0],q_one_R_36_32[0],median_R_36_32[0],q_three_R_36_32[0],max_R_36_32[0]]
minute_fifteen_R = [min_R_36_32[15],q_one_R_36_32[15],median_R_36_32[15],q_three_R_36_32[15],max_R_36_32[15]]
minute_thirty_R = [min_R_36_32[30],q_one_R_36_32[30],median_R_36_32[30],q_three_R_36_32[30],max_R_36_32[30]]

data_L = [minute_zero_L, minute_fifteen_L, minute_thirty_L]
data_C = [minute_zero_C, minute_fifteen_C, minute_thirty_C]
data_R = [minute_zero_R, minute_fifteen_R, minute_thirty_R]

#######_CONSTANTS_#################################################
timeline = np.linspace(0, 30, 31)

x1 = 0.0056#33    # optimum lies @0.0132 as found in the program 09_02_21NEW_dara_AllAtOnce_fit
x2 = 1       # stopping probablity at a cage
Cr = 0#5            # number of caged bees right...
Cl = 0            # ...and left

a = -2.28
b = 0.08
c = 5.15
d = 0.32
e = 28.5 #40 #50
f = 7
g = 1.5 #original was 15

dx = 1    # step size for forward gauß algorithm
M = 30        # depending in experiment eiter 30 or 105 time steps (minutes)
N_t = int(M / dx) # number of steps for the algorithm

Temp_l = 36#35.8
Temp_r = 32#32.2
B = 24
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
    return Temp_l + (6 * np.exp(-(t - start)/35) - 6) * PULSE(start, end, height, t)

def Tl(t):
    return exp_DECAY(30.00, 105.00, 1, t)

def Tr(t):
    return Temp_r + PULSE(30, 50, 0, t)

# def Wl(t):
#    return (((a + b * Tl(t)) ** c / ((a + b * Tl(t)) ** c + d ** c)) * e + f) / g
#
# def Wr(t):
#    return (((a + b * Tr(t)) ** c / ((a + b * Tr(t)) ** c + d ** c)) * e + f) / g

def Wl(t):
    return (12 * np.tanh(Tl(t)/1.2-27)+13)

def Wr(t):
    return (12 * np.tanh(Tr(t)/1.2-27)+13)



def euler_ODE_LFR():
    def f(u, t):
        L, F, R = u
        # mu, sigma = 1, 0.3
        # z1 = float(np.random.normal(mu, sigma, 1))
        # z2 = float(np.random.normal(mu, sigma, 1))
        # z3 = float(np.random.normal(mu, sigma, 1))
        # z1 = 1
        # z2 = 1
        # z3 = 1
        shape, scale = 4, 0.5
        z1 = float(np.random.gamma(shape, scale, 1))
        z2 = float(np.random.gamma(shape, scale, 1))
        z3 = float(np.random.gamma(shape, scale, 1))

        n = 1
        epsilon_1 = n * z1
        epsilon_2 = n * z2
        epsilon_3 = n * z3

        deL = (epsilon_1 * x1 * (F ** 2) / 2 + epsilon_2 * x1 * F * (L + x2 * Cl) - L / (Wl(t)))
        deF = (L / (Wl(t)) + R / (Wr(t)) - epsilon_1 * x1 * (F ** 2) - epsilon_2 * x1 * F * (L + x2 * Cl) - epsilon_3 * x1 * F * (R + x2 * Cr))
        deR = (epsilon_1 * x1 * (F ** 2) / 2 + epsilon_3 * x1 * F * (R + x2 * Cr) - R / (Wr(t)))
        #print(round(deL+deF+deR), min(z1,z2,z3))
        return [deL, deF, deR]

    dt = dx               # Step size
    D = M               # Duration in minutes
    N_t = int(D/dt)      # Corresponding number of steps
    T = dt*N_t           # End time

    U_0 = [0, B, 0]

    u, t = ode_FE(f, U_0, dt, T)

    L = u[:,0]
    F = u[:,1]
    R = u[:,2]


    #fig = plt.figure(figsize=(11,8))
    #plt.rcParams.update({'font.size': 15})
    #l1 = plt.plot(t, L, color='red', label="L_ODE")
    #l2 = plt.plot(t, F, color='orange', label="F_ODE")
    #l3 = plt.plot(t, R, color='blue', label="R_ODE")
    #e1 = plt.plot(timeline, np.linspace(0,30,31), color='red', ls = "--", label="L_EXP")
    #e2 = plt.plot(timeline, 2, color='orange', ls = "--", label="F_EXP")
    #e3 = plt.plot(timeline, 3, color='blue', ls = "--", label="R_EXP")
    # plt.legend(loc='best', fontsize="small")
    # plt.xlabel('time [min]', fontsize="12")
    # plt.ylabel('bees [%]', fontsize="12")
    # np.savetxt('cage' + str(name_number) + '.csv', u, delimiter=',', header="L,C,R", comments='')
    np.savetxt('NOcage' + str(name_number) + '.csv', (u * 100 / B), delimiter=',', header="L,C,R", comments='')
    # plt.show()
    return(L, F, R)


for i in range(20):
    #x1 -= 0.0001
    if __name__ == '__main__':
        ODE_L, ODE_F, ODE_R = euler_ODE_LFR()
        name_number +=1


# ticks = ['0', '15', '30']
#
# def set_box_color(bp, color):
#     plt.setp(bp['boxes'], color="black")
#     plt.setp(bp['whiskers'], color="black")
#     plt.setp(bp['caps'], color="black")
#     plt.setp(bp['medians'], color="red")
#
# plt.figure()
#
# bpl_L = plt.boxplot(data_L, positions=np.array(range(len(data_L)))*2.0, sym='x', widths=0.8, whis=10)
# set_box_color(bpl_L, '#D7191C') # colors are from http://colorbrewer2.org/
# plt.xticks(range(0, len(ticks) * 2, 2), ticks)
# plt.xlim(-2, len(ticks)*2)
# plt.ylim(-1, 110)
# plt.xlabel('time[min]')
# plt.ylabel('median fraction of bees[%]')
# plt.tight_layout()
# plt.show()
#
# bpl_C = plt.boxplot(data_C, positions=np.array(range(len(data_C)))*2, sym='x', widths=0.8, whis=10)
# set_box_color(bpl_C, '#D7191C') # colors are from http://colorbrewer2.org/
# plt.xticks(range(0, len(ticks) * 2, 2), ticks)
# plt.xlim(-2, len(ticks)*2)
# plt.ylim(-1, 110)
# plt.xlabel('time[min]')
# plt.ylabel('median fraction of bees[%]')
# plt.tight_layout()
# plt.show()
#
# bpl_R = plt.boxplot(data_R, positions=np.array(range(len(data_R)))*2.0, sym='x', widths=0.8, whis=10)
# set_box_color(bpl_R, '#D7191C') # colors are from http://colorbrewer2.org/
# plt.xticks(range(0, len(ticks) * 2, 2), ticks)
# plt.xlim(-2, len(ticks)*2)
# plt.ylim(-1, 110)
# plt.xlabel('time[min]')
# plt.ylabel('median fraction of bees[%]')
# plt.tight_layout()
# plt.show()
#
# plt.show()
