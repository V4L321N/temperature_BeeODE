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


new_Data_36_30 = pd.read_csv("C:\\Users\\vstok\\Desktop\\36_30.csv")

min_L_36_30 = new_Data_36_30.loc[:,'min_L']
q_one_L_36_30 = new_Data_36_30.loc[:,'q_one_L']
median_L_36_30 = new_Data_36_30.loc[:,'median_L']
q_three_L_36_30 = new_Data_36_30.loc[:,'q_three_L']
max_L_36_30 = new_Data_36_30.loc[:,'max_L']

min_C_36_30 = new_Data_36_30.loc[:,'min_C']
q_one_C_36_30 = new_Data_36_30.loc[:,'q_one_C']
median_C_36_30 = new_Data_36_30.loc[:,'median_C']
q_three_C_36_30 = new_Data_36_30.loc[:,'q_three_C']
max_C_36_30 = new_Data_36_30.loc[:,'max_C']

min_R_36_30 = new_Data_36_30.loc[:,'min_R']
q_one_R_36_30 = new_Data_36_30.loc[:,'q_one_R']
median_R_36_30 = new_Data_36_30.loc[:,'median_R']
q_three_R_36_30 = new_Data_36_30.loc[:,'q_three_R']
max_R_36_30 = new_Data_36_30.loc[:,'max_R']

################################

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

################################

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

timeline = np.linspace(0, 30, 31)

print(timeline)
print(median_L_36_36)
switch_timeline = np.linspace(0, 105, 106)


#######_CONSTANTS_#################################################

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

Temp_l = 36
Temp_r = 36


least_squares_list = []
x_list = []

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
    return np.heaviside(t - start, 0) * height - np.heaviside(t - end, 1) * height

def sin_DECREASE(start, end, height, t):
    return Temp_l - 6 * (np.sin(t/(11.00) + 47.39) ** 2) * PULSE(start, end, 1, t) - 6 * PULSE(49.99, M + 1, 1, t)

def Tl(t):
    return sin_DECREASE(30.00, 49.98, 6, t)

def Tr(t):
    return Temp_r + PULSE(30, 50, 0, t)

# def Wl(t):
#     return (((a + b * Tl(t)) ** c / ((a + b * Tl(t)) ** c + d ** c)) * e + f) / g
#
# def Wr(t):
#     return (((a + b * Tr(t)) ** c / ((a + b * Tr(t)) ** c + d ** c)) * e + f) / g
######################################################################################


A = 12
B = 1.2
C = 27
D = 13

def Wl_1():
    return (A * np.tanh((36/B)-C)+D)

def Wr_1():
    return (A * np.tanh((30/B)-C)+D)

def Wl_2():
    return (A * np.tanh((36/B)-C)+D)

def Wr_2():
    return (A * np.tanh((32/B)-C)+D)

def Wl_3():
    return (A * np.tanh((36/B)-C)+D)

def Wr_3():
    return (A * np.tanh((36/B)-C)+D)

# def Wl_1(t):
#     return (10 * np.tanh(Tl(t)-33)+15)
#
# def Wr_1(t):
#     return (10 * np.tanh(Tr(t)-33)+15)
#
# def Wl_2(t):
#     return (10 * np.tanh(Tl(t)-33)+15)
#
# def Wr_2(t):
#     return (10 * np.tanh(Tr(t)-33)+15)
#
# def Wl_3(t):
#     return (10 * np.tanh(Tl(t)-33)+15)
#
# def Wr_3(t):
#     return (10 * np.tanh(Tr(t)-33)+15)

# def Wl_1():
#     return (((a + b * 36) ** c / ((a + b * 36) ** c + d ** c)) * e + f) / g
#
# def Wr_1():
#     return (((a + b * 30) ** c / ((a + b * 30) ** c + d ** c)) * e + f) / g
#
# def Wl_2():
#     return (((a + b * 36) ** c / ((a + b * 36) ** c + d ** c)) * e + f) / g
#
# def Wr_2():
#     return (((a + b * 31) ** c / ((a + b * 31) ** c + d ** c)) * e + f) / g
#
# def Wl_3():
#     return (((a + b * 36) ** c / ((a + b * 36) ** c + d ** c)) * e + f) / g
#
# def Wr_3():
#     return (((a + b * 36) ** c / ((a + b * 36) ** c + d ** c)) * e + f) / g
######################################################################################

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


plt.plot(Tl(timeline))
plt.plot(Tr(timeline))
plt.show()


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

    fig = plt.figure(figsize=(11,8))
    l1, l2, l3 = plt.plot(t, (L*100/B), t, (F*100/B), t, (R*100/B))
    e1, e2, e3 = plt.plot(timeline, median_L_36_30, timeline, median_C_36_30, timeline, median_R_36_30, ls = "--")
    fig.legend((l1, l2, l3, e1, e2, e3), ('ODE_L', 'ODE_F', 'ODE_R', 'EXP_L', 'EXP_F', 'EXP_R'), 'lower right')
    plt.xlabel('time / minutes')
    plt.ylabel('bees / %')
    plt.title('Comparison of experimental data [36°C/30°C] and the solutions of the differential equations with x1='+str(round(x1,4)), fontsize=10)
    #save_results_to = r"C:\Users\vstok\Desktop\Atom-Projects\ODE_BEES\BEECLUST_cage"
    #plt.savefig(save_results_to + '\experiment3630.png', dpi = 300)
    plt.show()
    print(L)
    return(L, F, R)

def euler_ODE_LFR_2():
    def f(u, t):
        L, F, R = u
        # deL = (x1 * (F ** 2) / 2 + x1 * F * (L + x2 * Cl) - L / (Wl_1(Tl(t))))
        # deF = (L / (Wl_1(Tl(t))) + R / (Wr_1(Tr(t))) - x1 * (F ** 2) - x1 * F * (L + x2 * Cl) - x1 * F * (R + x2 * Cr))
        # deR = (x1 * (F ** 2) / 2 + x1 * F * (R + x2 * Cr) - R / (Wr_1(Tr(t))))
        deL = (x1 * (F ** 2) / 2 + x1 * F * (L + x2 * Cl) - L / (Wl_2()))
        deF = (L / (Wl_2()) + R / (Wr_2()) - x1 * (F ** 2) - x1 * F * (L + x2 * Cl) - x1 * F * (R + x2 * Cr)) # with the only F(ree) bees you could use (x1 PLUS x1)!!!!!!
        deR = (x1 * (F ** 2) / 2 + x1 * F * (R + x2 * Cr) - R / (Wr_2()))
        return [deL, deF, deR]

    dt = dx               # Step size
    D = M               # Duration in minutes
    N_t = int(D/dt)      # Corresponding number of steps
    T = dt*N_t           # End time
    B = 64                #number of bees
    U_0 = [0, B, 0]

    u, t = ode_FE(f, U_0, dt, T)

    L = u[:,0] * 100 / B
    F = u[:,1] * 100 / B
    R = u[:,2] * 100 / B

    fig = plt.figure(figsize=(11,8))
    l1, l2, l3 = plt.plot(t, L, t, F, t, R)
    e1, e2, e3 = plt.plot(timeline, median_L_36_32, timeline, median_C_36_32, timeline, median_R_36_32, ls = "--")
    fig.legend((l1, l2, l3, e1, e2, e3), ('ODE_L', 'ODE_F', 'ODE_R', 'EXP_L', 'EXP_F', 'EXP_R'), 'lower right')
    plt.xlabel('time / minutes')
    plt.ylabel('bees / %')
    plt.title('Comparison of experimental data [36°C/32°C] and the solutions of the differential equations with x1='+str(round(x1,4)), fontsize=10)
    #save_results_to = r"C:\Users\vstok\Desktop\Atom-Projects\ODE_BEES\BEECLUST_cage"
    #plt.savefig(save_results_to + '\experiment3630.png', dpi = 300)
    plt.show()
    return(L, F, R)

def euler_ODE_LFR_3():
    def f(u, t):
        L, F, R = u
        # deL = (x1 * (F ** 2) / 2 + x1 * F * (L + x2 * Cl) - L / (Wl_1(Tl(t))))
        # deF = (L / (Wl_1(Tl(t))) + R / (Wr_1(Tr(t))) - x1 * (F ** 2) - x1 * F * (L + x2 * Cl) - x1 * F * (R + x2 * Cr))
        # deR = (x1 * (F ** 2) / 2 + x1 * F * (R + x2 * Cr) - R / (Wr_1(Tr(t))))
        deL = (x1 * (F ** 2) / 2 + x1 * F * (L + x2 * Cl) - L / (Wl_3()))
        deF = (L / (Wl_3()) + R / (Wr_3()) - x1 * (F ** 2) - x1 * F * (L + x2 * Cl) - x1 * F * (R + x2 * Cr))   # with the only F(ree) bees you could use (x1 PLUS x1)!!!!!!
        deR = (x1 * (F ** 2) / 2 + x1 * F * (R + x2 * Cr) - R / (Wr_3()))
        return [deL, deF, deR]

    dt = dx               # Step size
    D = M               # Duration in minutes
    N_t = int(D/dt)      # Corresponding number of steps
    T = dt*N_t           # End time
    B = 64                #number of bees
    U_0 = [0, B, 0]

    u, t = ode_FE(f, U_0, dt, T)

    L = u[:,0] * 100 / B
    F = u[:,1] * 100 / B
    R = u[:,2] * 100 / B

    fig = plt.figure(figsize=(11,8))
    l1, l2, l3 = plt.plot(t, L, t, F, t, R)
    e1, e2, e3 = plt.plot(timeline, median_L_36_36, timeline, median_C_36_36, timeline, median_R_36_36, ls = "--")
    fig.legend((l1, l2, l3, e1, e2, e3), ('ODE_L', 'ODE_F', 'ODE_R', 'EXP_L', 'EXP_F', 'EXP_R'), 'lower right')
    plt.xlabel('time / minutes')
    plt.ylabel('bees / %')
    plt.title('Comparison of experimental data [36°C/36°C] and the solutions of the differential equations with x1='+str(round(x1,4)), fontsize=10)
    #save_results_to = r"C:\Users\vstok\Desktop\Atom-Projects\ODE_BEES\BEECLUST_cage"
    #plt.savefig(save_results_to + '\experiment3630.png', dpi = 300)

    plt.show()

    return(L, F, R)


if __name__ == '__main__':
    for i in range(1):
        #x1-=0.0001
        ODE_L_1, ODE_F_1, ODE_R_1 = euler_ODE_LFR_1()
        t = np.linspace(0, (len(ODE_L_1)-1), len(ODE_L_1))
        Diff_L_1 = Area_curve_exp((median_L_36_30), timeline) - (Area_curve_ODE(ODE_L_1, t) * dx)
        Diff_F_1 = Area_curve_exp((median_C_36_30), timeline) - (Area_curve_ODE(ODE_F_1, t) * dx)
        Diff_R_1 = Area_curve_exp((median_R_36_30), timeline) - (Area_curve_ODE(ODE_R_1, t) * dx)
        Sq_D_L_1 = np.sqrt(Diff_L_1 ** 2)
        Sq_D_F_1 = np.sqrt(Diff_F_1 ** 2)
        Sq_D_R_1 = np.sqrt(Diff_R_1 ** 2)

        ODE_L_2, ODE_F_2, ODE_R_2 = euler_ODE_LFR_2()
        t = np.linspace(0, (len(ODE_L_2)-1), len(ODE_L_2))
        Diff_L_2 = Area_curve_exp((median_L_36_32), timeline) - (Area_curve_ODE(ODE_L_2, t) * dx)
        Diff_F_2 = Area_curve_exp((median_C_36_32), timeline) - (Area_curve_ODE(ODE_F_2, t) * dx)
        Diff_R_2 = Area_curve_exp((median_R_36_32), timeline) - (Area_curve_ODE(ODE_R_2, t) * dx)
        Sq_D_L_2 = np.sqrt(Diff_L_2 ** 2)
        Sq_D_F_2 = np.sqrt(Diff_F_2 ** 2)
        Sq_D_R_2 = np.sqrt(Diff_R_2 ** 2)

        ODE_L_3, ODE_F_3, ODE_R_3 = euler_ODE_LFR_3()
        t = np.linspace(0, (len(ODE_L_3)-1), len(ODE_L_3))
        Diff_L_3 = Area_curve_exp((median_L_36_36), timeline) - (Area_curve_ODE(ODE_L_3, t) * dx)
        Diff_F_3 = Area_curve_exp((median_C_36_36), timeline) - (Area_curve_ODE(ODE_F_3, t) * dx)
        Diff_R_3 = Area_curve_exp((median_R_36_36), timeline) - (Area_curve_ODE(ODE_R_3, t) * dx)
        Sq_D_L_3 = np.sqrt(Diff_L_1 ** 2)
        Sq_D_F_3 = np.sqrt(Diff_F_1 ** 2)
        Sq_D_R_3 = np.sqrt(Diff_R_1 ** 2)

        least_squares = Sq_D_L_1 + Sq_D_F_1 + Sq_D_R_1 + Sq_D_L_2 + Sq_D_F_2 + Sq_D_R_2 + Sq_D_L_3 + Sq_D_F_3 + Sq_D_R_3
        least_squares_list.append(least_squares)
        x_list.append(x1)



print(least_squares_list, x_list)

# after running the program [36-32] we see that with this configuration we get a minimum at [x1, x1, min_Diff_Area] = [0.0126, 0.0126, 299.5765]
# after running the program [36-30] we see that with this configuration we get a minimum at [x1, x1, min_Diff_Area] = [0.0165, 0.0165, 235.4713]
# after running the program [36-36] we see that with this configuration we get a minimum at [x1, x1, min_Diff_Area] = [0.0162, 0.0162, 480.1426]
# after running the program and fitting them all together at once we get a minimum at 0.127.
# after running the program and fitting them all together WITH THE NEW DATA at once we get a minimum.
