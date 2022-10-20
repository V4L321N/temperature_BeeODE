import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = -2.28
b = 0.08
c = 5.15
d = 0.32
e = 28.5 #40 #50
f = 7
g = 1.4 #original was 15

Temperature = np.linspace(28,36,100)

def Tl(t):
    return sin_DECREASE(30.00, 49.98, 6, t)


# def W():
#     return (((a + b * Temperature) ** c / ((a + b * Temperature) ** c + d ** c)) * e + f) / g

A = 12
B = 1.2
C = 27
D = 13

def W():
    return (A * np.tanh((Temperature/B)-C)+D)

plt.plot(Temperature, W(), color = 'blue')
plt.xlabel('temperature [°C]', fontsize=12)
plt.ylabel('waiting time [s]', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
