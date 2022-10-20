import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Target_left	= [0.013157652525234,0.118418872727109,0.092103567676641,0,0.026315305050469,0.026315305050469,0,0,0.065788262626172,0.105261220201875,0.071544735605962,0.157891830302812,0.032071778030259, 0.078945915151406,0.03453883787874,0.032071778030259,0.108550633333183,0.05180825681811,0.046874137121147,0.044407077272666]
Target_right = [0.197364787878515,0,0.026315305050469,0,0,0.013157652525234,0,0.026315305050469,0.013157652525234,0,0.014802359090889,0.05180825681811,0.002467059848481,0.022203538636333,0.012335299242407,0.01726941893937,0.009868239393926,0.022203538636333,0.009868239393926,0.014802359090889]
Mid_merged = [0.015192341060064,0.028485639487621,0.028485639487621,0.045577023180193,0.041778937915177,0.039879895282669,0.045577023180193,0.041778937915177,0.034182767385145,0.030384682120129,0.033114555904359,0.015311031224596,0.04059203626986,0.030978132942788,0.038811683801883,0.038455613308288,0.028485639487621,0.034894908372336,0.037387401827502,0.037031331333907]

#29 °C; 24/128 bees
#%bees/cm2   	Min    	1st quartile	Median	3rd quartile	Max
# target_left = [0.00,0.03,0.05,0.08,0.16]
# target_right = [0.00,0.00,0.01,0.02,0.20]
# mid_merged = [0.02,0.03,0.04,0.04,0.05]

ticks = ["target left", "mid merged", "target right"]

plt.boxplot(Target_left)
plt.boxplot(Target_right)
plt.boxplot(Mid_merged)
plt.show()

data_a = [Target_left, Target_right, Mid_merged]

ticks = ['target_left', 'mid_merged', 'target_right']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color="black")
    plt.setp(bp['whiskers'], color="black")
    plt.setp(bp['caps'], color="orange")
    plt.setp(bp['medians'], color="red")

plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0, sym='', widths=0.9)

set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/


# # draw temporary red and blue lines and use them to create a legend
# plt.plot([], c='#D7191C', label='Apples')
# plt.plot([], c='#2C7BB6', label='Oranges')
# plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(-0.01, 0.3)
plt.tight_layout()
plt.show()


#print(len(Target_left), len(Target_right), len(Mid_merged))
