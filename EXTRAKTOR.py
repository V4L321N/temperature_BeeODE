from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

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


SAMPLE_0_L_30 = []
SAMPLE_0_L_30.append((NOISE_SWITCH_DATA_0.loc[30,'L']))
SAMPLE_0_L_105 = []
SAMPLE_0_L_105.append((NOISE_SWITCH_DATA_0.loc[105,'L']))

SAMPLE_0_C_30 = []
SAMPLE_0_C_30.append((NOISE_SWITCH_DATA_0.loc[30,'C']))
SAMPLE_0_C_105 = []
SAMPLE_0_C_105.append((NOISE_SWITCH_DATA_0.loc[105,'C']))

SAMPLE_0_R_30 = []
SAMPLE_0_R_30.append((NOISE_SWITCH_DATA_0.loc[30,'R']))
SAMPLE_0_R_105 = []
SAMPLE_0_R_105.append((NOISE_SWITCH_DATA_0.loc[105,'R']))

"""###############################"""

SAMPLE_1_L_30 = []
SAMPLE_1_L_30.append((NOISE_SWITCH_DATA_1.loc[30,'L']))
SAMPLE_1_L_105 = []
SAMPLE_1_L_105.append((NOISE_SWITCH_DATA_1.loc[105,'L']))

SAMPLE_1_C_30 = []
SAMPLE_1_C_30.append((NOISE_SWITCH_DATA_1.loc[30,'C']))
SAMPLE_1_C_105 = []
SAMPLE_1_C_105.append((NOISE_SWITCH_DATA_1.loc[105,'C']))

SAMPLE_1_R_30 = []
SAMPLE_1_R_30.append((NOISE_SWITCH_DATA_1.loc[30,'R']))
SAMPLE_1_R_105 = []
SAMPLE_1_R_105.append((NOISE_SWITCH_DATA_1.loc[105,'R']))

"""###############################"""

SAMPLE_2_L_30 = []
SAMPLE_2_L_30.append((NOISE_SWITCH_DATA_2.loc[30,'L']))
SAMPLE_2_L_105 = []
SAMPLE_2_L_105.append((NOISE_SWITCH_DATA_2.loc[105,'L']))

SAMPLE_2_C_30 = []
SAMPLE_2_C_30.append((NOISE_SWITCH_DATA_2.loc[30,'C']))
SAMPLE_2_C_105 = []
SAMPLE_2_C_105.append((NOISE_SWITCH_DATA_2.loc[105,'C']))

SAMPLE_2_R_30 = []
SAMPLE_2_R_30.append((NOISE_SWITCH_DATA_2.loc[30,'R']))
SAMPLE_2_R_105 = []
SAMPLE_2_R_105.append((NOISE_SWITCH_DATA_2.loc[105,'R']))

"""###############################"""

SAMPLE_3_L_30 = []
SAMPLE_3_L_30.append((NOISE_SWITCH_DATA_3.loc[30,'L']))
SAMPLE_3_L_105 = []
SAMPLE_3_L_105.append((NOISE_SWITCH_DATA_3.loc[105,'L']))

SAMPLE_3_C_30 = []
SAMPLE_3_C_30.append((NOISE_SWITCH_DATA_3.loc[30,'C']))
SAMPLE_3_C_105 = []
SAMPLE_3_C_105.append((NOISE_SWITCH_DATA_3.loc[105,'C']))

SAMPLE_3_R_30 = []
SAMPLE_3_R_30.append((NOISE_SWITCH_DATA_3.loc[30,'R']))
SAMPLE_3_R_105 = []
SAMPLE_3_R_105.append((NOISE_SWITCH_DATA_3.loc[105,'R']))

"""###############################"""

SAMPLE_4_L_30 = []
SAMPLE_4_L_30.append((NOISE_SWITCH_DATA_4.loc[30,'L']))
SAMPLE_4_L_105 = []
SAMPLE_4_L_105.append((NOISE_SWITCH_DATA_4.loc[105,'L']))

SAMPLE_4_C_30 = []
SAMPLE_4_C_30.append((NOISE_SWITCH_DATA_4.loc[30,'C']))
SAMPLE_4_C_105 = []
SAMPLE_4_C_105.append((NOISE_SWITCH_DATA_4.loc[105,'C']))

SAMPLE_4_R_30 = []
SAMPLE_4_R_30.append((NOISE_SWITCH_DATA_4.loc[30,'R']))
SAMPLE_4_R_105 = []
SAMPLE_4_R_105.append((NOISE_SWITCH_DATA_4.loc[105,'R']))
"""###############################"""

SAMPLE_5_L_30 = []
SAMPLE_5_L_30.append((NOISE_SWITCH_DATA_5.loc[30,'L']))
SAMPLE_5_L_105 = []
SAMPLE_5_L_105.append((NOISE_SWITCH_DATA_5.loc[105,'L']))

SAMPLE_5_C_30 = []
SAMPLE_5_C_30.append((NOISE_SWITCH_DATA_5.loc[30,'C']))
SAMPLE_5_C_105 = []
SAMPLE_5_C_105.append((NOISE_SWITCH_DATA_5.loc[105,'C']))

SAMPLE_5_R_30 = []
SAMPLE_5_R_30.append((NOISE_SWITCH_DATA_5.loc[30,'R']))
SAMPLE_5_R_105 = []
SAMPLE_5_R_105.append((NOISE_SWITCH_DATA_5.loc[105,'R']))

"""###############################"""

SAMPLE_6_L_30 = []
SAMPLE_6_L_30.append((NOISE_SWITCH_DATA_6.loc[30,'L']))
SAMPLE_6_L_105 = []
SAMPLE_6_L_105.append((NOISE_SWITCH_DATA_6.loc[105,'L']))

SAMPLE_6_C_30 = []
SAMPLE_6_C_30.append((NOISE_SWITCH_DATA_6.loc[30,'C']))
SAMPLE_6_C_105 = []
SAMPLE_6_C_105.append((NOISE_SWITCH_DATA_6.loc[105,'C']))

SAMPLE_6_R_30 = []
SAMPLE_6_R_30.append((NOISE_SWITCH_DATA_6.loc[30,'R']))
SAMPLE_6_R_105 = []
SAMPLE_6_R_105.append((NOISE_SWITCH_DATA_6.loc[105,'R']))

"""###############################"""

SAMPLE_7_L_30 = []
SAMPLE_7_L_30.append((NOISE_SWITCH_DATA_7.loc[30,'L']))
SAMPLE_7_L_105 = []
SAMPLE_7_L_105.append((NOISE_SWITCH_DATA_7.loc[105,'L']))

SAMPLE_7_C_30 = []
SAMPLE_7_C_30.append((NOISE_SWITCH_DATA_7.loc[30,'C']))
SAMPLE_7_C_105 = []
SAMPLE_7_C_105.append((NOISE_SWITCH_DATA_7.loc[105,'C']))

SAMPLE_7_R_30 = []
SAMPLE_7_R_30.append((NOISE_SWITCH_DATA_7.loc[30,'R']))
SAMPLE_7_R_105 = []
SAMPLE_7_R_105.append((NOISE_SWITCH_DATA_7.loc[105,'R']))

"""###############################"""

SAMPLE_8_L_30 = []
SAMPLE_8_L_30.append((NOISE_SWITCH_DATA_8.loc[30,'L']))
SAMPLE_8_L_105 = []
SAMPLE_8_L_105.append((NOISE_SWITCH_DATA_8.loc[105,'L']))

SAMPLE_8_C_30 = []
SAMPLE_8_C_30.append((NOISE_SWITCH_DATA_8.loc[30,'C']))
SAMPLE_8_C_105 = []
SAMPLE_8_C_105.append((NOISE_SWITCH_DATA_8.loc[105,'C']))

SAMPLE_8_R_30 = []
SAMPLE_8_R_30.append((NOISE_SWITCH_DATA_8.loc[30,'R']))
SAMPLE_8_R_105 = []
SAMPLE_8_R_105.append((NOISE_SWITCH_DATA_8.loc[105,'R']))
"""###############################"""

SAMPLE_9_L_30 = []
SAMPLE_9_L_30.append((NOISE_SWITCH_DATA_9.loc[30,'L']))
SAMPLE_9_L_105 = []
SAMPLE_9_L_105.append((NOISE_SWITCH_DATA_9.loc[105,'L']))

SAMPLE_9_C_30 = []
SAMPLE_9_C_30.append((NOISE_SWITCH_DATA_9.loc[30,'C']))
SAMPLE_9_C_105 = []
SAMPLE_9_C_105.append((NOISE_SWITCH_DATA_9.loc[105,'C']))

SAMPLE_9_R_30 = []
SAMPLE_9_R_30.append((NOISE_SWITCH_DATA_9.loc[30,'R']))
SAMPLE_9_R_105 = []
SAMPLE_9_R_105.append((NOISE_SWITCH_DATA_9.loc[105,'R']))

"""###############################"""

SAMPLE_10_L_30 = []
SAMPLE_10_L_30.append((NOISE_SWITCH_DATA_10.loc[30,'L']))
SAMPLE_10_L_105 = []
SAMPLE_10_L_105.append((NOISE_SWITCH_DATA_10.loc[105,'L']))

SAMPLE_10_C_30 = []
SAMPLE_10_C_30.append((NOISE_SWITCH_DATA_10.loc[30,'C']))
SAMPLE_10_C_105 = []
SAMPLE_10_C_105.append((NOISE_SWITCH_DATA_10.loc[105,'C']))

SAMPLE_10_R_30 = []
SAMPLE_10_R_30.append((NOISE_SWITCH_DATA_10.loc[30,'R']))
SAMPLE_10_R_105 = []
SAMPLE_10_R_105.append((NOISE_SWITCH_DATA_10.loc[105,'R']))

"""###############################"""

SAMPLE_11_L_30 = []
SAMPLE_11_L_30.append((NOISE_SWITCH_DATA_11.loc[30,'L']))
SAMPLE_11_L_105 = []
SAMPLE_11_L_105.append((NOISE_SWITCH_DATA_11.loc[105,'L']))

SAMPLE_11_C_30 = []
SAMPLE_11_C_30.append((NOISE_SWITCH_DATA_11.loc[30,'C']))
SAMPLE_11_C_105 = []
SAMPLE_11_C_105.append((NOISE_SWITCH_DATA_11.loc[105,'C']))

SAMPLE_11_R_30 = []
SAMPLE_11_R_30.append((NOISE_SWITCH_DATA_11.loc[30,'R']))
SAMPLE_11_R_105 = []
SAMPLE_11_R_105.append((NOISE_SWITCH_DATA_11.loc[105,'R']))

"""###############################"""

SAMPLE_12_L_30 = []
SAMPLE_12_L_30.append((NOISE_SWITCH_DATA_12.loc[30,'L']))
SAMPLE_12_L_105 = []
SAMPLE_12_L_105.append((NOISE_SWITCH_DATA_12.loc[105,'L']))

SAMPLE_12_C_30 = []
SAMPLE_12_C_30.append((NOISE_SWITCH_DATA_12.loc[30,'C']))
SAMPLE_12_C_105 = []
SAMPLE_12_C_105.append((NOISE_SWITCH_DATA_12.loc[105,'C']))

SAMPLE_12_R_30 = []
SAMPLE_12_R_30.append((NOISE_SWITCH_DATA_12.loc[30,'R']))
SAMPLE_12_R_105 = []
SAMPLE_12_R_105.append((NOISE_SWITCH_DATA_12.loc[105,'R']))
"""###############################"""

SAMPLE_13_L_30 = []
SAMPLE_13_L_30.append((NOISE_SWITCH_DATA_13.loc[30,'L']))
SAMPLE_13_L_105 = []
SAMPLE_13_L_105.append((NOISE_SWITCH_DATA_13.loc[105,'L']))

SAMPLE_13_C_30 = []
SAMPLE_13_C_30.append((NOISE_SWITCH_DATA_13.loc[30,'C']))
SAMPLE_13_C_105 = []
SAMPLE_13_C_105.append((NOISE_SWITCH_DATA_13.loc[105,'C']))

SAMPLE_13_R_30 = []
SAMPLE_13_R_30.append((NOISE_SWITCH_DATA_13.loc[30,'R']))
SAMPLE_13_R_105 = []
SAMPLE_13_R_105.append((NOISE_SWITCH_DATA_13.loc[105,'R']))

"""###############################"""

SAMPLE_14_L_30 = []
SAMPLE_14_L_30.append((NOISE_SWITCH_DATA_14.loc[30,'L']))
SAMPLE_14_L_105 = []
SAMPLE_14_L_105.append((NOISE_SWITCH_DATA_14.loc[105,'L']))

SAMPLE_14_C_30 = []
SAMPLE_14_C_30.append((NOISE_SWITCH_DATA_14.loc[30,'C']))
SAMPLE_14_C_105 = []
SAMPLE_14_C_105.append((NOISE_SWITCH_DATA_14.loc[105,'C']))

SAMPLE_14_R_30 = []
SAMPLE_14_R_30.append((NOISE_SWITCH_DATA_14.loc[30,'R']))
SAMPLE_14_R_105 = []
SAMPLE_14_R_105.append((NOISE_SWITCH_DATA_14.loc[105,'R']))

"""###############################"""

SAMPLE_15_L_30 = []
SAMPLE_15_L_30.append((NOISE_SWITCH_DATA_15.loc[30,'L']))
SAMPLE_15_L_105 = []
SAMPLE_15_L_105.append((NOISE_SWITCH_DATA_15.loc[105,'L']))

SAMPLE_15_C_30 = []
SAMPLE_15_C_30.append((NOISE_SWITCH_DATA_15.loc[30,'C']))
SAMPLE_15_C_105 = []
SAMPLE_15_C_105.append((NOISE_SWITCH_DATA_15.loc[105,'C']))

SAMPLE_15_R_30 = []
SAMPLE_15_R_30.append((NOISE_SWITCH_DATA_15.loc[30,'R']))
SAMPLE_15_R_105 = []
SAMPLE_15_R_105.append((NOISE_SWITCH_DATA_15.loc[105,'R']))

"""###############################"""

SAMPLE_16_L_30 = []
SAMPLE_16_L_30.append((NOISE_SWITCH_DATA_16.loc[30,'L']))
SAMPLE_16_L_105 = []
SAMPLE_16_L_105.append((NOISE_SWITCH_DATA_16.loc[105,'L']))

SAMPLE_16_C_30 = []
SAMPLE_16_C_30.append((NOISE_SWITCH_DATA_16.loc[30,'C']))
SAMPLE_16_C_105 = []
SAMPLE_16_C_105.append((NOISE_SWITCH_DATA_16.loc[105,'C']))

SAMPLE_16_R_30 = []
SAMPLE_16_R_30.append((NOISE_SWITCH_DATA_16.loc[30,'R']))
SAMPLE_16_R_105 = []
SAMPLE_16_R_105.append((NOISE_SWITCH_DATA_16.loc[105,'R']))
"""###############################"""

SAMPLE_17_L_30 = []
SAMPLE_17_L_30.append((NOISE_SWITCH_DATA_17.loc[30,'L']))
SAMPLE_17_L_105 = []
SAMPLE_17_L_105.append((NOISE_SWITCH_DATA_17.loc[105,'L']))

SAMPLE_17_C_30 = []
SAMPLE_17_C_30.append((NOISE_SWITCH_DATA_17.loc[30,'C']))
SAMPLE_17_C_105 = []
SAMPLE_17_C_105.append((NOISE_SWITCH_DATA_17.loc[105,'C']))

SAMPLE_17_R_30 = []
SAMPLE_17_R_30.append((NOISE_SWITCH_DATA_17.loc[30,'R']))
SAMPLE_17_R_105 = []
SAMPLE_17_R_105.append((NOISE_SWITCH_DATA_17.loc[105,'R']))

"""###############################"""

SAMPLE_18_L_30 = []
SAMPLE_18_L_30.append((NOISE_SWITCH_DATA_18.loc[30,'L']))
SAMPLE_18_L_105 = []
SAMPLE_18_L_105.append((NOISE_SWITCH_DATA_18.loc[105,'L']))

SAMPLE_18_C_30 = []
SAMPLE_18_C_30.append((NOISE_SWITCH_DATA_18.loc[30,'C']))
SAMPLE_18_C_105 = []
SAMPLE_18_C_105.append((NOISE_SWITCH_DATA_18.loc[105,'C']))

SAMPLE_18_R_30 = []
SAMPLE_18_R_30.append((NOISE_SWITCH_DATA_18.loc[30,'R']))
SAMPLE_18_R_105 = []
SAMPLE_18_R_105.append((NOISE_SWITCH_DATA_18.loc[105,'R']))

"""###############################"""

SAMPLE_19_L_30 = []
SAMPLE_19_L_30.append((NOISE_SWITCH_DATA_19.loc[30,'L']))
SAMPLE_19_L_105 = []
SAMPLE_19_L_105.append((NOISE_SWITCH_DATA_19.loc[105,'L']))

SAMPLE_19_C_30 = []
SAMPLE_19_C_30.append((NOISE_SWITCH_DATA_19.loc[30,'C']))
SAMPLE_19_C_105 = []
SAMPLE_19_C_105.append((NOISE_SWITCH_DATA_19.loc[105,'C']))

SAMPLE_19_R_30 = []
SAMPLE_19_R_30.append((NOISE_SWITCH_DATA_19.loc[30,'R']))
SAMPLE_19_R_105 = []
SAMPLE_19_R_105.append((NOISE_SWITCH_DATA_19.loc[105,'R']))



SAMPLE_ALL_L = [SAMPLE_0_L_30, SAMPLE_1_L_30]
# np.savetxt('MASZdataL.csv', SAMPLE_ALL_L, delimiter=',', header="L,C,R", comments='')
#
# SAMPLE_ALL_C = 0
# SAMPLE_ALL_R = 0
#
# print(SAMPLE_ALL_L)
