import numpy as np
from motion import motion_profile
from plot import draw_plots
import matplotlib.pyplot as plt
from acting_force import acting_force
import pandas as pd

dt=1e-5
t = np.arange(0, 1.1, dt)
u=len(t)
g = 9.81
k = 0.213
step=5*1e3
Sstep= 4*step

x1 = []
x2 = []
curr2 = []
curr1 = []
curr3 = []
curr4 = []
Ftotal = []
x1 = np.zeros(int(u))
x2 = np.zeros(int(u))
curr1 = np.zeros(int(u))
curr2 = np.zeros(int(u))
curr3 = np.zeros(int(u))
curr4 = np.zeros(int(u))
Ftotal = np.zeros(int(u))


R=100
L=1.3e-3
m=(0.0075 + 5 * 0.006);
for i in range(u - 1):
    x1[i+1] = dt*x2[i] + x1[i]
    if (x1[i] < 0.0005):
        phase = 1
        curr = curr1[i]
        v1=18
        v2=0 
        v3=0
        v4 =0
        Fx = acting_force(x1[i], phase)
        Ftotal[i] = Fx

    elif ( 0.0005 < x1[i] and x1[i] < 0.001):
            # print('Phase 2 ....')
            phase = 2
            curr = curr2[i]
            v2=18
            v1 =0
            v3=0 
            v4 =0
            Fx = acting_force(x1[i], phase)
            Ftotal[i] = Fx

    elif (0.001 < x1[i] and x1[i] < 0.0015):
            # print('Phase 3 ....')
            phase = 3
            curr = curr3[i]
            v3 = -18
            v2 =0
            v1=0 
            v4 =0
            Fx = acting_force(x1[i], phase)
            Ftotal[i] = Fx

    elif (0.0015 < x1[i] and x1[i] < 0.002):
            # print('phase 4 .....')
            phase = 4
            curr = curr4[i]
            v4 = -18
            v2 =0
            v3=0
            v1 =0
            Fx = acting_force(x1[i], phase)
            Ftotal[i] = Fx

    curr1[i+1]= dt*(((-R/L)*curr1[i]) + v1*(1/L)) + curr1[i]
    curr2[i+1]= dt*(((-R/L)*curr2[i]) + v2*(1/L)) + curr2[i]
    curr3[i+1]= dt*(((-R/L)*curr3[i]) + v3*(1/L)) + curr3[i]
    curr4[i+1]= dt*(((-R/L)*curr4[i]) + v4*(1/L)) + curr4[i]
    x2[i+1]= (dt*Ftotal[i]*(curr/0.1))/m + x2[i]


curr1 = [x for x in curr1 if x!=0]
print (np.shape(curr1))
# plt.plot(curr1)
# plt.show()


curr2 = [x for x in curr2 if x!=0]
print (np.shape(curr2))
# curr2.transpose()
# plt.plot(curr2)
# plt.show()


curr3 = [x for x in curr3 if x!=0]
# curr3.transpose()
# plt.plot(curr3)
# plt.show()

curr4 = [x for x in curr4 if x !=0]
# curr4.transpose()
# plt.plot(curr4)
# plt.show()

current = []
current = curr1 + curr2 + curr3 + curr4
print(np.shape(current))

index = int(len(x1)) +1
current = current[1:index]

plt.plot(x1, current)
plt.show()

# current.append(curr1)
# current.append(curr2)
# current.append(curr3)
# current.append(curr4)

df = pd.DataFrame(current)
# df = df.transpose()
print(df)
print(df.shape)

# plt.plot(x1,current)
# plt.show()
# plt.plot(t,x1)
# plt.show()

# plt.plot(x1,Ftotal)
# plt.show()

# plt.plot(x1,curr1)
# plt.plot(x1,curr2)
# plt.plot(x1,curr3)
# plt.plot(x1,curr4)
# plt.legend(['current1','current2','current3','current4'])
# plt.show()

# plt.plot(t,x2)
# plt.show()

# curr2.reverse()
# plt.plot(curr2)
# plt.show()