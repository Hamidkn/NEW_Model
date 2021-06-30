	System Description
In this section, the system will be described. The Micro-robots are moving on the coplanar coils which are actuated by magnetic field. As any motion problems the Newton’s second law was used to move Micro-robots. The equations are as follow: 
Ldi/dt= -Ri+u
v=  ds/dt         ,     a=  dv/dt    ,   and        F=f(i)
F=ma 

Where L is inductance, R is resistant, u is voltage. m refers to mass, a stands as the accelerator, and v is the derivative of position and F is total force. f(i) stands as the force as a function of current. 

i(t+1)= ∆t*((-R)/L i_1 (t)+ 1/L  u(t))+i(t)

Micro-robots will move forward by applying current in each coils and measuring forces. The Newton’s second law was used to measure the position of Micro-robots and from the equation (2) the final equation was achieved. 

mx ̈=F(x)*  (i(t))/0.1 

The Rung-Kuta method was used to solve this ode problem. 

x_1 (t+1)= ∆t*(x_2 )+ x_1 (t)
x_2 (t+1)=∆t*(1/m* F(x)* i(t)/0.1)+x_2 (t)

![](./../position.png)

![](./../Force.png)

![](./../current.png)

![](./../velocity.png)
