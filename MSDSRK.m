function O=MSDSRK(m,F0,i,x0,v0)
%
% Solver for Mass-Sprring-Damper System with Runge-Kutta Method
% ----- Input argument -----
% m: mass for particle
% b: damping coefficient
% k: spring constant
% F0: amplitude of external force
% w: angular freuency of external force 
% x0: initial condition for the position x(0)
% v0: initial condition for the velocity v(0)
% ----- Output argument -----
% t: time series with given time step from ti to tf.
% x: state variable matrix for corresponding time t matrix
% define time steps for solver and display
dt=0.1;
% set both initial time step size and maximum step size
% for Runge-Kutta solver
options=odeset('InitialStep',dt,'MaxStep',dt);
% set time span to be generated from Runge-Kutta solver
% from 0 sec to 50 sec with 0.1 sec time step
td=[0:dt:50];
% Solve differential equation with Runge-Kutta solver
[t,x]=ode45(@(t,X)MSD(t,X,m,F0,i),td,[x0;v0],options);
% Extract only particle position trajectory
O=[x(:,1)];
end
function dX=MSD(t,X,m,F0,i)
%
% With two 1st order diffeential equations,
% obtain the derivative of each variables
% (position and velocity of the particle)
%
% t: current time
% X: matrix for state variables
% The first column : the particle position
% The second column : the particle velocity
% m,b,k,F0,w: parameters for the system
%
% Apply two 1st order differential equations
% dx(n+1)/dt<-v(n)
dX(1,1)=X(2,1);
% dv(n+1)/dt<-1/m*(F0sin(wt)-bv(n)-kx(n))
dX(2,1)=(1/m)*((F0*i)/0.1)+X(2,1);
end 