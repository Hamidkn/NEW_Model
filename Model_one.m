clc
clear
close all

tic
%%
dt=1e-5;
t=0:dt:1;
u=numel(t);
g = 9.81;
k = 0.213;
R=100;
L=1.3e-3;
m=(0.0075 + 5 * 0.006);
step=5*1e3;
Sstep= 4*step;
x1(1)=zeros();
x2(1)=zeros();
curr1(1)=0;
curr2(1)=0;
Ftotal(1)=0;
stop = [0.0050,0.0100,0.0150,0.0200];
stop_time1 = 0:dt:0.3;
stop_time2 = 0.3:dt:0.6;
phase1_end_time = 30061;
phase2_end_time = 60122;
%%
for i=1:step
    x1(i+1)= dt*x2(i) + x1(i);
    % Phase1
    if (x1(i) < 0.005)
        phase = 1;
%          curr = curr1(i);
%          curr2(i)=0;
%          v1=18;
%          v2=0;
         Fx1 = 0.0002706*sin(3143*x1(i)+1.57) + 3.811e-05*sin(9430*x1(i)-1.574)...
             + 1.294e-05*sin(1.572e+04*x1(i)+1.566) + 5.519e-06*sin(2.2e+04*x1(i)-1.577);
         Ftotal(i) = (Fx1);
         % Phase2
%          if Ftotal(i) <= 0
% %              Ftotal(i)=0;
%              break
%          end
    end
%     curr1(i+1)= dt*(((-R/L)*curr1(i)) + v1*(1/L)) + curr1(i);
%     curr2(i+1)= dt*(((-R/L)*curr2(i)) + v2*(1/L)) + curr2(i);
    x2(i+1)= (dt*Ftotal(i))/m + x2(i);
end

% for j=phase1_end_time:phase2_end_time
%     x1(j+1)= dt*x2(j) + x1(j);
%     % Phase2
%         if ( x1(j) >= 0.005 && x1(j) < 0.01 )
%          phase = 2;
%          curr = curr2(j);
%          curr1(j)=0;
%          v2=18;
%          v1=0;
%          Fx2 = 0.0002707*sin(3142*x1(j)-2.138e-12) + 3.815e-05*sin(9425*x1(j)-6.356e-12)...
%              + 1.295e-05*sin(1.571e+04*x1(j)-1.05e-11) + 5.544e-06*sin(2.199e+04*x1(j)-1.471e-11);
%          Ftotal(j) = (Fx2 );
% %          if Ftotal(j) <= 0
% %              break
% %          end
%         end
%     curr1(j+1)= dt*(((-R/L)*curr1(j)) + v1*(1/L)) + curr1(j);
%     curr2(j+1)= dt*(((-R/L)*curr2(j)) + v2*(1/L)) + curr2(j);
%     x2(j+1)= (dt*Ftotal(j)*((curr)/0.1))/m + x2(j);
% end


% for i=1:numel(x1)
%     if (x1(i) < 0.005)
%         xx1(i) = x1(i);
%     end
% end
% 
Plot_Vnew

toc
