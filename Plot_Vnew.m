%% Plots
plot_numel = numel(Ftotal);
% curr1=curr1(1:plot_numel);
% curr2=curr2(1:plot_numel);
x1=x1(1:plot_numel);
plot_t=t(1:plot_numel);
x1=x1(:);
x2=x2(1:plot_numel);
x2=x2(:);
% curr1=curr1(:);
% curr2=curr2(:);
% current= cat(2,curr1,curr2);

save('data.mat');
% 
% figure(1)
% hold on 
% plot(x1,curr1)
% plot(x1,curr2)
% title('Current')
% xlabel('Position')
% ylabel('Current')
% legend('Current1', 'Current2')
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% figure(2)
% hold on 
% plot(t,curr1)
% plot(t,curr2)
% title('Current')
% xlabel('Time')
% ylabel('Current')
% legend('Current1', 'Current2')
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)

% figure(3)
% subplot(2,1,1)
% plot(t,curr1)
% ylabel('Current1')
% subplot(2,1,2)
% plot(t,curr2)
% xlabel('Time')
% ylabel('Current2')
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)

% figure(4)
% plot(t,Ftotal)
% title('Force (mN)')
% xlabel('Time')
% ylabel('Force')
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
figure(5)
plot(plot_t,x1)
title('Position')
xlabel('Time')
ylabel('Porsition')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% figure(6)
% plot(t,x2)
% title('Velocity')
% xlabel('Time')
% ylabel('velocity')
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)

figure(7)
% subplot(3,1,1)
% hold on 
% plot(plot_t,curr1)
% plot(plot_t,curr2)
% ylabel('Current')
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
subplot(3,1,2)
plot(plot_t,Ftotal)
ylabel('Force')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
subplot(3,1,3)
plot(plot_t,x1)
ylabel('Position')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)

figure(8)
plot(x1,Ftotal)
title('Force (mN)')
xlabel('Position')
ylabel('Force')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% figure(9)
% plot(t,Ftotal)
% title('Force (mN)')
% xlabel('Time')
% ylabel('Force')
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 14)
% Ftotal = Ftotal(:);
% posforce= cat(2,x1,Ftotal);
% writematrix(posforce, 'PosForce.txt')
% poscurr= cat(2,x1,curr1);
% writematrix(poscurr,'posscurr.txt')
% ds = datastore('posscurr.txt')